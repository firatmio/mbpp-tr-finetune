"""Qwen3-1.7B uzerinde mbpp-tr ile LoRA fine-tune (Colab T4 hedefli).

Veri bolunmesi (task_id araliklari MBPP'de ayrik):
  egitim   : full/train       (374, task 601-974)  -- referansi kendi testini gecmeyenler atilir
  eval loss: full/validation  (90,  task 511-600)
  test     : sanitized/test   (257, task 11-510)   -- evaluate.py ile, burada dokunulmaz

Loss sadece cevap (kod) tokenlarinda hesaplanir; prompt tokenlari -100 ile maskelenir.

    python scripts/train_lora.py --output_dir outputs/lora
"""

import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from common import BASE_MODEL_ID, DATASET_ID, build_prompt_text, build_target, normalize_code, test_setup
from execution import run_many


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="outputs/lora")
    ap.add_argument("--model_id", default=BASE_MODEL_ID)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=-1, help="Duman testi icin; -1 = epochs kullan")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16"])
    return ap.parse_args()


def drop_broken_references(ds):
    results = run_many(
        [{"code": normalize_code(ex["code"]), "tests": ex["test_list"], "setup": test_setup(ex)} for ex in ds]
    )
    bad = {ex["task_id"] for ex, r in zip(ds, results) if not r["passed"]}
    if bad:
        print(f"Referansi testini gecmeyen {len(bad)} ornek egitimden cikarildi: {sorted(bad)}")
    return ds.filter(lambda ex: ex["task_id"] not in bad), sorted(bad)


def tokenize_fn(tokenizer, max_len):
    def fn(ex):
        prompt_ids = tokenizer(build_prompt_text(tokenizer, ex), add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(build_target(ex) + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_len]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return fn


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = load_dataset(DATASET_ID, "full", split="train")
    val_ds = load_dataset(DATASET_ID, "full", split="validation")
    test_ids = set(load_dataset(DATASET_ID, "sanitized", split="test")["task_id"])
    overlap = test_ids & (set(train_ds["task_id"]) | set(val_ds["task_id"]))
    assert not overlap, f"Test setiyle cakisan task_id'ler var: {sorted(overlap)}"

    train_ds, dropped = drop_broken_references(train_ds)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # Qwen3'te eos_token = <|im_end|>; chat sablonundaki asistan turu da bununla biter.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    fn = tokenize_fn(tokenizer, args.max_len)
    train_tok = train_ds.map(fn, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(fn, remove_columns=val_ds.column_names)
    print(f"train={len(train_tok)} val={len(val_tok)} max_tokens={max(len(x) for x in train_tok['input_ids'])}")

    # T4 bf16 desteklemez -> fp16. LoRA agirliklari peft tarafindan float32 tutulur.
    if args.precision == "auto":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else:
        use_bf16 = args.precision == "bf16"
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # warmup_ratio transformers 5.17'de kaldirildi; adim sayisini kendimiz hesapliyoruz (her surumde calisir).
    steps_per_epoch = math.ceil(len(train_tok) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = max(1, round(0.05 * total_steps))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=not use_bf16,
        bf16=use_bf16,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Model card icin seffaf kayit
    with open(os.path.join(args.output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.model_id,
                "dataset": DATASET_ID,
                "train_split": "full/train",
                "val_split": "full/validation",
                "train_examples": len(train_tok),
                "dropped_task_ids": dropped,
                "hyperparameters": {**vars(args), "warmup_steps": warmup_steps, "total_steps": total_steps},
                "log_history": trainer.state.log_history,
                "best_checkpoint": trainer.state.best_model_checkpoint,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Adapter kaydedildi: {final_dir}")


if __name__ == "__main__":
    main()
