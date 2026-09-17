"""pass@1 (greedy) degerlendirmesi: model kod uretir, kod test_list ile gercekten calistirilir.

Base model (baseline) ve LoRA adapter'li model ayni script, ayni prompt ile olculur.

    python scripts/evaluate.py --out outputs/eval_base                       # baseline
    python scripts/evaluate.py --adapter outputs/lora/final --out outputs/eval_lora
"""

import argparse
import json
import os
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import BASE_MODEL_ID, DATASET_ID, build_prompt_text, extract_code, test_setup
from execution import run_many


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=BASE_MODEL_ID)
    ap.add_argument("--adapter", default=None, help="LoRA adapter klasoru veya HF repo id")
    ap.add_argument("--config", default="sanitized", choices=["sanitized", "full"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="Hizli deneme icin ilk N ornek")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--prompt_lang", default="tr", choices=["tr", "en"], help="en: tani icin prompt_en")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_model(model_id, adapter=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_all(model, tokenizer, prompts, batch_size, max_new_tokens, **gen_kwargs):
    """gen_kwargs verilmezse greedy; ornekleme icin do_sample=True, temperature=... gecilir."""
    gen_kwargs = gen_kwargs or {"do_sample": False}
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )
        new_tokens = gen[:, batch["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outputs


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = load_dataset(DATASET_ID, args.config, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    model, tokenizer = load_model(args.model_id, args.adapter)

    prompts = [build_prompt_text(tokenizer, ex, args.prompt_lang) for ex in ds]
    completions = generate_all(model, tokenizer, prompts, args.batch_size, args.max_new_tokens)

    codes = [extract_code(c) for c in completions]
    results = run_many(
        [{"code": code, "tests": ex["test_list"], "setup": test_setup(ex)} for code, ex in zip(codes, ds)]
    )

    with open(os.path.join(args.out, "samples.jsonl"), "w", encoding="utf-8") as f:
        for ex, comp, code, r in zip(ds, completions, codes, results):
            prompt = ex[f"prompt_{args.prompt_lang}"]
            row = {"task_id": ex["task_id"], "prompt_tr": prompt, "completion": comp, "code": code, **r}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pass = sum(r["passed"] for r in results)
    summary = {
        "model_id": args.model_id,
        "adapter": args.adapter,
        "dataset": f"{DATASET_ID}/{args.config}/{args.split}",
        "prompt_lang": args.prompt_lang,
        "n": len(ds),
        "passed": n_pass,
        "pass@1": round(n_pass / len(ds), 4),
        "status_counts": dict(Counter(r["status"] for r in results)),
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
