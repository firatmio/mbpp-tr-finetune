"""RFT (rejection sampling fine-tuning) verisi: base model train sorularini cozer, sadece testi gecen
ciktilar egitim hedefi olarak tutulur.

Neden: Deney 1'de MBPP referans koduyla egitim modeli kotulestirdi (55.6% -> 49.0%); model referanslarin
eski stilini kopyaladi, yorumla "dusunme" davranisi kayboldu. Modelin kendi, testle dogrulanmis
cevaplariyla egitmek stilini korur; her hedef ornek calistirilarak kanitlanmis olur.

Tur 0 tum gorevlerde greedy uretir; --num_samples > 0 ise sonraki turlar sadece henuz cozulmemis
gorevlerde ornekleme (temperature) ile tekrar dener.

    python scripts/generate_rft_data.py --out outputs/rft
    python scripts/generate_rft_data.py --out outputs/rft --num_samples 2   # daha fazla kapsama
"""

import argparse
import json
import os
from collections import Counter

from datasets import load_dataset

from common import BASE_MODEL_ID, DATASET_ID, build_prompt_text, extract_code, test_setup
from evaluate import generate_all, load_model
from execution import run_many


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=BASE_MODEL_ID)
    ap.add_argument("--out", default="outputs/rft")
    ap.add_argument("--num_samples", type=int, default=0, help="Greedy sonrasi cozulmeyenler icin ornekleme turu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="Duman testi icin ilk N gorev")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def is_complete(completion: str) -> bool:
    """Kod blogu kapanmis olmali; max_new_tokens'a takilip yarim kalan cevap hedef olarak ogretilmemeli."""
    return "```" in completion and completion.count("```") % 2 == 0


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from transformers import set_seed

    set_seed(args.seed)

    ds = load_dataset(DATASET_ID, "full", split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    examples = {ex["task_id"]: ex for ex in ds}

    model, tokenizer = load_model(args.model_id)

    accepted = {}  # task_id -> kayit
    round_stats = []
    for rnd in range(args.num_samples + 1):
        todo = [tid for tid in examples if tid not in accepted]
        if not todo:
            break
        if rnd == 0:
            gen_kwargs = {"do_sample": False}
        else:
            gen_kwargs = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
        print(f"Tur {rnd} ({'greedy' if rnd == 0 else 'sampling'}): {len(todo)} gorev")

        prompts = [build_prompt_text(tokenizer, examples[t]) for t in todo]
        completions = generate_all(model, tokenizer, prompts, args.batch_size, args.max_new_tokens, **gen_kwargs)
        codes = [extract_code(c) for c in completions]
        results = run_many(
            [
                {"code": code, "tests": examples[t]["test_list"], "setup": test_setup(examples[t])}
                for t, code in zip(todo, codes)
            ]
        )

        reasons = Counter()
        for t, comp, code, r in zip(todo, completions, codes, results):
            if not r["passed"]:
                reasons[r["status"]] += 1
            elif not is_complete(comp):
                reasons["passed_but_incomplete"] += 1
            else:
                reasons["accepted"] += 1
                accepted[t] = {"task_id": t, "round": rnd, "target": comp.strip(), "code": code}
        round_stats.append({"round": rnd, "attempted": len(todo), **dict(reasons)})
        print(f"  {dict(reasons)}")
        del completions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = sorted(accepted.values(), key=lambda r: r["task_id"])
    with open(os.path.join(args.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_id": args.model_id,
        "source": f"{DATASET_ID}/full/train",
        "n_tasks": len(examples),
        "n_accepted": len(rows),
        "coverage": round(len(rows) / len(examples), 4),
        "rounds": round_stats,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
