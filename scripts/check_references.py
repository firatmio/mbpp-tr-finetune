"""Eval altyapisinin kendisini dogrular: veri setindeki referans `code` kendi testlerini gecmeli.

Model sonuclarina guvenmeden once harness'e guvenmek gerekiyor ("iddia degil kanit").
GPU gerektirmez.

    python scripts/check_references.py --config sanitized --split test
"""

import argparse
import sys

from datasets import load_dataset

from common import DATASET_ID, normalize_code, test_setup
from execution import run_many


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sanitized", choices=["sanitized", "full"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds = load_dataset(DATASET_ID, args.config, split=args.split)
    jobs = [
        {"code": normalize_code(ex["code"]), "tests": ex["test_list"], "setup": test_setup(ex)}
        for ex in ds
    ]
    results = run_many(jobs)
    failed = [(ex["task_id"], r) for ex, r in zip(ds, results) if not r["passed"]]
    print(f"{args.config}/{args.split}: {len(ds) - len(failed)}/{len(ds)} referans cozum gecti")
    for task_id, r in failed:
        print(f"  task {task_id}: {r['status']} {r['error']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
