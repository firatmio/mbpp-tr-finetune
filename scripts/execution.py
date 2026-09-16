"""Uretilen kodu test_list ile ayri bir surecte, timeout ile calistirir.

Not: Bu gercek bir sandbox degil (dosya sistemi/ag izolasyonu yok). Colab gibi atilabilir
bir ortamda calistirilmasi onerilir.
"""

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = 30.0) -> dict:
    program = "\n\n".join([setup, code, "\n".join(tests)]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout", "error": f"> {timeout}s"}
    if proc.returncode == 0:
        return {"passed": True, "status": "passed", "error": ""}
    err_lines = proc.stderr.strip().splitlines()
    last = err_lines[-1] if err_lines else ""
    status = "assertion_error" if last.startswith("AssertionError") else "error"
    return {"passed": False, "status": status, "error": last[:300]}


def run_many(jobs: list[dict], timeout: float = 30.0, workers: int | None = None) -> list[dict]:
    """jobs: [{"code", "tests", "setup"}] -> ayni sirada sonuc listesi.

    Paralel is sayisi CPU sayisini gecmez: Colab'da 2 vCPU var, fazlasi yavas testleri
    (orn. task 123, tek basina ~2-3 sn) CPU yarisiyla yapay olarak timeout'a dusurur.
    """
    workers = workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(lambda j: run_tests(j["code"], j["tests"], j.get("setup", ""), timeout), jobs)
        )
