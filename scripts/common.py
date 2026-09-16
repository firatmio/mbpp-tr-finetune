"""Egitim ve degerlendirmede ortak kullanilan prompt formati ve kod ayiklama.

Prompt formati egitimde ve eval'de BIREBIR ayni olmali; bu yuzden tek yerde tutuluyor.
"""

import re

DATASET_ID = "firatmio/mbpp-tr"
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"

# Modele fonksiyon adini/imzasini gostermek icin sadece ILK test veriliyor.
# Degerlendirme ise tum test_list uzerinden yapiliyor (model card'da belirtilecek).
USER_TEMPLATE = "{prompt}\n\nKodunuz şu testi geçmeli:\n{first_test}"


def normalize_code(code: str) -> str:
    return code.replace("\r\n", "\n").strip()


def build_messages(example: dict) -> list[dict]:
    content = USER_TEMPLATE.format(
        prompt=example["prompt_tr"].strip(),
        first_test=example["test_list"][0].strip(),
    )
    return [{"role": "user", "content": content}]


def build_target(example: dict) -> str:
    return f"```python\n{normalize_code(example['code'])}\n```"


def build_prompt_text(tokenizer, example: dict) -> str:
    # Qwen3 hybrid-thinking modeli: enable_thinking=False ile bos <think></think> blogu eklenir,
    # model dogrudan cevaba gecer. Egitim ve eval ayni sablonu kullanir.
    return tokenizer.apply_chat_template(
        build_messages(example),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


_FENCE_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return blocks[0].strip()
    # Kapanmamis blok (max_new_tokens'a takilmis olabilir)
    if "```" in text:
        after = text.split("```", 1)[1]
        return after.split("\n", 1)[1].strip() if "\n" in after else after.strip()
    return text.strip()


def test_setup(example: dict) -> str:
    """sanitized config'te `test_imports` (liste), full config'te `test_setup_code` (str) var."""
    parts = list(example.get("test_imports") or [])
    if example.get("test_setup_code"):
        parts.append(example["test_setup_code"])
    return "\n".join(parts)
