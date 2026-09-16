"""Egitim ve degerlendirmede ortak kullanilan prompt formati ve kod ayiklama.

Prompt formati egitimde ve eval'de BIREBIR ayni olmali; bu yuzden tek yerde tutuluyor.
"""

import re

DATASET_ID = "firatmio/mbpp-tr"
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"

# Modele fonksiyon adini/imzasini gostermek icin sadece ILK test veriliyor.
# Degerlendirme ise tum test_list uzerinden yapiliyor (model card'da belirtilecek).
# Son satirdaki cikti talimati base modelin uzun aciklama yazip max_new_tokens'a takilmasini azaltir;
# boylece baseline bicim yuzunden degil, kod yuzunden basarisiz olur (adil karsilastirma).
USER_TEMPLATE = (
    "{prompt}\n\nKodunuz şu testi geçmeli:\n{first_test}\n\n"
    "Yalnızca Python kodunu ```python bloğu içinde verin."
)


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


_FENCE_OPEN_RE = re.compile(r"```[ \t]*(?:python|py|Python)?[ \t]*\n")
_DEF_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s", re.MULTILINE)


def _code_blocks(text: str) -> list[str]:
    """Tum ``` bloklarini sirayla dondurur; kapanmamis son blok da dahil (max_new_tokens kesmesi)."""
    blocks, pos = [], 0
    while (m := _FENCE_OPEN_RE.search(text, pos)) is not None:
        end = text.find("```", m.end())
        if end == -1:
            blocks.append(text[m.end() :])
            break
        blocks.append(text[m.end() : end])
        pos = end + 3
    return [b.strip() for b in blocks]


def extract_code(text: str) -> str:
    """Icinde def/class olan ilk kod blogunu secer.

    Modeller bazen once sadece `assert` orneklerinden olusan bir blok, sonra asil fonksiyonu yazar;
    ilk blogu almak fonksiyonu hic calistirmamak demek (baseline'da task 63 bu yuzden dusmustu).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = _code_blocks(text)
    if not blocks:
        return text.strip()
    for block in blocks:
        if _DEF_RE.search(block):
            return block
    return blocks[0]


def test_setup(example: dict) -> str:
    """sanitized config'te `test_imports` (liste), full config'te `test_setup_code` (str) var."""
    parts = list(example.get("test_imports") or [])
    if example.get("test_setup_code"):
        parts.append(example["test_setup_code"])
    return "\n".join(parts)
