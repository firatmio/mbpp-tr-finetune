# mbpp-tr-finetune

[firatmio/mbpp-tr](https://huggingface.co/datasets/firatmio/mbpp-tr) veri seti uzerinde Qwen3-1.7B icin
LoRA fine-tune deneyleri. Uretilen kod, gorevin kendi testleriyle gercekten calistirilarak degerlendirildi.

Bu proje, [mbpp-tr](https://github.com/firatmio/mbpp-tr) dataset projesinin devami:
**dataset (mbpp-tr) -> model / deney raporu (burasi)**.

## Sonuc

**Fine-tune base modeli gecemedi.** pass@1, `sanitized/test` (257 gorev), greedy:

| Model | pass@1 |
|---|---|
| Base `Qwen/Qwen3-1.7B` | **55.6%** |
| Deney 1: LoRA, hedef = MBPP referans kodu | 49.0% |
| Deney 2: LoRA, RFT (modelin testi gecen kendi cevaplari) | 54.9% |

- Referans kodla egitim, modele MBPP'nin eski kod stilini kopyalatti ve yorumlarla "dusunme" davranisini
  yok etti.
- Modelin kendi greedy cevaplariyla egitim ogrenme sinyali tasimadi.
- Ayni gorevlerde Ingilizce prompt yalnizca +3.9 puan (anlamli degil): sinir dil degil, modelin kodlama becerisi.

Tum deneyler, ara kararlar ve teshisler: [`EXPERIMENTS.md`](EXPERIMENTS.md).
Adapter'lar, gorev bazli eval ciktilari ve model card:
[firatmio/qwen3-1.7b-mbpp-tr-lora](https://huggingface.co/firatmio/qwen3-1.7b-mbpp-tr-lora).

Proje burada tamamlandi; ayri bir demo (Space) yapilmadi.

## Calistirma

Egitim Colab T4'te yapildi (`notebooks/train_colab.ipynb`: kurulum, baseline, Deney 1). Script'ler:

```bash
python scripts/check_references.py --config sanitized --split test    # harness dogrulama
python scripts/evaluate.py --out outputs/eval_base                     # base model pass@1
python scripts/evaluate.py --prompt_lang en --out outputs/eval_base_en # Ingilizce prompt tanisi

# Deney 1: referans kodla egitim
python scripts/train_lora.py --output_dir outputs/lora
python scripts/evaluate.py --adapter outputs/lora/final --out outputs/eval_lora

# Deney 2: RFT
python scripts/generate_rft_data.py --out outputs/rft
python scripts/train_lora.py --output_dir outputs/lora_rft --rft_file outputs/rft/train.jsonl --epochs 2 --lr 1e-4
python scripts/evaluate.py --adapter outputs/lora_rft/final --out outputs/eval_rft
```

## Lisans

Bu projenin kodu MIT lisanslidir. Kullanilan dataset (CC-BY-4.0) ve base model (Apache 2.0)
kendi lisanslarini korur.
