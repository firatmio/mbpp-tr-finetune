# mbpp-tr-finetune

[firatmio/mbpp-tr](https://huggingface.co/datasets/firatmio/mbpp-tr) veri seti uzerinde
Qwen3-1.7B modelinin LoRA ile fine-tune edilmesi. Turkce soru -> Python kodu uretimi
gorevine ozellestirme.

Bu proje, [mbpp-tr](https://github.com/firatmio/mbpp-tr) dataset projesinin devami:
**dataset -> model (burasi) -> demo (Space, siradaki adim)**.

## Durum

Egitim/eval kodu hazir ve yerelde duman testinden gecti; tam egitim Colab'da calistirilacak.

## Calistirma

`notebooks/train_colab.ipynb` dosyasini Colab'da (T4) ac ve sirayla calistir. Ya da dogrudan:

```bash
python scripts/check_references.py --config sanitized --split test   # harness dogrulama
python scripts/evaluate.py --out outputs/eval_base                    # baseline
python scripts/train_lora.py --output_dir outputs/lora                # egitim
python scripts/evaluate.py --adapter outputs/lora/final --out outputs/eval_lora
```

## Plan

- Base model: Qwen3-1.7B (Apache 2.0)
- Yontem: LoRA, Google Colab (T4 free tier)
- Veri: `firatmio/mbpp-tr` (`prompt_tr` -> `code`)
- Degerlendirme: `sanitized/test` (257) uzerinde greedy pass@1, uretilen kod kendi test'leriyle calistirilarak

Detaylar icin bkz. `CLAUDE.md`.

## Lisans

Bu projenin kodu MIT lisanslidir. Kullanilan dataset (CC-BY-4.0) ve base model (Apache 2.0)
kendi lisanslarini korur.
