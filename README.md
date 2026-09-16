# mbpp-tr-finetune

[firatmio/mbpp-tr](https://huggingface.co/datasets/firatmio/mbpp-tr) veri seti uzerinde
Qwen3-1.7B-Instruct modelinin LoRA ile fine-tune edilmesi. Turkce soru -> Python kodu uretimi
gorevine ozellestirme.

Bu proje, [mbpp-tr](https://github.com/firatmio/mbpp-tr) dataset projesinin devami:
**dataset -> model (burasi) -> demo (Space, siradaki adim)**.

## Durum

Henuz baslangic asamasinda -- egitim notebook'u hazirlaniyor.

## Plan

- Base model: Qwen3-1.7B-Instruct (Apache 2.0)
- Yontem: LoRA, Google Colab (T4 free tier)
- Veri: `firatmio/mbpp-tr` (`prompt_tr` -> `code`)
- Degerlendirme: uretilen kodun kendi test'leriyle calistirilmasi

Detaylar icin bkz. `CLAUDE.md`.

## Lisans

Bu projenin kodu MIT lisanslidir. Kullanilan dataset (CC-BY-4.0) ve base model (Apache 2.0)
kendi lisanslarini korur.
