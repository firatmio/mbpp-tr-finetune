# mbpp-tr-finetune

## Proje amacı

[firatmio/mbpp-tr](https://huggingface.co/datasets/firatmio/mbpp-tr) veri seti uzerinde kucuk bir acik
modeli (Qwen3-1.7B-Instruct) LoRA ile fine-tune ederek "Turkce soru -> Python kodu" gorevine
ozellestirmek. Amac, Firat'in HuggingFace profilini (CV icin) guclendiren uclu bir proje zincirinin
son parcasini tamamlamak: **dataset (mbpp-tr) -> model (bu proje) -> demo (Space, siradaki adim)**.

## Baglam / onceki adimlar

- `firatmio/mbpp-tr` zaten HuggingFace'de yayinda (CC-BY-4.0, MBPP'nin Turkce cevirisi, sanitized+full
  config'leri, satir satir dogrulanmis).
- Kaynak repo: https://github.com/firatmio/mbpp-tr (ceviri/inceleme/dogrulama araclari burada).
- Bu proje ondan ayri, yeni bir repo -- fine-tune'a ozel.

## Base model karari

**Qwen3-1.7B-Instruct** secildi (Qwen2.5-1.5B-Instruct yerine):
- Apache 2.0 lisansli, mbpp-tr'nin CC-BY-4.0'i ile uyumlu bir kombinasyon.
- Coding/LiveCodeBench'te Qwen2.5-1.5B'den belirgin sekilde daha guclu, boyut neredeyse ayni.
- Cok dilli egitilmis, Turkce icin ayrica bir "Turkce base model" aramaya gerek yok.

## Planlanan yontem

- **LoRA** fine-tune (tam model degil) -- adapter agirliklari kucuk, GPU butcesi dusuk kalir
  (Colab T4 free tier hedefleniyor).
- Egitim verisi: `firatmio/mbpp-tr` -> `prompt_tr` (girdi) + `code` (hedef cikti).
- Degerlendirme: uretilen kodun kendi `test_list`'i ile gercekten calistirilmasi (mbpp-tr'deki
  `validate_final.py` mantigina benzer -- "iddia degil kanit" prensibi bu projede de gecerli).

## Yayinlama plani

- LoRA adapter'i (tam merge edilmis model degil) ayri bir HF model repo'suna yuklenecek.
- Model card'inda: base model, veri seti linki, hyperparameter'lar, eval sonuclari, lisans
  (base model + dataset lisanslarina dikkat).

## Klasor yapisi

- `notebooks/` -- Colab'da calistirilacak .ipynb dosyalari
- `scripts/` -- yardimci Python script'leri (eval, upload vb.)
- `data/` -- gerekirse ara veri dosyalari (buyuk dosyalar burada tutulmaz, HF Hub'dan cekilir)
- `outputs/` -- egitim ciktilari, checkpoint'ler (buyuk olabilir, .gitignore'da)

## Notlar

- Firat'in tercih ettigi dil: Turkce, samimi ama gereksiz uzatmadan.
- Bu proje bir ogrenme/portfoy projesi -- hyperparameter secimlerinin ve sonuclarin model card'da
  seffafca belgelenmesi onemli (mbpp-tr'deki inceleme seffafligi standardi burada da gecerli).
- GPU egitimi bu makinede degil, Colab'da yapilacak; bu repo Colab notebook'unu ve destekleyici
  script'leri barindiriyor, agir hesaplama burada calismiyor.
