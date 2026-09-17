---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
datasets:
- firatmio/mbpp-tr
language:
- tr
pipeline_tag: text-generation
tags:
- code
- python
- turkish
- lora
- mbpp
- negative-result
---

# Qwen3-1.7B × MBPP-TR: LoRA fine-tune deneyleri

[firatmio/mbpp-tr](https://huggingface.co/datasets/firatmio/mbpp-tr) üzerinde `Qwen/Qwen3-1.7B` için
LoRA fine-tune denemeleri ve **üretilen kodun testleriyle gerçekten çalıştırılarak** ölçülen sonuçları.

**Kısa sonuç: Fine-tune base modeli geçemedi.** MBPP referans koduyla eğitim modeli kötüleştirdi;
modelin kendi testle doğrulanmış cevaplarıyla (RFT) eğitim ise etkisiz kaldı. Bu repo bir "hazır model"den
çok, dürüstçe raporlanmış bir deney kaydıdır.

*LoRA fine-tuning experiments of Qwen3-1.7B on the Turkish MBPP translation, evaluated by executing the
generated code against its tests. **The fine-tunes did not beat the base model**: training on MBPP reference
solutions hurt (55.6% → 49.0% pass@1), and rejection-sampling fine-tuning on the model's own greedy
solutions had no effect (54.9%). Published as a transparent experiment report rather than an improved model.*

## Sonuçlar

pass@1, greedy decoding, `max_new_tokens=1024`, Colab T4 (fp16).

| Model | sanitized/test (257) | full/validation (90) |
|---|---|---|
| Base `Qwen/Qwen3-1.7B` | **55.6%** (143) | 33.3% (30) |
| Deney 1: LoRA, hedef = MBPP referans kodu | 49.0% (126) | **41.1%** (37) |
| Deney 2: LoRA, RFT (modelin testi geçen greedy cevapları) | 54.9% (141) | 34.4% (31) |

Ana metrik `sanitized/test`. `full/validation` ek bilgi olarak verilmiştir (neden ayrı yorumlanması gerektiği
aşağıda).

### Türkçe çevirinin maliyeti (tanı)

Aynı 257 görev, aynı şablonun İngilizce karşılığıyla (`prompt_en`). Donanım farkı greedy çıktıları
değiştirdiği için iki dil aynı makinede ölçüldü (RTX 4060, bf16).

| Prompt dili | sanitized/test (257) | Açıklamalı cevap oranı |
|---|---|---|
| Türkçe | 53.7% (138) | %58 |
| İngilizce | **57.6%** (148) | %23 |

- İngilizce +3.9 puan (10 görev). Görev bazında: 116 görev iki dilde de geçti, 22 yalnızca Türkçe'de, 32 yalnızca
  İngilizce'de. Fark istatistiksel olarak anlamlı değil (exact McNemar p = 0.22).
- **Türkçe'nin maliyeti küçük; asıl sınır dil değil, modelin kodlama becerisi.** Fine-tune'un kapatabileceği
  bir Türkçe açığı zaten küçüktü; bu, deneylerin base modeli geçememesiyle tutarlı.
- Talimat aynı olmasına rağmen model Türkçe prompt'ta çok daha sık açıklama yazıyor.
- Yerel Türkçe ölçüm (53.7%) ile Colab T4 ölçümü (55.6%) 257 görevin 250'sinde aynı sonucu veriyor; fark
  donanıma bağlı greedy farklılıklarından geliyor. Yukarıdaki ana tablo T4 ölçümleridir.

## Bulgular

**1. MBPP referans koduyla eğitim stili bozdu.** Model referans çözümlerin stilini kopyaladı (2 boşluk
girinti, `res`/`test_list` gibi isimler, yorumsuz kod). Base modelin açıklamalı/yorumlu cevap oranı %58'den
%0'a düştü; yorumlarda adım adım düşünme davranışı kayboldu. Bazı cevaplar dejenere oldu (değerleri tek tek
sabit yazan `if n == 10: return 10` zincirleri). Test setinde 22 görev kazanıldı, 39 görev kaybedildi.

**2. `full` ve `sanitized` aynı görevler için farklı prompt'lar içeriyor.** `sanitized/test`'teki 257
görevin 123'ünün prompt'u, 66'sının testleri `full`'dakinden farklı. `full` açıklamaları kısa ve belirsiz,
fonksiyonun ne döndüreceğini tahmin etmek için MBPP'ye özgü alışkanlıkları bilmeyi gerektiriyor.
`full/train` ile eğitilen Deney 1 bu alışkanlıkları ezberledi: aynı türdeki `full/validation`'da +7.8 puan,
netleştirilmiş `sanitized/test`'te −6.6 puan. Bu yüzden `full/validation`, `sanitized/test` için geçerli bir
model seçim seti değil.

**3. Kendi greedy çıktılarıyla RFT öğrenme sinyali taşımıyor.** Deney 2'nin hedefleri modelin zaten
ürettiği cevaplardı; ilk train loss 0.09 idi. Model değişmedi ama Deney 1'deki bozulma da oluşmadı
(referans kod üzerindeki validation loss 3.49; Deney 1'de 0.62). RFT'nin kazancı normalde greedy'nin
çözemediği görevlerde örnekleme ile bulunan doğru çözümlerden gelir; bu deneyde o adım yapılmadı.

## Değerlendirme yöntemi

- **Prompt:** Türkçe görev açıklaması + fonksiyon adını göstermek için **yalnızca ilk test** + "Yalnızca Python
  kodunu \`\`\`python bloğu içinde verin." Değerlendirme `test_list`'teki **tüm** testlerle yapılır.
  Qwen3 chat şablonu `enable_thinking=False` ile.
- **Kod ayıklama:** İçinde `def`/`class` geçen ilk kod bloğu.
- **Çalıştırma:** Her çözüm ayrı bir Python sürecinde, 30 sn zaman sınırıyla. Tüm referans çözümlerin kendi
  testlerini geçtiği önceden doğrulandı (`sanitized/test` 257/257, `full/validation` 90/90).
- **Düzeltilen ilk ölçüm:** Base modelin ilk ölçümü 47.5% çıktı. Kod ayıklama ilk bloğu alıyordu (model bazen
  önce `assert` örneği, sonra fonksiyonu yazıyor) ve 512 token sınırı 16 cevabı kesiyordu. Düzeltme sonrası
  55.6%; tüm karşılaştırmalar düzeltilmiş hat ile yapıldı.

## Eğitim ayarları

| | Deney 1 | Deney 2 (RFT) |
|---|---|---|
| Veri | `full/train`, 373 görev (referansı testini geçmeyen task 927 çıkarıldı) | `full/train` üzerinde base modelin greedy cevapları: 374 görevden testi geçen **151** |
| Hedef | MBPP referans kodu | Modelin kendi cevabı (açıklamalar dahil) |
| LoRA | r=16, alpha=32, dropout=0.05, tüm attention + MLP projeksiyonları | aynı |
| Optimizasyon | lr 2e-4, cosine, warmup %5, efektif batch 16, 3 epoch | lr 1e-4, 2 epoch, diğerleri aynı |
| Epoch seçimi | `full/validation` eval_loss ile (epoch 2) | Seçim yok, son epoch |
| Süre | 23 dk (T4) | 9.3 dk (T4) |

Loss yalnızca cevap tokenlarında hesaplandı. Deney 2'nin ayarları sonuçlar görülmeden önce git geçmişinde
sabitlendi; test setinde ayar veya epoch seçimi yapılmadı.

## Repo yapısı

| Klasör | İçerik |
|---|---|
| `exp1-reference/` | Deney 1 adapter'ı, `train_summary.json` (log geçmişi dahil) |
| `exp2-rft/` | Deney 2 adapter'ı, `train_summary.json`, `rft_train.jsonl` (151 testle doğrulanmış hedef) |
| `eval/test/`, `eval/val/` | Her model için `summary.json` ve görev bazlı `samples.jsonl` (üretilen cevap, ayıklanan kod, test sonucu) |
| `eval/lang/tr`, `eval/lang/en` | Türkçe/İngilizce prompt tanısı (RTX 4060, bf16) |

Not: `eval/val/exp1/summary.json` içindeki `adapter` alanı repo kökünü gösterir; ölçüm, Deney 1 adapter'ı
`exp1-reference/` altına taşınmadan önce yapıldı. Ağırlıklar aynıdır.

Kök dizinde bilerek adapter yok: iki adapter da base modeli geçmediği için bu repo "doğrudan yüklenecek model"
olarak değil, deney kaydı olarak düzenlendi.

## Kullanım

Türkçe soru → Python kodu için **base modeli doğrudan kullanmanız önerilir**; yukarıdaki sonuçlara göre
adapter'lar ek fayda sağlamıyor. Deneyleri yeniden üretmek veya karşılaştırmak için:

```python
import os
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Qwen/Qwen3-1.7B"
experiment = "exp2-rft"  # veya "exp1-reference"
repo_dir = snapshot_download("firatmio/qwen3-1.7b-mbpp-tr-lora", allow_patterns=[f"{experiment}/*"])

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, os.path.join(repo_dir, experiment))

messages = [{"role": "user", "content": (
    "Verilen bir string'de ilk tekrarlanan karakteri bulmak için bir Python fonksiyonu yazın.\n\n"
    "Kodunuz şu testi geçmeli:\nassert first_repeated_char(\"abcabc\") == \"a\"\n\n"
    "Yalnızca Python kodunu ```python bloğu içinde verin."
)}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Adapter'lar önce `snapshot_download` ile indirilip yerel yoldan yükleniyor. `PeftModel.from_pretrained(..., subfolder=...)`
Windows'ta çalışmıyor (PEFT 0.21, Hub dosya yolunu `os.path.join` ile ters eğik çizgiyle kuruyor); yukarıdaki yol
her işletim sisteminde çalışır.

Değerlendirmeyi tekrar çalıştırmak için GitHub repo'sundaki `scripts/evaluate.py` kullanılabilir.

## Sınırlamalar

- Test setleri küçük: 257 görevde 1 görev ≈ 0.4 puan; ±2–3 görevlik farklar gürültü sınırında.
- Tüm sonuçlar tek bir greedy çalıştırma; farklı donanımda (ör. bf16 GPU) greedy çıktılar ve skorlar biraz
  değişebilir.
- Çalıştırma ortamı gerçek bir sandbox değildir; üretilen kodu güvenilmeyen bir makinede çalıştırmayın.
- Qwen3'ün thinking modu değerlendirilmedi.

## Kaynaklar ve lisans

- Kod ve deney kaydı: https://github.com/firatmio/mbpp-tr-finetune (`EXPERIMENTS.md`)
- Veri seti: [firatmio/mbpp-tr](https://huggingface.co/datasets/firatmio/mbpp-tr) (CC-BY-4.0), MBPP'nin
  Türkçe çevirisi ([Austin et al., 2021](https://arxiv.org/abs/2108.07732))
- Base model: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0)
- Adapter ağırlıkları Apache 2.0; eğitim verisi CC-BY-4.0 koşullarına tabidir.
