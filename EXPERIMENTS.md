# Deney kaydi

Tum pass@1 degerleri greedy decoding, `max_new_tokens=1024`, uretilen kod `test_list` ile calistirilarak
olculdu. Prompt formati: `scripts/common.py`.

## Deney 0 -- Base model (Qwen/Qwen3-1.7B)

| Set | pass@1 |
|---|---|
| sanitized/test (257) | **55.6%** (143) |
| full/validation (90) | 33.3% (30) |

Not: Ilk test olcumu 47.5% (122) cikti. `extract_code` ilk kod blogunu aliyordu (model bazen once
`assert` ornegi, sonra fonksiyonu yaziyordu) ve 512 token siniri 16 cevabi kesiyordu. Duzeltme sonrasi
55.6% (commit ef12172). Karsilastirmalarda 55.6% kullanilir.

## Deney 1 -- LoRA, hedef = MBPP referans kodu

- Veri: full/train (373; referansi testini gecmeyen task 927 cikarildi)
- r=16, alpha=32, dropout=0.05, lr=2e-4, cosine, efektif batch 16, 3 epoch
- Epoch secimi full/validation eval_loss ile: 0.6495 / **0.6171** / 0.6215 -> epoch 2

| Set | Base | Deney 1 |
|---|---|---|
| sanitized/test (257) | **55.6%** | 49.0% (126) -- kazanilan 22, kaybedilen 39 |
| full/validation (90) | 33.3% | **41.1%** (37) |

Teshis:
- Model MBPP referans stilini kopyaladi (2 bosluk girinti, `res`/`test_list` isimleri, yorumsuz kod).
  Base modelin aciklamali/yorumlu cevap orani %58 -> %0; yorumlarla "dusunme" kayboldu.
  Bazi cevaplar dejenere (degerleri tek tek hardcode eden `if n == 10: return 10` zincirleri).
- `full` ve `sanitized` ayni gorevler icin farkli prompt'lar iceriyor (sanitized/test'te 123/257 prompt,
  66/257 test farkli). `full` prompt'lari belirsiz ve MBPP'ye ozgu donus kurallarini bilmeyi gerektiriyor.
  full/train ile egitim bu kurallari ezberletti: ayni dagilimdaki full/validation'da artis, netlestirilmis
  sanitized/test'te dusus.
- Sonuc: full/validation, sanitized/test icin gecerli bir secim seti degil.

## Deney 2 -- RFT (onceden sabitlenmis ayarlar)

Ayarlar sonuclar gorulmeden, bu commit'te sabitlendi. Test setinde ayar/epoch secimi yapilmayacak;
sanitized/test'te yalnizca bir kez olculecek.

- Veri: `scripts/generate_rft_data.py --out outputs/rft` (base model, full/train, greedy, `--num_samples 0`);
  yalnizca testi gecen ve kod blogu tamamlanmis cevaplar hedef olur.
- Egitim: `scripts/train_lora.py --rft_file outputs/rft/train.jsonl --epochs 2 --lr 1e-4`
  (diger ayarlar Deney 1 ile ayni: r=16, alpha=32, dropout=0.05, cosine, efektif batch 16)
- Degerlendirilecek adapter: son epoch (`final`). eval_loss ile secim yok (referans stiline benzerlik olcer).
- Raporlanacak: sanitized/test pass@1 (ana metrik), full/validation pass@1 (ek bilgi).
