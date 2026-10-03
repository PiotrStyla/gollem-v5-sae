# GoLLeM-v5 - SAE (interpretowalnosc modelu)

Slowniczek "mysli" dla **[GoLLeM-v5-128M-Muon-v1](https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1)**:
16 TopK sparse autoencoderow na rezydualach modelu + lokalne demo `probe / find / steer`.
Model sam **nie zostal zmieniony** (wagi bit-po-bicie identyczne) - to narzedzie do patrzenia
do srodka i pociskania wybranymi "pomyslami" (steering).

| Artefakt | Gdzie |
|---|---|
| Wagi SAE (16 warstw, 577 MB) | https://huggingface.co/PiotrSty/gollem-v5-128m-sae-res-v5 |
| Model bazowy | https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1 |
| Wpis na Neuronpedia | https://www.neuronpedia.org/gollem-v5-128m-muon-v1 |
| Handover dla wlasciciela modelu | https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1/discussions/1 |

## Co zrobione (2026-10-02/03)

1. **Rejestracja modelu w Neuronpedia** (API `POST /api/model/new`) - id `gollem-v5-128m-muon-v1`,
   16 warstw, powiazanie z repo HF. Wpis UNLISTED (link-only), `neuronsPerLayer` rosnie po
   akceptacji SAE przez zespol Neuronpedia (formularz SAE Upload - patrz handover).
2. **Trening 16 TopK SAE** - noc na CPU (i7-8650U, 8 h, 93.1 tok/s): rezydualy post-block,
   768 -> 6144 (8x), k=50, 2.68M tokenow WikiText (z korpusu 138M tokenow przygotowanego
   tokenizatorem modelu, 12 288 vocab). 16 SAE trenowanych rownolegle z jednego forwardu modela.
3. **Demo lokalne** (`sae/demo.py`) - pelna petla interpretowalnosci, patrz nizej.

## Wyniki treningu

FVU (fraction of variance unexplained, nizsze = lepiej; srednie **0.16** = ~84% wariancji
rezyduali wyjasnionej), martwe featury ~0.02% (ok. 1 z 6144 na warstwe):

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| FVU | .079 | .076 | .132 | .072 | .073 | .099 | .121 | .214 |

| layer | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| FVU | .199 | .202 | .196 | .206 | .207 | .223 | .233 | .238 |

Pelne metryki (co 20 krokow): `results/metrics.jsonl`. Wagi i tabela sha256: repo HF.

## Wyniki demo (warstwa 8)

- `probe` na zdaniu "The capital of France is Paris..." - per-token swieczace featury,
  od razu widac struktury: `f4978` na *The/the*, `f420` na kropce, `f1188` na *The* otwierajacym zdanie.
- `find` (147k tokenow korpusu) - semantyka potwierdzona twardo:
  - **f4978 = "the"**: 8 280 aktywacji, top-6 kontekstow to wszystkie tokeny `the`
  - **f420 = interpunkcja**: top aktywacje `.` `.` `.` `,` `-`
- `steer` (pociskanie kierunkiem `W_dec[f]` w warstwie 8):
  - waga 15 / 200: brak efektu (naturalne aktywacje maja rzedu ~1000)
  - **waga 600: model zmienia tekst** - `BASE: ...The capital of France is Paris. / The capital`
    vs `STEERED: ... History  The city of Paris was founded` (artystyl wikipedyjny)

## Jak uruchomic

```bash
# 1) wagi modelu + kod (z publicznych repo HF)
curl -o model/model.safetensors https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1/resolve/main/model.safetensors
curl -o model/config.json       https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1/resolve/main/config.json
curl -o model/tokenizer.json    https://huggingface.co/Maggio33/GoLLeM-v5-128M-Muon-v1/resolve/main/tokenizer.json
curl -o model/train_gpt_ref.py  https://huggingface.co/SlayerLab/gollem-v5-ckpts/resolve/963440ce6ab4ada7e95da4c1faa28ebb00c082d4/train_gpt_ref.py

# 2) wagi SAE (16 plikow) do out/
#    https://huggingface.co/PiotrSty/gollem-v5-128m-sae-res-v5/tree/main

# 3) demo (tekst przez argv TYLKO ASCII - Windows argv niszczy nie-ASCII)
cd sae
python demo.py --layer 8 probe --text "The capital of France is Paris."
python demo.py --layer 8 find  --feature 4978
python demo.py --layer 8 steer --text "The capital of France is" --feature 4978 --weight 600
```

Zaleznosci: `torch`, `safetensors`, `tokenizers`, `numpy`, `pyarrow` (tylko `prep_data.py`).

## Struktura

```
sae/topk_sae.py     modul SAE (TopK, untied, unit-norm decoder)
sae/prep_data.py    parquet wikitext -> tokens.bin (uint16) tokenizatorem modelu
sae/train_sae.py    trening 16 SAE rownolegle + checkpointy + metryki
sae/pack_release.py skladanie release pod HF (README z tabela sha256)
sae/demo.py         probe / find / steer
results/            metryki treningu (metrics.jsonl, sae_config.json)
```

Pominiete w repo (za duze / do pobrania): wagi modelu i SAE, korpus, checkpointy posrednie.

## Nastepne kroki

Checklist dla wlasciciela modelu (formularz SAE Upload, PUBLIC listing, inference):
https://huggingface.co/PiotrSty/gollem-v5-128m-sae-res-v5#next-steps-handover-checklist-for-the-model-owner

Quality note: 2.68M tokenow to first-pass (pelne SAE jedza 50M+); na GPU dokrecamy ~20x danych
dla czystszych featurow - inwestycja w jakosc, nie w dzialanie.
