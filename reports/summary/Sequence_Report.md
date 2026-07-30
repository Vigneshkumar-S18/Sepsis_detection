# THAARU Sepsis AI — Temporal Sequence Generation Report
**Generated:** July 28, 2026

---

## 1. Experiment Dataset Summary

| Config | Window | Horizon | Train Seqs | Val Seqs | Test Seqs | Train +Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| w6_h0 | 6h | +0h | 250,711 | 54,473 | 53,052 | 2.23% |
| w12_h0 | 12h | +0h | 206,426 | 44,982 | 43,558 | 2.29% |
| w24_h0 | 24h | +0h | 125,533 | 27,602 | 26,181 | 2.91% |
| w12_h3 | 12h | +3h | 184,765 | 40,335 | 38,918 | 2.37% |
| w12_h6 | 12h | +6h | 163,786 | 35,808 | 34,404 | 2.50% |

---

## 2. Per-Configuration Details

### w6_h0 — Minimal observation: can 6h detect sepsis?

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 250,711 | 5,596 | 245,115 | 2.23% | 43.8:1 |
| Validation | 54,473 | 1,224 | 53,249 | 2.25% | 43.5:1 |
| Test | 53,052 | 1,211 | 51,841 | 2.28% | 42.8:1 |

### w12_h0 — Standard 12h observation window

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 206,426 | 4,720 | 201,706 | 2.29% | 42.7:1 |
| Validation | 44,982 | 1,080 | 43,902 | 2.40% | 40.6:1 |
| Test | 43,558 | 1,009 | 42,549 | 2.32% | 42.2:1 |

### w24_h0 — Extended observation: does more history help?

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 125,533 | 3,657 | 121,876 | 2.91% | 33.3:1 |
| Validation | 27,602 | 871 | 26,731 | 3.16% | 30.7:1 |
| Test | 26,181 | 739 | 25,442 | 2.82% | 34.4:1 |

### w12_h3 — Early warning: 3-hour advance prediction

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 184,765 | 4,373 | 180,392 | 2.37% | 41.2:1 |
| Validation | 40,335 | 1,005 | 39,330 | 2.49% | 39.1:1 |
| Test | 38,918 | 924 | 37,994 | 2.37% | 41.1:1 |

### w12_h6 — Early warning: 6-hour advance prediction

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 163,786 | 4,091 | 159,695 | 2.50% | 39.0:1 |
| Validation | 35,808 | 946 | 34,862 | 2.64% | 36.9:1 |
| Test | 34,404 | 857 | 33,547 | 2.49% | 39.1:1 |

---

## 3. Validation

- **w6_h0**: ✅ ALL PASSED
- **w12_h0**: ✅ ALL PASSED
- **w24_h0**: ✅ ALL PASSED
- **w12_h3**: ✅ ALL PASSED
- **w12_h6**: ✅ ALL PASSED