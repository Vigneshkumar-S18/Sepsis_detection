# THAARU Sepsis AI — Temporal Sequence Generation Report
**Generated:** July 19, 2026

---

## 1. Experiment Dataset Summary

| Config | Window | Horizon | Train Seqs | Val Seqs | Test Seqs | Train +Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| w6_h0 | 6h | +0h | 945,261 | 201,222 | 204,047 | 1.87% |
| w12_h0 | 12h | +0h | 776,999 | 165,213 | 167,979 | 1.91% |
| w24_h0 | 24h | +0h | 472,453 | 100,373 | 102,520 | 2.48% |
| w12_h3 | 12h | +3h | 695,059 | 147,719 | 150,382 | 2.00% |
| w12_h6 | 12h | +6h | 615,848 | 130,823 | 133,352 | 2.13% |

---

## 2. Per-Configuration Details

### w6_h0 — Minimal observation: can 6h detect sepsis?

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 945,261 | 17,643 | 927,618 | 1.87% | 52.6:1 |
| Validation | 201,222 | 3,690 | 197,532 | 1.83% | 53.5:1 |
| Test | 204,047 | 3,803 | 200,244 | 1.86% | 52.6:1 |

### w12_h0 — Standard 12h observation window

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 776,999 | 14,827 | 762,172 | 1.91% | 51.4:1 |
| Validation | 165,213 | 3,061 | 162,152 | 1.85% | 53.0:1 |
| Test | 167,979 | 3,221 | 164,758 | 1.92% | 51.1:1 |

### w24_h0 — Extended observation: does more history help?

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 472,453 | 11,715 | 460,738 | 2.48% | 39.3:1 |
| Validation | 100,373 | 2,241 | 98,132 | 2.23% | 43.8:1 |
| Test | 102,520 | 2,576 | 99,944 | 2.51% | 38.8:1 |

### w12_h3 — Early warning: 3-hour advance prediction

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 695,059 | 13,887 | 681,172 | 2.00% | 49.0:1 |
| Validation | 147,719 | 2,821 | 144,898 | 1.91% | 51.4:1 |
| Test | 150,382 | 3,017 | 147,365 | 2.01% | 48.8:1 |

### w12_h6 — Early warning: 6-hour advance prediction

| Split | Sequences | Positive | Negative | +Rate | Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Train | 615,848 | 13,089 | 602,759 | 2.13% | 46.0:1 |
| Validation | 130,823 | 2,585 | 128,238 | 1.98% | 49.6:1 |
| Test | 133,352 | 2,841 | 130,511 | 2.13% | 45.9:1 |

---

## 3. Validation

- **w6_h0**: ✅ ALL PASSED
- **w12_h0**: ✅ ALL PASSED
- **w24_h0**: ✅ ALL PASSED
- **w12_h3**: ✅ ALL PASSED
- **w12_h6**: ✅ ALL PASSED