# THAARU Sepsis AI — Phase 9 Optimization & Fine-Tuning Report
**Date:** July 19, 2026
**Author:** Advanced Agentic Coding Subagent

---

## 1. Executive Summary
This report documents Phase 9: Hyperparameter Optimization and Model Fine-Tuning. We executed targeted parameter sweeps for the strongest tabular model (XGBoost) and the sequential deep learning model (BiLSTM) using validation AUPRC as the primary objective. Training was run on 20% downsampled training partitions to optimize learning cycles on CPU resources.

---

## 2. XGBoost Tuning Results
* **Best Configuration:** max_depth=4, lr=0.1, n_estimators=100, subsample=1.0, colsample_bytree=0.8
* **Best Validation AUPRC:** 0.0927 (Validation AUROC: 0.8216)

### XGBoost Trial History:
| Trial | Max Depth | Learning Rate | N Estimators | Subsample | Colsample | Val AUPRC | Val AUROC | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 6 | 0.1 | 100 | 1.0 | 1.0 | 0.0829 | 0.8054 | 5.0s |
| 2 | 6 | 0.2 | 100 | 1.0 | 0.8 | 0.0761 | 0.7830 | 2.6s |
| 3 | 4 | 0.05 | 150 | 1.0 | 0.8 | 0.0902 | 0.8231 | 4.0s |
| 4 | 6 | 0.1 | 150 | 0.8 | 0.8 | 0.0762 | 0.8025 | 4.2s |
| 5 | 4 | 0.1 | 100 | 1.0 | 0.8 | 0.0927 | 0.8216 | 1.8s |

---

## 3. BiLSTM Tuning Results
* **Best Configuration:** hidden_size=32, num_layers=1, dropout=0.3, lr=0.002, batch_size=512
* **Best Validation AUPRC:** 0.0924 (Validation AUROC: 0.8188)

### BiLSTM Trial History:
| Trial | Hidden Size | Layers | Dropout | Learning Rate | Batch Size | Val AUPRC | Val AUROC | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 64 | 1 | 0.2 | 0.001 | 1024 | 0.0902 | 0.8159 | 27.8s |
| 2 | 32 | 2 | 0.3 | 0.001 | 1024 | 0.0839 | 0.8158 | 22.6s |
| 3 | 32 | 1 | 0.3 | 0.002 | 512 | 0.0924 | 0.8188 | 13.8s |
| 4 | 128 | 2 | 0.2 | 0.0005 | 512 | 0.0917 | 0.8220 | 114.9s |
| 5 | 32 | 2 | 0.1 | 0.0005 | 1024 | 0.0777 | 0.8098 | 22.4s |

---

## 4. Final Recommendation & Selected Champion
Comparing the optimized results, **XGBoost** achieved a peak validation AUPRC of **0.0927**, while **BiLSTM** reached a validation AUPRC of **0.0924**.
Both models showed marginal performance gains over baseline runs, and confirm the stability of the baseline parameters chosen in Phase 6 and 7.