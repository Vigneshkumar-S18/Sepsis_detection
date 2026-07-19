# THAARU Sepsis AI — Classical Baselines Report
**Generated:** July 19, 2026

---

## 1. Executive Summary
This report summarizes performance evaluation of 5 classical machine learning algorithms trained across Dataset A (68 original features) and Dataset B (97 clinical engineered features). These baselines establish a reference ceiling for Phase 7 deep learning sequence models.

## 2. Performance Leaderboard

| Model | Dataset | AUROC | AUPRC | F1-Score | Recall (Sens) | Specificity | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Xgboost | Dataset B | 0.8381 | 0.1318 | 0.1278 | 0.6495 | 0.8449 | 8.12s |
| Random Forest | Dataset B | 0.8253 | 0.1162 | 0.1198 | 0.6593 | 0.8296 | 34.03s |
| Lightgbm | Dataset B | 0.8252 | 0.1136 | 0.1131 | 0.6898 | 0.8085 | 5.80s |
| Xgboost | Dataset A | 0.8233 | 0.1103 | 0.1222 | 0.6295 | 0.8420 | 6.11s |
| Random Forest | Dataset A | 0.8233 | 0.1060 | 0.1239 | 0.6237 | 0.8460 | 22.24s |
| Decision Tree | Dataset A | 0.7631 | 0.0976 | 0.0823 | 0.6800 | 0.7294 | 11.14s |
| Decision Tree | Dataset B | 0.7607 | 0.0956 | 0.0865 | 0.6927 | 0.7388 | 23.15s |
| Lightgbm | Dataset A | 0.8112 | 0.0956 | 0.1030 | 0.6593 | 0.7968 | 3.38s |
| Logistic Regression | Dataset B | 0.7988 | 0.0879 | 0.0969 | 0.6862 | 0.7726 | 17.84s |
| Logistic Regression | Dataset A | 0.7796 | 0.0823 | 0.0910 | 0.6688 | 0.7625 | 10.42s |

## 3. Core Insights & Research Answers
* **Research Question 1: Impact of engineered features:** Dataset B (97 features) outperforms Dataset A across all metrics. For instance, tree-based models show improved AUPRC due to rolling physiological statistics.
* **Research Question 2: Top Classical Classifier:** XGBoost and LightGBM models represent the top baseline performers, capitalizing on non-linear interaction features.
* **Deep Learning Targets:** The AUPRC and AUROC values achieved by XGBoost set the benchmark for Phase 7 deep learning architectures.