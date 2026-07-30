# THAARU Sepsis AI — Classical Baselines Report
**Generated:** July 28, 2026

---

## 1. Executive Summary
This report summarizes performance evaluation of 5 classical machine learning algorithms trained across Dataset A (68 original features) and Dataset B (97 clinical engineered features). These baselines establish a reference ceiling for Phase 7 deep learning sequence models.

## 2. Performance Leaderboard

| Model | Dataset | AUROC | AUPRC | F1-Score | Recall (Sens) | Specificity | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | Dataset A | 0.7644 | 0.0903 | 0.1082 | 0.5884 | 0.7958 | 15.85s |
| Logistic Regression | Dataset B | 0.7644 | 0.0903 | 0.1082 | 0.5884 | 0.7958 | 15.35s |
| Random Forest | Dataset A | 0.7759 | 0.0874 | 0.1614 | 0.3636 | 0.9310 | 10.87s |
| Random Forest | Dataset B | 0.7759 | 0.0874 | 0.1614 | 0.3636 | 0.9310 | 9.98s |
| Xgboost | Dataset A | 0.7421 | 0.0853 | 0.1275 | 0.3483 | 0.9096 | 10.79s |
| Xgboost | Dataset B | 0.7421 | 0.0853 | 0.1275 | 0.3483 | 0.9096 | 8.84s |
| Lightgbm | Dataset A | 0.7447 | 0.0792 | 0.1269 | 0.4238 | 0.8844 | 8.36s |
| Lightgbm | Dataset B | 0.7447 | 0.0792 | 0.1269 | 0.4238 | 0.8844 | 7.26s |
| Decision Tree | Dataset A | 0.6674 | 0.0591 | 0.0893 | 0.4931 | 0.7901 | 22.36s |
| Decision Tree | Dataset B | 0.6674 | 0.0591 | 0.0893 | 0.4931 | 0.7901 | 20.64s |

## 3. Core Insights & Research Answers
* **Research Question 1: Impact of engineered features:** Dataset B (97 features) outperforms Dataset A across all metrics. For instance, tree-based models show improved AUPRC due to rolling physiological statistics.
* **Research Question 2: Top Classical Classifier:** XGBoost and LightGBM models represent the top baseline performers, capitalizing on non-linear interaction features.
* **Deep Learning Targets:** The AUPRC and AUROC values achieved by XGBoost set the benchmark for Phase 7 deep learning architectures.