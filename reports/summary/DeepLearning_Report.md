# THAARU Sepsis AI — Deep Learning Evaluation Report
**Generated:** July 28, 2026

---

## 1. Executive Summary
This report documents Phase 7: Deep Temporal Learning. We benchmarked 4 sequential neural architectures (LSTM, GRU, BiLSTM, and Transformer Encoder) using the baseline `w12_h0` sequence configurations. The best performing architecture was then evaluated across multiple observation window lengths and prediction horizons.

## 2. Deep Learning Leaderboard

| Model | Dataset Config | Window | Horizon | AUROC | AUPRC | F1-Score | Recall (Sens) | Specificity | Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LSTM | w12_h0 | 12h | 0h | 0.7450 | 0.0970 | 0.1446 | 0.3221 | 0.9257 | 150,337 |
| GRU | w12_h0 | 12h | 0h | 0.7764 | 0.0947 | 0.1422 | 0.4381 | 0.8880 | 112,769 |
| BiLSTM | w12_h0 | 12h | 0h | 0.7931 | 0.0951 | 0.1351 | 0.5391 | 0.8472 | 333,441 |
| Transformer | w12_h0 | 12h | 0h | 0.7888 | 0.0987 | 0.1029 | 0.7106 | 0.7129 | 92,097 |
| GRU (Selected) | w6_h0 | 6h | 0h | 0.7675 | 0.0828 | 0.1066 | 0.5665 | 0.7884 | 112,769 |
| GRU (Selected) | w24_h0 | 24h | 0h | 0.7922 | 0.1157 | 0.1551 | 0.5480 | 0.8398 | 112,769 |
| GRU (Selected) | w12_h3 | 12h | +3h | 0.7528 | 0.0852 | 0.1368 | 0.4881 | 0.8627 | 112,769 |
| GRU (Selected) | w12_h6 | 12h | +6h | 0.7722 | 0.1018 | 0.1299 | 0.5204 | 0.8341 | 112,769 |
| BiLSTM (Final) | w12_h0 | 12h | 0h | 0.7899 | 0.1238 | 0.1288 | 0.5669 | 0.8284 | 333,441 |
| BiLSTM (Final) | w6_h0 | 6h | 0h | 0.7898 | 0.0936 | 0.1144 | 0.6268 | 0.7821 | 333,441 |
| BiLSTM (Final) | w24_h0 | 24h | 0h | 0.7667 | 0.1040 | 0.1590 | 0.4899 | 0.8643 | 333,441 |
| BiLSTM (Final) | w12_h3 | 12h | +3h | 0.7666 | 0.0951 | 0.1130 | 0.5833 | 0.7875 | 333,441 |
| BiLSTM (Final) | w12_h6 | 12h | +6h | 0.7529 | 0.0795 | 0.1780 | 0.2847 | 0.9511 | 333,441 |

## 3. Comparative Findings
* **Sequence Models vs XGBoost Baseline:** Deep sequence models leverage the multi-hour observation window to track trajectory changes. Direct comparison shows whether temporal representations outperform static feature boosting (XGBoost baseline Test AUROC 0.8381, AUPRC 0.1318).
* **Best Architecture:** Bidirectional LSTM and Transformer models are compared on context retention, representing the most powerful components of our temporal pipeline.