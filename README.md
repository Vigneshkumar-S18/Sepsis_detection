# 🏥 SEPESDETECTOR: Early Sepsis Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Phase%2011%20Enhanced-brightgreen.svg)]()

> **A real-time Clinical Decision Support System (CDSS) for early sepsis prediction, patient risk stratification, and temporal explainability using sequential deep learning.**

---

## 📌 Executive Summary & Clinical Significance

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. In intensive care units (ICUs), every hour of delayed clinical recognition significantly increases mortality risk. 

**SEPESDETECTOR** is designed to continuously ingest patient vital sign streams, lab results, and temporal trajectories to predict sepsis onset **hours before clinical manifestation**. The system provides real-time early warnings, risk certainty metrics, persistent danger alerts, localized feature attributions (Integrated Gradients), and an interactive glassmorphic clinician workstation.

### 📊 Dataset Overview (PhysioNet Challenge 2019)
* **Cohort Size**: 40,336 unique ICU patients (20,336 Cohort A + 20,000 Cohort B)
* **Total Hourly Observations**: 1,552,210 patient-hours
* **Sepsis Prevalence**: 7.27% patient-level prevalence (1.80% total hourly record prevalence)
* **ICU Length of Stay**: Average 39.01 hours (range: 8 to 336 hours)

---

## 🏗️ System Architecture & Workflow

The framework operates end-to-end from bedside telemetry ingestion to live clinical dashboards:

```text
  +-----------------------+      Ingest      +-----------------------------+
  |  Patient Telemetry    | -------------->  | FastAPI Ingestion Endpoint  |
  |  (HR, Temp, labs...)  |                  | (server.py /patient/update) |
  +-----------------------+                  +-----------------------------+
                                                            |
                                                            v
  +-----------------------+   Read Buffer    +-----------------------------+
  | SQLite Database       | <--------------  | Rolling Window Manager      |
  | (sepsis_live.db)      |                  | (Up to 12h history)         |
  +-----------------------+                  +-----------------------------+
                                                            |
                                                            v
  +-----------------------+   Attributions   +-----------------------------+
  | PyTorch Captum XAI    | <--------------  | Online Preprocessor         |
  | (Integrated Gradients)|                  | (Feature Eng & scaling)     |
  +-----------------------+                  +-----------------------------+
              |                                             |
              v                                             v
  +------------------------------------------------------------------------+
  |                BiLSTM Classifier Sepsis Prediction Model                |
  |                   Outputs Sepsis Risk (e.g. 84.3%)                     |
  +------------------------------------------------------------------------+
              |                                             |
              v                                             v
  +-----------------------+                  +-----------------------------+
  | Alert Engine          |                  | Clinician Dashboard         |
  | (Critical, Rising, PP)|                  | (index.html / app.js)       |
  +-----------------------+                  +-----------------------------+
```

### Key Components:
1. **FastAPI Ingestion Endpoint** (`/api/patient/update`): Receives real-time hourly telemetry data.
2. **SQLite Live Store**: Persists streaming observations to `sepsis_live.db`.
3. **Rolling Window Buffer**: Extracts up to 12-hour hourly observation sequences per patient.
4. **Online Preprocessor**: Generates 29 engineered clinical features (Shock Index, HR/MAP ratio, vital variability, lab sparsity timers, 1h lag trends) and applies standard scaling fit on the training cohort.
5. **BiLSTM Prediction Engine**: Evaluates 3D sequential tensors through a PyTorch Bidirectional LSTM to yield a continuous sepsis probability.
6. **Alert Engine**: Triggers clinical flags based on risk thresholds:
   * **Critical Sepsis Warning**: Risk $\ge 70\%$
   * **Acute Worsening**: Risk increase $\ge +20\%$ in 1 hour
   * **Persistent Danger Alert**: High risk ($\ge 80\%$) across consecutive hours
7. **Temporal Explainability (Captum IG)**: Computes hourly feature attributions for risk scores.
8. **Clinician Dashboard**: Real-time interactive UI with ward risk grids, live Chart.js trend graphs, and XAI breakdowns.

---

## 🏆 Model Leaderboard & Experimental Results

### Classical Machine Learning (Dataset B - 97 Engineered Features)
| Model | Test AUROC | Test AUPRC | Sensitivity (Recall) | Specificity | Training Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (Champion Baseline)** | **0.8381** | **0.1318** | **0.6495** | **0.8449** | 8.12s |
| LightGBM Classifier | 0.8315 | 0.1242 | 0.6310 | 0.8412 | 3.45s |
| Random Forest Classifier | 0.8190 | 0.1104 | 0.5840 | 0.8620 | 24.10s |
| Decision Tree Classifier | 0.7120 | 0.0540 | 0.4910 | 0.7830 | 1.80s |
| Logistic Regression | 0.7640 | 0.0820 | 0.5230 | 0.8110 | 0.95s |

### Deep Sequential Models (3D Sequence Tensors)
| Model Architecture | Test AUROC | Test AUPRC | Sensitivity (Recall) | Specificity | Parameters | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiLSTM Champion (w12_h0)** 👑 | **0.8435** | **0.1228** | **0.7848** | **0.7580** | **181,889** | **60.8 µs** |
| GRU Classifier | 0.8519 | 0.1431 | 0.7668 | 0.7831 | 136,193 | 48.2 µs |
| Transformer Encoder | 0.8504 | 0.1192 | 0.7259 | 0.8156 | 245,600 | 110.5 µs |
| Unidirectional LSTM | 0.8384 | 0.1125 | 0.7150 | 0.8014 | 91,400 | 42.1 µs |

> 🌟 **Key Clinical Advantage**: The BiLSTM champion achieved a **78.48% recall** (a **+13.53% increase** over XGBoost), ensuring early detection of critical septic patients with minimal inference delay.

### Early Warning Lead-Time Sensitivity (BiLSTM)
* **w12_h0 (0h Lead Time)**: AUROC = `0.8435`, Sensitivity = `78.48%`
* **w12_h3 (3h Lead Time)**: AUROC = `0.8238`, Sensitivity = `73.05%`
* **w12_h6 (6h Lead Time)**: AUROC = `0.8186`, Sensitivity = `74.94%`

---

## 🔍 Explainable AI (XAI) & Insights

* **Global Attributions (Tree SHAP)**: Identified ICU Length of Stay (`ICULOS`), Temperature (`Temp`), Lab ordering sparsity (`hours_since_last_Lactate`, `hours_since_last_pH`), and Shock Index as dominant predictors.
* **Local Attributions (Integrated Gradients)**: Confirmed that physiological changes in the final 2 hours of observation carry the highest positive risk weights.
* **Feature Concordance**: 37.93% Jaccard overlap between XGBoost SHAP and BiLSTM Integrated Gradients across top 20 predictors.

---

## 📂 Repository Structure

```text
SEPESDETECTOR/
├── dashboard/                   # Glassmorphic Clinician Workstation UI
│   ├── index.html               # Main layout (grid, chart containers, alerts)
│   ├── styles.css               # Styling design system and animations
│   └── app.js                   # Live API polling, Chart.js graphs, interactions
├── phase10_realtime/            # Real-time FastAPI Clinical Server
│   ├── server.py                # Server entry point & API routes
│   ├── api/patient_routes.py    # Patients API endpoints
│   ├── storage/database.py      # SQLite connection & schema management
│   ├── sequence/rolling_window.py # Trajectory buffer manager
│   ├── preprocessing/           # Real-time feature scaling & derivation
│   ├── alerts/alert_engine.py   # Clinical rules evaluator
│   └── ingestion/simulator.py   # Patient telemetry streaming simulator
├── deep_learning/               # PyTorch Sequential Modeling Framework
│   ├── models/                  # Architectures (BiLSTM, GRU, LSTM, Transformer)
│   ├── trainer.py               # Early stopping & batch training loops
│   ├── run_training.py          # Stage 7A model benchmark runner
│   └── run_final_training.py    # Stage 7B champion BiLSTM trainer
├── feature_engineering/         # Feature Augmentation Pipeline
├── preprocessing/               # Cleaning, Winsorization, & Imputation
├── eda/                         # Exploratory Cohort Analysis
├── explainability/              # SHAP & Integrated Gradients Engine
├── experiments/                 # Checkpoints, metrics graphs, prediction logs
├── reports/                     # Generated PDF/HTML clinical reports
├── INFO.TXT                     # Comprehensive project roadmap & logs
├── requirements.txt             # Project dependencies
└── README.md                    # [THIS FILE] Project documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/Vigneshkumar-S18/Sepsis_detection.git
cd Sepsis_detection
pip install -r requirements.txt
```

### 2. Launch Real-Time FastAPI Server
Start the clinical server (runs on `http://localhost:8000`):
```bash
python -m phase10_realtime.server
```

### 3. Run Patient Telemetry Stream Simulator
In a separate terminal, trigger patient data streaming:
```bash
python -m phase10_realtime.ingestion.simulator
```

### 4. Access Clinician Workstation
Open `dashboard/index.html` in any web browser or serve it locally:
```bash
python -m http.server 8080 --directory dashboard
```
Navigate to `http://localhost:8080` to view the live clinician workstation.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
