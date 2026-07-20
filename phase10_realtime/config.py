# Real-Time Inference Platform Configuration
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Database Path
DB_PATH = os.path.join(project_root, "phase10_realtime", "storage", "sepsis_live.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Scaler and Model paths
SCALER_PATH = os.path.join(project_root, "datasets", "processed", "scaler.pkl")
MODEL_PATH = os.path.join(project_root, "experiments", "checkpoints", "bilstm_w12_h0_final_best.pt")

# Output directory for latency logs
LOG_DIR = os.path.join(project_root, "phase10_realtime", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Performance threshold settings
CRITICAL_RISK_THRESHOLD = 0.70
INCREASE_RISK_THRESHOLD = 0.20
CONSECUTIVE_RISK_THRESHOLD = 0.80

# Suppression window for acknowledged alerts (in minutes)
ALERT_SUPPRESSION_MINUTES = 60
