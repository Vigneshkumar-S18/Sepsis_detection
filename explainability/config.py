# Configuration for Phase 8 Explainable AI (XAI) Framework
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Output directories
VISUALIZATIONS_DIR = os.path.join(project_root, "explainability", "visualizations")
REPORTS_DIR = os.path.join(project_root, "explainability", "reports")
OUTPUTS_DIR = os.path.join(project_root, "explainability", "outputs")

os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Device configuration (default CPU)
DEVICE = "cpu"
SEED = 42

# Label and Exclusions
LABEL_COLUMN = "SepsisLabel"
PATIENT_ID_COLUMN = "PatientID"
EXCLUDE_COLUMNS = [PATIENT_ID_COLUMN, LABEL_COLUMN]
