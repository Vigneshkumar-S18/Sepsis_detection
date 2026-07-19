# Preprocessing configuration parameters for THAARU Sepsis AI

# 1. Acceptable clinical ranges for vital signs and physiological metrics.
# Values outside these ranges represent impossible outliers (e.g. sensor errors)
# and should be set to NaN before imputation.
CLINICAL_RANGES = {
    "HR": (20, 250),      # Heart Rate (bpm)
    "Temp": (25, 45),     # Body Temperature (°C)
    "Resp": (4, 80),      # Respiration Rate (breaths/min)
    "O2Sat": (0, 100),    # Oxygen Saturation (%)
    "SBP": (40, 300),     # Systolic Blood Pressure (mmHg)
    "MAP": (20, 200),     # Mean Arterial Pressure (mmHg)
    "DBP": (20, 200),     # Diastolic Blood Pressure (mmHg)
    "pH": (6.8, 8.0)      # Blood acidity level
}

# 2. Variable Groupings
VITALS_COLUMNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]

LAB_COLUMNS = [
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
    'Platelets'
]

DEMOGRAPHIC_COLUMNS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

# Columns that should NOT be scaled during preprocessing (identifiers, labels, and indicators)
# Any column ending with "_measured" will also be dynamically skipped during scaling.
EXCLUDE_SCALING_COLUMNS = ['PatientID', 'SepsisLabel']
