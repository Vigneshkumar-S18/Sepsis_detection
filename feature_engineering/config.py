# Configuration parameters for clinical feature engineering

# 1. Key vitals to compute rolling statistics (mean, std/variability) over 6-hour windows
ROLLING_VITALS = ["HR", "Resp", "Temp", "MAP", "O2Sat"]
ROLLING_WINDOW_SIZE = 6

# 2. Key sparse laboratory parameters to compute "Hours Since Last Measurement" features for
SPARSITY_LABS = ["Lactate", "Creatinine", "WBC", "Platelets", "BUN", "pH", "HCO3"]

# 3. Key vitals to compute 1-hour temporal lag difference and slope features for
LAG_VITALS = ["HR", "Temp", "Resp", "O2Sat"]
LAG_INTERVAL = 1
