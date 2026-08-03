# Real-Time Telemetry Preprocessing Engine
import os
import sys
import pickle
import numpy as np
import pandas as pd

# Fetch project root from config
from phase10_realtime.config import project_root, SCALER_PATH


class OnlinePreprocessor:
    """
    Applies real-time feature engineering and standardization matching offline training.
    """
    def __init__(self):
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Standard Scaler not found at: {SCALER_PATH}")
            
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load the feature names in the exact order expected by the deep learning model (95 columns)
        test_feat_path = os.path.join(project_root, "datasets", "processed", "test_features.parquet")
        if not os.path.exists(test_feat_path):
            raise FileNotFoundError(f"Processed test features parquet not found: {test_feat_path}")
            
        test_df = pd.read_parquet(test_feat_path)
        # SepsisLabel and PatientID are excluded from model inputs
        self.expected_features = [col for col in test_df.columns if col not in ["PatientID", "SepsisLabel"]]
        
        # Raw 40 features
        self.raw_cols = [
            "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
            "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
            "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
            "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
            "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
            "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2",
            "HospAdmTime", "ICULOS"
        ]

    def normalize_column_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes common clinical feature column names (case-insensitive).
        """
        col_map = {}
        for col in df.columns:
            c_upper = str(col).strip().upper().replace("_", "").replace(" ", "").replace("-", "")
            if c_upper in ["HR", "HEARTRATE", "PULSE", "PULSERATE", "BPM"]:
                col_map[col] = "HR"
            elif c_upper in ["O2SAT", "SPO2", "OXYGEN", "OXYGENSATURATION", "SAT"]:
                col_map[col] = "O2Sat"
            elif c_upper in ["TEMP", "TEMPERATURE", "BODYTEMP", "FEVER"]:
                col_map[col] = "Temp"
            elif c_upper in ["MAP", "MEANBP", "MEANARTERIALBP", "MEANARTERIALPRESSURE"]:
                col_map[col] = "MAP"
            elif c_upper in ["SBP", "SYSBP", "SYSTOLIC", "SYSTOLICBP"]:
                col_map[col] = "SBP"
            elif c_upper in ["DBP", "DIABP", "DIASTOLIC", "DIASTOLICBP"]:
                col_map[col] = "DBP"
            elif c_upper in ["RESP", "RR", "RESPIRATORYRATE", "RESPRATE", "BREATHS"]:
                col_map[col] = "Resp"
            elif c_upper in ["WBC", "WHITEBLOODCELL", "LEUKOCYTES", "LEUCOCYTES"]:
                col_map[col] = "WBC"
            elif c_upper in ["PLATELETS", "PLT", "PLATELETCOUNT"]:
                col_map[col] = "Platelets"
            elif c_upper in ["LACTATE", "LACTICACID", "LAC"]:
                col_map[col] = "Lactate"
            elif c_upper in ["CREATININE", "CREAT", "CR"]:
                col_map[col] = "Creatinine"
            elif c_upper in ["GLUCOSE", "GLU", "SUGAR", "BSL"]:
                col_map[col] = "Glucose"
            elif c_upper in ["BUN", "UREA"]:
                col_map[col] = "BUN"
            elif c_upper in ["PH"]:
                col_map[col] = "pH"
            elif c_upper in ["HCO3", "BICARBONATE"]:
                col_map[col] = "HCO3"
            elif c_upper in ["AGE"]:
                col_map[col] = "Age"
            elif c_upper in ["GENDER", "SEX"]:
                col_map[col] = "Gender"
            elif c_upper in ["ICULOS", "LOS", "HOUR", "HOURS", "TIME", "TIMESTEP"]:
                col_map[col] = "ICULOS"
            elif c_upper in ["PATIENTID", "PATIENT", "PID", "SUBJECTID"]:
                col_map[col] = "PatientID"
        return df.rename(columns=col_map)

    def validate_clinical_schema(self, df: pd.DataFrame):
        """
        Validates that the input dataset contains legitimate clinical vital signs/lab features.
        Raises ValueError if the dataset is unrelated or invalid.
        """
        df_norm = self.normalize_column_aliases(df)
        
        # Check matched clinical raw features
        matched_all_raw = [c for c in self.raw_cols if c in df_norm.columns]
        
        # Check matched 95 model feature columns
        matched_expected = [c for c in self.expected_features if c in df_norm.columns]

        if not matched_all_raw and not matched_expected:
            found_cols = list(df.columns[:10])
            raise ValueError(
                f"Invalid dataset schema. The uploaded file contains columns {found_cols}, "
                f"which do not match any recognized clinical physiological parameters (such as Temp, SpO2/O2Sat, Heart Rate/HR, MAP, Resp, SBP, DBP, Glucose, Lactate, WBC, etc.). "
                f"Please upload a valid ICU patient telemetry dataset."
            )

    def preprocess_sequence(self, raw_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocesses a single patient's stay trajectory (up to 12 rows)
        and returns a scaled 2D array of shape (k, 95).
        """
        self.validate_clinical_schema(raw_df)
        df_unscaled = self.normalize_column_aliases(raw_df.copy())
        
        # 1. Fill missing columns with NaN to ensure complete schema
        for col in self.raw_cols:
            if col not in df_unscaled.columns:
                df_unscaled[col] = np.nan

        # 2. Add measured flags for sparse labs (1 if not null, 0 otherwise)
        sparsity_labs = ["Lactate", "Creatinine", "WBC", "Platelets", "BUN", "pH", "HCO3"]
        for col in sparsity_labs:
            df_unscaled[f"{col}_measured"] = (~df_unscaled[col].isna()).astype(int)

        # 3. Apply forward-fill and back-fill
        df_unscaled = df_unscaled.ffill().bfill()
        
        # Apply clinical default values for any remaining NaN cells
        clinical_defaults = {
            "HR": 80.0, "O2Sat": 97.0, "Temp": 37.0, "MAP": 80.0, "SBP": 120.0, "DBP": 80.0,
            "Resp": 16.0, "WBC": 7.5, "Platelets": 250.0, "Glucose": 100.0, "Lactate": 1.0,
            "Creatinine": 1.0, "BUN": 15.0, "pH": 7.4, "HCO3": 24.0, "Unit1": 0.0, "Unit2": 0.0,
            "HospAdmTime": -1.0
        }
        for col, val in clinical_defaults.items():
            if col in df_unscaled.columns:
                df_unscaled[col] = df_unscaled[col].fillna(val)
        df_unscaled = df_unscaled.fillna(0.0)  # Catch-all

        # 4. Initialize output DataFrame
        df_out = df_unscaled.copy()

        # 5. Cardiovascular Indices
        df_out['Shock_Index'] = df_unscaled['HR'] / df_unscaled['SBP'].clip(lower=40)
        df_out['Pulse_Pressure'] = df_unscaled['SBP'] - df_unscaled['DBP']
        est_map = (df_unscaled['DBP'] * 2 + df_unscaled['SBP']) / 3
        df_out['MAP_deviation'] = df_unscaled['MAP'] - est_map

        # 6. Clinical Ratios
        df_out['HR_MAP_Ratio'] = df_unscaled['HR'] / df_unscaled['MAP'].clip(lower=20)
        df_out['Resp_O2Sat_Ratio'] = df_unscaled['Resp'] / df_unscaled['O2Sat'].clip(lower=10)
        df_out['HR_Temp_Ratio'] = df_unscaled['HR'] / df_unscaled['Temp'].clip(lower=25)

        # 7. Rolling Statistics (6h window)
        rolling_vitals = ["HR", "Resp", "Temp", "MAP", "O2Sat"]
        for col in rolling_vitals:
            df_out[f"{col}_roll_mean_6h"] = df_unscaled[col].rolling(6, min_periods=1).mean()
            df_out[f"{col}_roll_std_6h"] = df_unscaled[col].rolling(6, min_periods=1).std().fillna(0.0)

        # 8. Trends and Lags (1h diff)
        lag_vitals = ["HR", "Temp", "Resp", "O2Sat"]
        for col in lag_vitals:
            df_out[f"{col}_diff_1h"] = df_unscaled[col].diff(periods=1).fillna(0.0)

        # 9. Sparsity hours since last measured
        for col in sparsity_labs:
            measured_col = f"{col}_measured"
            temp_col = f"temp_{col}_last"
            df_unscaled[temp_col] = df_unscaled['ICULOS'].where(df_unscaled[measured_col] == 1)
            last_measured_ffilled = df_unscaled[temp_col].ffill()
            df_unscaled.drop(columns=[temp_col], inplace=True)
            hours_since = df_unscaled['ICULOS'] - last_measured_ffilled
            df_out[f"hours_since_last_{col}"] = hours_since.fillna(999.0)

        # 10. ICU timeline flags
        df_out['First_24h_Flag'] = (df_unscaled['ICULOS'] <= 24).astype(int)
        df_out['Diurnal_Proxy_Hour'] = (df_unscaled['ICULOS'] % 24).astype(int)

        # 11. Reindex to align with scaler feature names
        df_out = df_out.reindex(columns=self.scaler.feature_names_in_, fill_value=0.0)

        # 12. Apply standardized scaling
        cols_to_scale = list(self.scaler.feature_names_in_)
        df_out[cols_to_scale] = self.scaler.transform(df_out[cols_to_scale])

        # 13. Reorder columns to match expected feature ordering
        df_out = df_out.reindex(columns=self.expected_features, fill_value=0.0)
                
        return df_out[self.expected_features].values
