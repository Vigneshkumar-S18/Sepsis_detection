import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer

def run_evaluation_suite():
    processed_dir = os.path.join(project_root, "datasets", "processed")
    reports_dir = os.path.join(project_root, "reports", "features")
    os.makedirs(reports_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, "train_features.parquet")
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    
    if not os.path.exists(train_path) or not os.path.exists(scaler_path):
        logger.error("train_features.parquet or scaler.pkl not found. Please run run_feature_engineering.py first.")
        sys.exit(1)
        
    # 1. Load scaled training features & scaler
    with Timer("Loading train features and scaler"):
        train_df = pd.read_parquet(train_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
    # 2. Reconstruct unscaled clinical values
    with Timer("Reconstructing unscaled clinical features for validation"):
        # Recreate list of scaled columns
        columns_to_scale = [
            col for col in train_df.columns
            if col not in ['PatientID', 'SepsisLabel', 'Gender', 'Unit1', 'Unit2', 'First_24h_Flag']
            and not col.endswith('_measured')
        ]
        
        train_df_unscaled = train_df.copy()
        train_df_unscaled[columns_to_scale] = scaler.inverse_transform(train_df[columns_to_scale])
        
    # 3. Step 1: Feature Validation Checks
    with Timer("Performing clinical range validation on engineered features"):
        validation_results = {}
        
        # Shock Index check (Normal 0.5 - 0.9)
        shock = train_df_unscaled['Shock_Index']
        anomalous_shock = ((shock < 0.3) | (shock > 2.0)).sum()
        validation_results['Shock_Index'] = {
            "mean": float(shock.mean()),
            "std": float(shock.std()),
            "min": float(shock.min()),
            "max": float(shock.max()),
            "anomalous_count": int(anomalous_shock),
            "anomalous_pct": float(anomalous_shock / len(shock) * 100)
        }
        
        # Pulse Pressure check (Normal 30 - 60 mmHg)
        pp = train_df_unscaled['Pulse_Pressure']
        anomalous_pp = ((pp < 15) | (pp > 120)).sum()
        validation_results['Pulse_Pressure'] = {
            "mean": float(pp.mean()),
            "std": float(pp.std()),
            "min": float(pp.min()),
            "max": float(pp.max()),
            "anomalous_count": int(anomalous_pp),
            "anomalous_pct": float(anomalous_pp / len(pp) * 100)
        }
        
        # MAP Deviation check (Normal < 15 mmHg)
        map_dev = train_df_unscaled['MAP_deviation'].abs()
        anomalous_map = (map_dev > 30.0).sum()
        validation_results['MAP_deviation'] = {
            "mean_abs": float(map_dev.mean()),
            "std_abs": float(map_dev.std()),
            "max_abs": float(map_dev.max()),
            "anomalous_count": int(anomalous_map),
            "anomalous_pct": float(anomalous_map / len(map_dev) * 100)
        }
        
        # Respiratory stress ratio check (Normal 0.1 - 0.3)
        resp_ratio = train_df_unscaled['Resp_O2Sat_Ratio']
        anomalous_resp = ((resp_ratio < 0.05) | (resp_ratio > 0.8)).sum()
        validation_results['Resp_O2Sat_Ratio'] = {
            "mean": float(resp_ratio.mean()),
            "std": float(resp_ratio.std()),
            "min": float(resp_ratio.min()),
            "max": float(resp_ratio.max()),
            "anomalous_count": int(anomalous_resp),
            "anomalous_pct": float(anomalous_resp / len(resp_ratio) * 100)
        }
        
        val_json_path = os.path.join(reports_dir, "feature_validation.json")
        with open(val_json_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Saved feature validation report to: {val_json_path}")
        
    # 4. Step 2: Correlation Analysis (finding duplicate/redundant features)
    with Timer("Analyzing feature correlations to detect redundancies"):
        # Select newly engineered features
        base_columns = [
            'PatientID', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
            'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
            'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
            'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
            'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
            'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'SepsisLabel'
        ]
        engineered_features = [col for col in train_df.columns if col not in base_columns and not col.endswith('_measured')]
        
        # Compute correlation matrix of engineered features
        corr_matrix = train_df[engineered_features].corr().abs()
        
        # Extract pairs with correlation > 0.90
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        redundant_pairs = []
        for col in upper_tri.columns:
            high_corr_features = upper_tri.index[upper_tri[col] > 0.90].tolist()
            for index_feat in high_corr_features:
                val = upper_tri.loc[index_feat, col]
                redundant_pairs.append({
                    "Feature_A": col,
                    "Feature_B": index_feat,
                    "Correlation": float(val)
                })
                
        redundant_df = pd.DataFrame(redundant_pairs)
        corr_path = os.path.join(reports_dir, "duplicate_correlations.csv")
        redundant_df.to_csv(corr_path, index=False)
        logger.info(f"Saved highly correlated feature pairs to: {corr_path}")
        
    # 5. Step 3: Feature Importance Baseline
    with Timer("Training baseline Random Forest classifier for feature importance"):
        # Subsample 5% of training data to keep it fast
        sample_df = train_df.sample(frac=0.05, random_state=42)
        X = sample_df.drop(columns=['PatientID', 'SepsisLabel'])
        y = sample_df['SepsisLabel']
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": rf.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        
        importance_path = os.path.join(reports_dir, "baseline_importance.csv")
        importances.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importances to: {importance_path}")
        
        # Print top 15 features
        logger.info("Top 15 features by baseline Random Forest importance:")
        for idx, row in importances.head(15).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']*100:.2f}%")

if __name__ == "__main__":
    run_evaluation_suite()
