# PhysioNet ICU Telemetry Patient Stream Simulator
import os
import sys
import time
import pandas as pd
import requests

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger


def run_simulator(server_url="http://127.0.0.1:8000", interval_sec=2.0):
    logger.info("Initializing Patient Telemetry Stream Simulator...")
    
    # Locate raw data directories
    raw_dir_a = os.path.join(project_root, "datasets", "raw", "training_setA")
    raw_dir_b = os.path.join(project_root, "datasets", "raw", "training_setB")
    
    # Select a small group of patients to simulate
    # Patient IDs matching case studies to see real-time triggers:
    # TP: p000185, TN: p000003, FN: p000022
    sim_patient_ids = ["p000185", "p000003", "p000022"]
    
    patient_files = {}
    for p_id in sim_patient_ids:
        # Check in A
        path_a = os.path.join(raw_dir_a, f"{p_id}.psv")
        path_b = os.path.join(raw_dir_b, f"{p_id}.psv")
        if os.path.exists(path_a):
            patient_files[p_id] = path_a
        elif os.path.exists(path_b):
            patient_files[p_id] = path_b
            
    if not patient_files:
        logger.error("No case study patient files found in training raw directories.")
        return

    logger.info(f"Loaded {len(patient_files)} simulator patient tracks: {list(patient_files.keys())}")

    # Load patient stays dataframes
    patient_dfs = {}
    for p_id, filepath in patient_files.items():
        df = pd.read_csv(filepath, sep='|')
        # Fill missing key columns with nan for mapping
        patient_dfs[p_id] = df

    # Find the maximum stay length
    max_len = max(len(df) for df in patient_dfs.values())
    
    logger.info(f"Starting real-time simulation timeline over {max_len} observation hours...")
    
    for t in range(max_len):
        logger.info(f"\n--- Simulation Step Hour {t+1} ---")
        
        for p_id, df in patient_dfs.items():
            if t >= len(df):
                continue
                
            row = df.iloc[t]
            
            # Map columns to JSON structure
            payload = {
                "patient_id": p_id,
                "timestamp": f"2026-07-20T{10+t:02d}:15:00",
                "HR": float(row["HR"]) if not pd.isna(row["HR"]) else 80.0,
                "MAP": float(row["MAP"]) if not pd.isna(row["MAP"]) else 80.0,
                "Temp": float(row["Temp"]) if not pd.isna(row["Temp"]) else 37.0,
                "SpO2": float(row["O2Sat"]) if not pd.isna(row["O2Sat"]) else 98.0,
                "Resp": float(row["Resp"]) if not pd.isna(row["Resp"]) else 16.0,
                "SBP": float(row["SBP"]) if not pd.isna(row["SBP"]) else 120.0,
                "DBP": float(row["DBP"]) if not pd.isna(row["DBP"]) else 80.0,
                "Age": float(row["Age"]) if not pd.isna(row["Age"]) else 65.0,
                "Gender": int(row["Gender"]) if not pd.isna(row["Gender"]) else 1,
                "ICULOS": float(row["ICULOS"]) if not pd.isna(row["ICULOS"]) else float(t + 1)
            }
            
            # Post telemetry update
            update_url = f"{server_url}/api/patient/update"
            try:
                # 1. Post telemetry update
                res_up = requests.post(update_url, json=payload, timeout=2.0)
                if res_up.status_code == 200:
                    # 2. Trigger risk prediction assessment
                    pred_url = f"{server_url}/api/prediction/{p_id}"
                    res_pred = requests.get(pred_url, timeout=2.0)
                    pred_data = res_pred.json()
                    
                    logger.info(
                        f"  [{p_id}] Stay Hour: {t+1} | Risk: {pred_data['risk']:.1%} "
                        f"| Status: {pred_data['status']} | Latency: {pred_data['latency_ms']:.1f}ms"
                    )
                    if pred_data["alert_triggered"]:
                        logger.warning(f"  *** ALERT GENERATED for [{p_id}]! Priority: {pred_data['alert']['priority']}")
                else:
                    logger.error(f"  Failed update post for {p_id}: {res_up.text}")
            except Exception as e:
                logger.error(f"  Connection error posting for {p_id}: {e}")
                
        time.sleep(interval_sec)

    logger.info("Simulator execution completed.")


if __name__ == "__main__":
    # Wait for server to boot up before running
    time.sleep(3)
    run_simulator()
