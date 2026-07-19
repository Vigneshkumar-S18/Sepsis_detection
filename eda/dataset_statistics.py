import os
import sys
import json
import pandas as pd
from eda.utils import get_paths, load_dataset, logger, Timer

def run_dataset_statistics(df=None):
    """
    Computes overall dataset shapes, types, memory usage, duplicate rows,
    and aggregate statistics for patient ICU stay lengths.
    """
    paths = get_paths()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Computing general dataset statistics...")
    
    # 1. Row, patient, feature counts
    num_rows = len(df)
    unique_patients = df['PatientID'].nunique()
    
    # Features (excluding PatientID and SepsisLabel)
    clinical_features = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel']]
    num_features = len(clinical_features)
    
    # 2. Duplicate rows
    num_duplicates = df.duplicated().sum()
    
    # 3. Memory usage
    mem_bytes = df.memory_usage(deep=True).sum()
    mem_mb = round(mem_bytes / (1024 * 1024), 2)
    
    # 4. ICU Stay Lengths (from ICULOS per patient)
    # The length of stay is the max ICULOS reached by each patient
    stay_lengths = df.groupby('PatientID')['ICULOS'].max()
    avg_stay = round(stay_lengths.mean(), 2)
    med_stay = round(stay_lengths.median(), 2)
    max_stay = int(stay_lengths.max())
    min_stay = int(stay_lengths.min())
    
    # 5. Get file size of CSV
    csv_file = os.path.join(paths["interim"], "merged_dataset.csv")
    csv_size_mb = 0.0
    if os.path.exists(csv_file):
        csv_size_mb = round(os.path.getsize(csv_file) / (1024 * 1024), 1)
        
    # 6. Compile structures
    stats_dict = {
        "unique_patients": int(unique_patients),
        "total_records": int(num_rows),
        "num_features": int(num_features),
        "duplicate_rows": int(num_duplicates),
        "memory_usage_mb": float(mem_mb),
        "stay_length_avg_hours": float(avg_stay),
        "stay_length_median_hours": float(med_stay),
        "stay_length_max_hours": int(max_stay),
        "stay_length_min_hours": int(min_stay),
        "size_mb": float(csv_size_mb)
    }
    
    # Save JSON report
    json_path = os.path.join(paths["statistics"], "dataset_statistics.json")
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    logger.info(f"Saved JSON statistics to: {json_path}")
    
    # Save CSV report
    csv_path = os.path.join(paths["statistics"], "dataset_statistics.csv")
    stats_df = pd.DataFrame(list(stats_dict.items()), columns=["Metric", "Value"])
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV statistics to: {csv_path}")
    
    # Save Text summary
    txt_path = os.path.join(paths["statistics"], "dataset_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("      PHYSIONET CHALLENGE 2019 STATISTICS\n")
        f.write("="*40 + "\n")
        for k, v in stats_dict.items():
            name = k.replace("_", " ").title()
            f.write(f"{name:<30}: {v}\n")
        f.write("="*40 + "\n")
    logger.info(f"Saved text summary to: {txt_path}")
    
    return stats_dict

if __name__ == "__main__":
    with Timer("Step 1 - Dataset Statistics"):
        run_dataset_statistics()
