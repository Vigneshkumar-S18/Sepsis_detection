import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_patient_analysis(df=None):
    """
    Computes patient-level stays, record counts, sepsis occurrence rates,
    saves summary statistics and plots the distribution of ICU stay lengths.
    """
    paths = get_paths()
    palette = set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Computing patient level characteristics...")
    
    # 1. Stay statistics per patient
    patient_grouped = df.groupby('PatientID')
    records_per_patient = patient_grouped.size()
    sepsis_per_patient = patient_grouped['SepsisLabel'].max()
    stay_per_patient = patient_grouped['ICULOS'].max()
    
    # 2. Compile metrics
    total_septic_patients = int(sepsis_per_patient.sum())
    total_non_septic_patients = int(len(sepsis_per_patient) - total_septic_patients)
    sepsis_patient_pct = round((total_septic_patients / len(sepsis_per_patient)) * 100, 2)
    
    patient_stats = {
        "metric": [
            "Total Patients",
            "Septic Patients Count",
            "Non-Septic Patients Count",
            "Septic Patients %",
            "Records per Patient (Mean)",
            "Records per Patient (Median)",
            "Records per Patient (Min)",
            "Records per Patient (Max)",
            "Stay Length (Mean Hours)",
            "Stay Length (Median Hours)",
            "Stay Length (Min Hours)",
            "Stay Length (Max Hours)"
        ],
        "value": [
            len(sepsis_per_patient),
            total_septic_patients,
            total_non_septic_patients,
            sepsis_patient_pct,
            round(records_per_patient.mean(), 2),
            round(records_per_patient.median(), 2),
            int(records_per_patient.min()),
            int(records_per_patient.max()),
            round(stay_per_patient.mean(), 2),
            round(stay_per_patient.median(), 2),
            int(stay_per_patient.min()),
            int(stay_per_patient.max())
        ]
    }
    
    patient_stats_df = pd.DataFrame(patient_stats)
    tbl_path = os.path.join(paths["tables"], "patient_statistics.csv")
    patient_stats_df.to_csv(tbl_path, index=False)
    logger.info(f"Saved patient statistics table to: {tbl_path}")
    
    # 3. Plot ICU stay histogram
    plt.figure(figsize=(9, 5))
    sns.histplot(stay_per_patient, bins=50, kde=True, color=palette["primary"], edgecolor="white", alpha=0.8)
    
    # Customize titles/labels
    plt.title("Distribution of ICU Stay Lengths (ICULOS)", pad=15)
    plt.xlabel("Stay Length (Hours)")
    plt.ylabel("Number of Patients")
    plt.xlim(0, stay_per_patient.max() + 5)
    plt.tight_layout()
    
    fig_path = os.path.join(paths["figures"], "patient_stay_histogram.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved patient stay histogram to: {fig_path}")
    
    return patient_stats_df

if __name__ == "__main__":
    with Timer("Step 2 - Patient Analysis"):
        run_patient_analysis()
