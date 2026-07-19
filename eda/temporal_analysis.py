import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_temporal_analysis(df=None):
    """
    Selects patients who developed Sepsis, aligns them by the hour of onset,
    computes the physiological trajectories of key vitals during the 12 hours
    leading up to diagnosis, and saves the trend graphs.
    """
    paths = get_paths()
    palette = set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Performing temporal analysis on Sepsis onset trajectories...")
    
    # 1. Find the hour of first SepsisLabel == 1 for each patient who developed Sepsis
    septic_patients_labels = df.groupby('PatientID')['SepsisLabel'].max()
    septic_patient_ids = septic_patients_labels[septic_patients_labels == 1].index
    
    if len(septic_patient_ids) == 0:
        logger.warning("No septic patients found in the dataset. Skipping temporal trends.")
        return None
        
    septic_df = df[df['PatientID'].isin(septic_patient_ids)].copy()
    
    # Find onset hour (the minimum ICULOS where SepsisLabel is 1)
    septic_rows = septic_df[septic_df['SepsisLabel'] == 1]
    onset_hours = septic_rows.groupby('PatientID')['ICULOS'].min().reset_index()
    onset_hours.columns = ['PatientID', 'Onset_Hour']
    
    # Merge onset hour back
    septic_df = pd.merge(septic_df, onset_hours, on='PatientID')
    
    # Calculate Hour Relative to Onset (t = current_hour - onset_hour)
    septic_df['Hours_To_Onset'] = septic_df['ICULOS'] - septic_df['Onset_Hour']
    
    # Filter 12 hours before onset (Hours_To_Onset is from -12 to 0)
    pre_sepsis_df = septic_df[(septic_df['Hours_To_Onset'] >= -12) & (septic_df['Hours_To_Onset'] <= 0)]
    
    # 2. Get baseline of Non-Septic patients for comparison
    # Since non-septic patients do not have an onset time, we compute their overall average values
    non_septic_patient_ids = septic_patients_labels[septic_patients_labels == 0].index
    non_septic_df = df[df['PatientID'].isin(non_septic_patient_ids)]
    
    baselines = {
        "HR": non_septic_df['HR'].mean(),
        "Temp": non_septic_df['Temp'].mean(),
        "Resp": non_septic_df['Resp'].mean(),
        "O2Sat": non_septic_df['O2Sat'].mean()
    }
    
    # 3. Compute mean trajectory per hour relative to onset
    trajectory = pre_sepsis_df.groupby('Hours_To_Onset')[["HR", "Temp", "Resp", "O2Sat"]].mean().reset_index()
    
    tbl_path = os.path.join(paths["tables"], "temporal_trajectories.csv")
    trajectory.to_csv(tbl_path, index=False)
    logger.info(f"Saved temporal trajectories table to: {tbl_path}")
    
    # 4. Plot 2x2 grid of trajectories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    vitals = ["HR", "Temp", "Resp", "O2Sat"]
    titles = [
        "Heart Rate Trajectory (HR)", 
        "Temperature Trajectory (Temp)", 
        "Respiration Rate Trajectory (Resp)", 
        "Oxygen Saturation Trajectory (O2Sat)"
    ]
    ylabels = ["HR (bpm)", "Temp (°C)", "Resp (breaths/min)", "O2Sat (%)"]
    colors = [palette["secondary"], palette["primary"], palette["accent"], palette["neutral_dark"]]
    
    for i, vital in enumerate(vitals):
        ax = axes[i]
        # Plot septic patient trajectory
        sns.lineplot(
            x="Hours_To_Onset", 
            y=vital, 
            data=pre_sepsis_df, 
            ax=ax, 
            color=colors[i], 
            linewidth=2.5,
            errorbar=('ci', 95),
            label="Septic Patients (95% CI)"
        )
        
        # Plot non-septic patient baseline as a horizontal line
        baseline_val = baselines[vital]
        ax.axhline(
            y=baseline_val, 
            color="red", 
            linestyle="--", 
            alpha=0.7, 
            label=f"Non-Septic Baseline ({baseline_val:.2f})"
        )
        
        ax.set_title(titles[i])
        ax.set_xlabel("Hours Relative to Sepsis Onset")
        ax.set_ylabel(ylabels[i])
        ax.set_xlim(-12, 0)
        ax.legend(loc="upper left")
        
    plt.suptitle("Clinical Trajectories of Vital Signs 12 Hours Prior to Sepsis Onset", y=0.98, fontsize=16)
    plt.tight_layout()
    
    fig_path = os.path.join(paths["figures"], "temporal_trends.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved temporal trends plot to: {fig_path}")
    
    return trajectory

if __name__ == "__main__":
    with Timer("Step 8 - Temporal Analysis"):
        run_temporal_analysis()
