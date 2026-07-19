import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_class_distribution(df=None):
    """
    Computes Sepsis class imbalance at both hourly record and patient level,
    generates summary tables, and plots class distributions as figures.
    """
    paths = get_paths()
    palette = set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Computing Sepsis class distribution...")
    
    # 1. Hourly record level label distribution
    hourly_counts = df['SepsisLabel'].value_counts()
    total_hourly = len(df)
    sepsis_hourly = hourly_counts.get(1, 0)
    non_sepsis_hourly = hourly_counts.get(0, 0)
    sepsis_hourly_pct = round((sepsis_hourly / total_hourly) * 100, 2)
    non_sepsis_hourly_pct = round((non_sepsis_hourly / total_hourly) * 100, 2)
    
    # 2. Patient level label distribution (Ever Sepsis vs Never Sepsis)
    patient_labels = df.groupby('PatientID')['SepsisLabel'].max()
    total_patients = len(patient_labels)
    sepsis_patients = int(patient_labels.sum())
    non_sepsis_patients = int(total_patients - sepsis_patients)
    sepsis_patient_pct = round((sepsis_patients / total_patients) * 100, 2)
    non_sepsis_patient_pct = round((non_sepsis_patients / total_patients) * 100, 2)
    
    # 3. Create consolidated tables
    distribution_data = {
        "Level": ["Hourly Record", "Hourly Record", "Patient", "Patient"],
        "Class": ["Sepsis (1)", "Non-Sepsis (0)", "Sepsis (Ever)", "Non-Sepsis (Never)"],
        "Count": [sepsis_hourly, non_sepsis_hourly, sepsis_patients, non_sepsis_patients],
        "Percentage": [sepsis_hourly_pct, non_sepsis_hourly_pct, sepsis_patient_pct, non_sepsis_patient_pct]
    }
    
    dist_df = pd.DataFrame(distribution_data)
    tbl_path = os.path.join(paths["tables"], "class_table.csv")
    dist_df.to_csv(tbl_path, index=False)
    logger.info(f"Saved class distribution table to: {tbl_path}")
    
    # 4. Plots
    # Plot 1: Hourly Record Distribution (Bar Chart)
    plt.figure(figsize=(7, 5))
    ax1 = sns.barplot(
        x=["Non-Sepsis (0)", "Sepsis (1)"], 
        y=[non_sepsis_hourly, sepsis_hourly], 
        palette=[palette["accent"], palette["secondary"]],
        edgecolor="black",
        alpha=0.9
    )
    plt.title("Hourly Sepsis Label Distribution", pad=15)
    plt.ylabel("Record Count")
    
    # Add labels on top of the bars
    for p in ax1.patches:
        val = int(p.get_height())
        pct = round(val / total_hourly * 100, 2)
        ax1.annotate(f"{val:,}\n({pct}%)", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
    plt.ylim(0, max(non_sepsis_hourly, sepsis_hourly) * 1.15)
    plt.tight_layout()
    fig1_path = os.path.join(paths["figures"], "class_distribution.png")
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    logger.info(f"Saved hourly class distribution plot to: {fig1_path}")
    
    # Plot 2: Patient Level Distribution (Bar Chart)
    plt.figure(figsize=(7, 5))
    ax2 = sns.barplot(
        x=["Never Septic (0)", "Septic (1)"], 
        y=[non_sepsis_patients, sepsis_patients], 
        palette=[palette["primary"], palette["secondary"]],
        edgecolor="black",
        alpha=0.9
    )
    plt.title("Patient-Level Sepsis Distribution (Ever Septic)", pad=15)
    plt.ylabel("Patient Count")
    
    for p in ax2.patches:
        val = int(p.get_height())
        pct = round(val / total_patients * 100, 2)
        ax2.annotate(f"{val:,}\n({pct}%)", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)
    plt.ylim(0, max(non_sepsis_patients, sepsis_patients) * 1.15)
    plt.tight_layout()
    fig2_path = os.path.join(paths["figures"], "patient_distribution.png")
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    logger.info(f"Saved patient distribution plot to: {fig2_path}")
    
    return dist_df

if __name__ == "__main__":
    with Timer("Step 3 - Class Distribution"):
        run_class_distribution()
