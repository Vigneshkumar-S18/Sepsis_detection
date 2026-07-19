import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_outlier_analysis(df=None):
    """
    Identifies clinical/physiological impossibilities and statistical outliers
    using IQR, logs counts and percentages, and saves a boxplot figure.
    """
    paths = get_paths()
    palette = set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Detecting physiological and statistical outliers...")
    
    key_vitals = ["HR", "Temp", "Resp", "SBP", "MAP", "DBP"]
    
    # 1. Define physiological limits (standard clinical criteria)
    limits = {
        "HR": (20, 250),      # Heart Rate < 20 or > 250 bpm is physiologically impossible/extreme code blue
        "Temp": (25, 45),     # Temp < 25°C (extreme hypothermia) or > 45°C (extreme hyperpyrexia)
        "Resp": (3, 70),      # Respiration Rate < 3 or > 70 breaths per minute
        "SBP": (30, 280),     # Systolic BP < 30 or > 280 mmHg
        "MAP": (20, 200),     # Mean Arterial BP < 20 or > 200 mmHg
        "DBP": (10, 150)      # Diastolic BP < 10 or > 150 mmHg
    }
    
    outlier_list = []
    
    for vital in key_vitals:
        data = df[vital].dropna()
        if len(data) == 0:
            continue
            
        # Statistical limits (IQR)
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        stat_outliers_count = ((data < lower_bound) | (data > upper_bound)).sum()
        stat_outliers_pct = round((stat_outliers_count / len(data)) * 100, 2)
        
        # Clinical limits
        lower_limit, upper_limit = limits[vital]
        clin_outliers_count = ((data < lower_limit) | (data > upper_limit)).sum()
        clin_outliers_pct = round((clin_outliers_count / len(data)) * 100, 2)
        
        outlier_list.append({
            "Vital": vital,
            "Total_Non_Null": len(data),
            "IQR_Lower_Bound": round(lower_bound, 2),
            "IQR_Upper_Bound": round(upper_bound, 2),
            "IQR_Outlier_Count": int(stat_outliers_count),
            "IQR_Outlier_Pct": stat_outliers_pct,
            "Clinical_Range": f"{lower_limit} - {upper_limit}",
            "Clinical_Outlier_Count": int(clin_outliers_count),
            "Clinical_Outlier_Pct": clin_outliers_pct
        })
        
    outliers_summary_df = pd.DataFrame(outlier_list)
    tbl_path = os.path.join(paths["tables"], "outliers_summary.csv")
    outliers_summary_df.to_csv(tbl_path, index=False)
    logger.info(f"Saved outlier summary table to: {tbl_path}")
    
    # 2. Plot: Boxplots of key vitals
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    
    for index, vital in enumerate(key_vitals):
        sns.boxplot(
            y=df[vital].dropna(), 
            ax=axes[index], 
            color=palette["primary"] if index % 2 == 0 else palette["secondary"],
            width=0.4
        )
        axes[index].set_title(f"{vital} Distribution & Outliers")
        axes[index].set_ylabel("")
        
        # Highlight clinical limits
        lower_limit, upper_limit = limits[vital]
        axes[index].axhline(y=lower_limit, color="red", linestyle="--", alpha=0.5, label="Physiological Min")
        axes[index].axhline(y=upper_limit, color="red", linestyle="--", alpha=0.5, label="Physiological Max")
        
    plt.suptitle("Boxplots and Outlier Analysis for Key Vital Signs", y=0.98)
    plt.tight_layout()
    
    fig_path = os.path.join(paths["figures"], "outlier_boxplots.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved outlier boxplots figure to: {fig_path}")
    
    return outliers_summary_df

if __name__ == "__main__":
    with Timer("Step 6 - Outlier Detection"):
        run_outlier_analysis()
