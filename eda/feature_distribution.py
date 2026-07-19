import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_feature_distribution(df=None):
    """
    Computes summary statistics for all clinical features, saves them as a CSV,
    and saves distribution figures for key clinical vitals.
    """
    paths = get_paths()
    palette = set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Computing feature distributions and statistics...")
    
    # 1. Feature statistics computation
    clinical_features = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2', 'Gender']]
    
    feature_stats_list = []
    for col in clinical_features:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
            
        mean_val = col_data.mean()
        median_val = col_data.median()
        std_val = col_data.std()
        var_val = col_data.var()
        min_val = col_data.min()
        max_val = col_data.max()
        q1 = np.percentile(col_data, 25)
        q3 = np.percentile(col_data, 75)
        iqr_val = q3 - q1
        
        feature_stats_list.append({
            "Feature": col,
            "Mean": round(mean_val, 2),
            "Median": round(median_val, 2),
            "Std": round(std_val, 2),
            "Variance": round(var_val, 2),
            "Min": round(min_val, 2),
            "Max": round(max_val, 2),
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr_val, 2)
        })
        
    stats_df = pd.DataFrame(feature_stats_list)
    tbl_path = os.path.join(paths["tables"], "feature_statistics.csv")
    stats_df.to_csv(tbl_path, index=False)
    logger.info(f"Saved feature statistics table to: {tbl_path}")
    
    # 2. Plot: HR distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df['HR'].dropna(), kde=True, color=palette["primary"], stat="density", edgecolor="white", alpha=0.8)
    plt.title("Distribution of Patient Heart Rate (HR)", pad=15)
    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("Density")
    plt.tight_layout()
    hr_fig_path = os.path.join(paths["figures"], "hr_histogram.png")
    plt.savefig(hr_fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved HR histogram plot to: {hr_fig_path}")
    
    # 3. Plot: Temp distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Temp'].dropna(), kde=True, color=palette["secondary"], stat="density", edgecolor="white", alpha=0.8)
    plt.title("Distribution of Patient Body Temperature (Temp)", pad=15)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Density")
    plt.tight_layout()
    temp_fig_path = os.path.join(paths["figures"], "temp_histogram.png")
    plt.savefig(temp_fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved Temp histogram plot to: {temp_fig_path}")
    
    return stats_df

if __name__ == "__main__":
    with Timer("Step 5 - Feature Distribution"):
        run_feature_distribution()
