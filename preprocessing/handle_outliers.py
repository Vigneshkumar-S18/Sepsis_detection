import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer, get_paths
from preprocessing.config import VITALS_COLUMNS

def run_outlier_handling(train_df, val_df, test_df):
    """
    Analyzes statistical outliers using Z-score and IQR methods.
    To prevent data leakage, outlier clipping thresholds (0.1% and 99.9%)
    are fit exclusively on the Training set and applied across all splits.
    Saves outlier counts and boxplot figures.
    """
    logger.info("Detecting and handling clinical outliers...")
    reports_dir = os.path.join(project_root, "reports", "preprocessing")
    os.makedirs(reports_dir, exist_ok=True)
    
    outlier_summary = []
    clipping_thresholds = {}
    
    # 1. Fit clipping thresholds on Train set & apply to all splits
    for col in VITALS_COLUMNS:
        train_data = train_df[col].dropna()
        if len(train_data) == 0:
            continue
            
        # Statistical outlier counts using standard IQR
        q1 = train_data.quantile(0.25)
        q3 = train_data.quantile(0.75)
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr
        
        # Clinically conservative clipping limits (0.1% and 99.9%)
        # This removes extreme artifact noise while preserving critical clinical shock/states
        lower_clip = train_data.quantile(0.001)
        upper_clip = train_data.quantile(0.999)
        clipping_thresholds[col] = (lower_clip, upper_clip)
        
        # Track statistics before clipping
        for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
                
            iqr_outliers = ((col_data < lower_iqr) | (col_data > upper_iqr)).sum()
            clip_outliers = ((col_data < lower_clip) | (col_data > upper_clip)).sum()
            
            outlier_summary.append({
                "Feature": col,
                "Split": split_name,
                "Lower_IQR": round(lower_iqr, 2),
                "Upper_IQR": round(upper_iqr, 2),
                "IQR_Outliers_Count": int(iqr_outliers),
                "IQR_Outliers_Pct": round((iqr_outliers / len(col_data)) * 100, 2),
                "Lower_Clip_Limit": round(lower_clip, 2),
                "Upper_Clip_Limit": round(upper_clip, 2),
                "Clipped_Outliers_Count": int(clip_outliers),
                "Clipped_Outliers_Pct": round((clip_outliers / len(col_data)) * 100, 2)
            })
            
            # Apply clipping to data
            df[col] = df[col].clip(lower_clip, upper_clip)
            
    # Save CSV summary
    summary_df = pd.DataFrame(outlier_summary)
    summary_path = os.path.join(reports_dir, "outlier_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved outlier summary to: {summary_path}")
    
    # 2. Plot Boxplots of key vitals after clipping
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, col in enumerate(VITALS_COLUMNS):
        if col in train_df.columns:
            sns.boxplot(y=train_df[col], ax=axes[idx], color="#0f766e")
            axes[idx].set_title(f"{col} Post-Clipping")
            axes[idx].set_ylabel("")
            
    plt.suptitle("Vital Signs Outlier Distributions (Post-Clipping)", y=0.98, fontsize=14)
    plt.tight_layout()
    
    fig_path = os.path.join(reports_dir, "boxplots.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved post-clipping boxplots to: {fig_path}")
    
    return train_df, val_df, test_df, summary_df

if __name__ == "__main__":
    with Timer("Step 3.5 - Outlier Handling"):
        processed_dir = os.path.join(project_root, "datasets", "processed")
        train_path = os.path.join(processed_dir, "train_split.parquet")
        val_path = os.path.join(processed_dir, "val_split.parquet")
        test_path = os.path.join(processed_dir, "test_split.parquet")
        
        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            logger.error("Split parquets not found. Please run split_data.py first.")
            sys.exit(1)
            
        train = pd.read_parquet(train_path)
        val = pd.read_parquet(val_path)
        test = pd.read_parquet(test_path)
        
        train, val, test, _ = run_outlier_handling(train, val, test)
        
        train.to_parquet(train_path, index=False)
        val.to_parquet(val_path, index=False)
        test.to_parquet(test_path, index=False)
        logger.info("Saved outlier-handled datasets back to processed split files.")
