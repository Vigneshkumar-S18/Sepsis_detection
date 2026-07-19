import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_missing_analysis(df=None):
    """
    Analyzes missing data per feature, ranks features by missingness,
    generates summary tables, and plots heatmaps and bar charts.
    """
    paths = get_paths()
    palette = set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Analyzing missing values...")
    
    # 1. Missing stats per feature (excluding PatientID and SepsisLabel)
    cols_to_analyze = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel']]
    total_records = len(df)
    
    missing_counts = df[cols_to_analyze].isnull().sum()
    missing_pct = (missing_counts / total_records) * 100
    
    missing_df = pd.DataFrame({
        "Feature": missing_counts.index,
        "MissingCount": missing_counts.values,
        "MissingPercentage": missing_pct.round(2).values
    })
    
    # Sort descending
    missing_df = missing_df.sort_values(by="MissingPercentage", ascending=False)
    
    tbl_path = os.path.join(paths["tables"], "missing_table.csv")
    missing_df.to_csv(tbl_path, index=False)
    logger.info(f"Saved missing statistics table to: {tbl_path}")
    
    # 2. Plot 1: Missing values heatmap
    # We sample 1000 rows to make it fast to render and highly legible.
    # We sort by PatientID to show temporal continuity of missingness.
    sample_df = df[cols_to_analyze].sample(n=min(2000, len(df)), random_state=42)
    sample_df = sample_df.reindex(sample_df.isnull().sum(axis=1).sort_values().index)
    
    plt.figure(figsize=(12, 6))
    # We use a clean teal/coral mask for missing values (white = present, dark teal = missing)
    sns.heatmap(sample_df.isnull(), cbar=False, cmap="GnBu", yticklabels=False)
    plt.title("Missing Value Matrix Heatmap (Sample of 2000 Records)", pad=15)
    plt.xlabel("Features")
    plt.ylabel("Records (Sorted by Missingness)")
    plt.tight_layout()
    
    heatmap_path = os.path.join(paths["figures"], "missing_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    logger.info(f"Saved missing values heatmap to: {heatmap_path}")
    
    # 3. Plot 2: Missing percentage per feature (Horizontal Bar Chart)
    # Filter features to only show those with missingness > 0% to keep it clean,
    # or just show the top 20 missing features.
    top_missing = missing_df.head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="MissingPercentage", 
        y="Feature", 
        data=top_missing, 
        color=palette["primary"], 
        edgecolor="black",
        alpha=0.9
    )
    plt.title("Top 20 Missing Features by Percentage", pad=15)
    plt.xlabel("Missing Percentage (%)")
    plt.ylabel("Feature Name")
    plt.xlim(0, 105)
    
    # Add values next to the bars
    for index, value in enumerate(top_missing['MissingPercentage']):
        plt.text(value + 1, index, f"{value}%", va='center', fontsize=9)
        
    plt.tight_layout()
    bar_path = os.path.join(paths["figures"], "missing_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    logger.info(f"Saved missing values bar chart to: {bar_path}")
    
    return missing_df

if __name__ == "__main__":
    with Timer("Step 4 - Missing Analysis"):
        run_missing_analysis()
