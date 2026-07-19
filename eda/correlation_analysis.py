import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from eda.utils import get_paths, load_dataset, set_plot_style, logger, Timer

def run_correlation_analysis(df=None):
    """
    Calculates the Pearson correlation matrix for clinical features,
    identifies top positive/negative correlations, saves the correlation matrix,
    and plots the correlation heatmap.
    """
    paths = get_paths()
    set_plot_style()
    
    if df is None:
        df = load_dataset()
        
    logger.info("Computing correlations between clinical features...")
    
    # 1. Select numeric features (exclude ID, units and label)
    cols_to_correlate = [c for c in df.columns if c not in ['PatientID', 'SepsisLabel', 'Unit1', 'Unit2']]
    
    # Drop columns that are completely null to avoid NaN rows in correlation
    valid_cols = [c for c in cols_to_correlate if df[c].dropna().size > 100]
    
    corr_matrix = df[valid_cols].corr()
    
    tbl_path = os.path.join(paths["tables"], "correlation_table.csv")
    corr_matrix.to_csv(tbl_path)
    logger.info(f"Saved correlation table to: {tbl_path}")
    
    # Extract top positive/negative correlations (excluding self-correlations of 1.0)
    corr_unstacked = corr_matrix.unstack()
    corr_sorted = corr_unstacked.sort_values(ascending=False)
    
    # Filter out self-correlations and duplicates (since corr(A, B) = corr(B, A))
    pairs_done = set()
    top_corr_pairs = []
    
    for (feat_a, feat_b), val in corr_sorted.items():
        if feat_a != feat_b and (feat_b, feat_a) not in pairs_done:
            pairs_done.add((feat_a, feat_b))
            top_corr_pairs.append({
                "Feature_A": feat_a,
                "Feature_B": feat_b,
                "Correlation": round(val, 4)
            })
            
    top_corr_df = pd.DataFrame(top_corr_pairs)
    top_corr_path = os.path.join(paths["tables"], "top_correlations.csv")
    top_corr_df.to_csv(top_corr_path, index=False)
    logger.info(f"Saved top correlations list to: {top_corr_path}")
    
    # 2. Plot: Correlation Heatmap
    # Heatmaps are best viewed if we focus on the core clinical vital signs and key labs
    # to avoid a massive 35x35 block that is completely unreadable.
    # Let's plot the top 20 features with the highest variance or simply a selection of key clinical vitals + labs.
    key_features = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Glucose", "Lactate", "HCO3", "BUN", "Creatinine", "Hct", "WBC", "Platelets", "Age"]
    # Filter only if they exist in valid_cols
    key_features = [f for f in key_features if f in valid_cols]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df[key_features].corr(), 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix Heatmap of Key Clinical Vitals & Labs", pad=20)
    plt.tight_layout()
    
    fig_path = os.path.join(paths["figures"], "correlation_heatmap.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved correlation heatmap to: {fig_path}")
    
    return corr_matrix

if __name__ == "__main__":
    with Timer("Step 7 - Correlation Analysis"):
        run_correlation_analysis()
