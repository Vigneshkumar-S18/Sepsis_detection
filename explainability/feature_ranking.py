# Feature Ranking Comparison Module
import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from explainability.config import OUTPUTS_DIR


def compare_feature_rankings():
    logger.info("Starting Module 5: Feature Ranking Comparison...")
    
    shap_path = os.path.join(OUTPUTS_DIR, "shap_feature_rankings.csv")
    ig_path = os.path.join(OUTPUTS_DIR, "ig_feature_rankings.csv")
    
    if not os.path.exists(shap_path) or not os.path.exists(ig_path):
        raise FileNotFoundError("SHAP and/or Integrated Gradients ranking files are missing. Run shap_analysis and integrated_gradients first.")

    shap_df = pd.read_csv(shap_path)
    ig_df = pd.read_csv(ig_path)

    # Add ranks (1-based)
    shap_df["SHAP_Rank"] = shap_df.index + 1
    ig_df["IG_Rank"] = ig_df.index + 1

    # Merge on Feature name
    merged_df = pd.merge(
        shap_df[["Feature", "SHAP_Rank", "Mean_Abs_SHAP"]],
        ig_df[["Feature", "IG_Rank", "Mean_Abs_Attribution"]],
        on="Feature"
    )

    # Save merged comparison
    comparison_path = os.path.join(OUTPUTS_DIR, "feature_rankings_comparison.csv")
    merged_df.to_csv(comparison_path, index=False)
    logger.info(f"  Saved comparison matrix to: {comparison_path}")

    # Extract Top 20 for both models
    top_20_shap = shap_df.head(20)[["Feature", "SHAP_Rank"]]
    top_20_ig = ig_df.head(20)[["Feature", "IG_Rank"]]
    
    # Check overlap
    overlap = set(top_20_shap["Feature"]).intersection(set(top_20_ig["Feature"]))
    logger.info(f"  Top 20 Features Overlap count: {len(overlap)} features")
    logger.info(f"  Shared Top Features: {list(overlap)}")

    return merged_df, top_20_shap, top_20_ig, overlap


if __name__ == "__main__":
    compare_feature_rankings()
