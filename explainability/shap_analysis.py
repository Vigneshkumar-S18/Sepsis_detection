# SHAP Analysis Module for XGBoost Explanations
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from explainability.config import VISUALIZATIONS_DIR, OUTPUTS_DIR
from explainability.data_loader import load_tabular_data


def run_shap_explanations():
    logger.info("Starting Module 1: SHAP Explanations on XGBoost Model...")
    
    # Load model
    model_path = os.path.join(project_root, "experiments", "checkpoints", "xgboost_dataset_b.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost baseline model not found at: {model_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load data (Test split)
    X, y = load_tabular_data("test")
    
    # Subsample test set for fast SHAP evaluation on CPU (e.g. 100 samples)
    # This preserves the underlying dataset distribution while preventing CPU timeouts
    np.random.seed(42)
    sample_indices = np.random.choice(len(X), size=100, replace=False)
    X_sample = X.iloc[sample_indices]
    
    # Initialize KernelExplainer on predict_proba to bypass version-specific binary parsing bugs in TreeExplainer
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    background_data = shap.sample(X, 30)
    explainer = shap.KernelExplainer(predict_fn, background_data)
    
    # Compute raw SHAP values
    shap_values_raw = explainer.shap_values(X_sample)
    
    # Format into shap.Explanation object for downstream plotting compatibility
    shap_values = shap.Explanation(
        values=shap_values_raw,
        base_values=np.array([explainer.expected_value] * len(X_sample)),
        data=X_sample.values,
        feature_names=list(X_sample.columns)
    )

    # 1. Save SHAP Summary Bar Plot (Global Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("XGBoost Global Feature Importance (Mean Absolute SHAP Value)", fontsize=12, pad=15, color='#0f766e', weight='bold')
    plt.tight_layout()
    summary_path = os.path.join(VISUALIZATIONS_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    logger.info(f"  Saved SHAP Summary Plot to: {summary_path}")

    # 2. Save SHAP Beeswarm Plot (Impact of high/low values)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=False)
    plt.title("XGBoost SHAP Beeswarm Plot", fontsize=12, pad=15, color='#0f766e', weight='bold')
    plt.tight_layout()
    beeswarm_path = os.path.join(VISUALIZATIONS_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    logger.info(f"  Saved SHAP Beeswarm Plot to: {beeswarm_path}")

    # Get the top features based on mean absolute SHAP value
    mean_shap = np.abs(shap_values.values).mean(axis=0)
    shap_ranking_df = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Abs_SHAP": mean_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    
    # Save Feature Rankings CSV
    ranking_path = os.path.join(OUTPUTS_DIR, "shap_feature_rankings.csv")
    shap_ranking_df.to_csv(ranking_path, index=False)
    logger.info(f"  Saved SHAP rankings to: {ranking_path}")

    # 3. Save Dependence Plots for top 2 features
    top_features = shap_ranking_df["Feature"].head(2).tolist()
    for idx, feature in enumerate(top_features, 1):
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feature, shap_values.values, X_sample, show=False)
        plt.title(f"SHAP Dependence Plot: {feature}", fontsize=11, pad=15, color='#0f766e', weight='bold')
        plt.tight_layout()
        dep_path = os.path.join(VISUALIZATIONS_DIR, f"shap_dependence_{idx}.png")
        plt.savefig(dep_path, dpi=150)
        plt.close()
        logger.info(f"  Saved SHAP Dependence Plot for '{feature}' to: {dep_path}")

    return shap_ranking_df, explainer, X, shap_values, sample_indices


if __name__ == "__main__":
    run_shap_explanations()
