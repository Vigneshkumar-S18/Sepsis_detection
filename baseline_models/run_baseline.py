# Main Orchestrator for Phase 6 Classical ML Baselines
import os
import sys
import gc
import time
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from baseline_models.config import MODEL_CONFIGS
from baseline_models.data_loader import load_dataset
from baseline_models.metrics import calculate_comprehensive_metrics
from baseline_models.plots import (
    save_confusion_matrix_plot, save_feature_importance_plot,
    save_comparative_roc_and_pr_plots
)
from baseline_models.save_model import save_checkpoint
from baseline_models.generate_report import (
    compile_baseline_pdf, compile_baseline_markdown_and_html
)

def run_baseline_pipeline():
    """
    Executes the classical machine learning baseline pipeline.
    Trains 5 models across Dataset A and Dataset B (10 runs),
    logs performance, creates plots, updates the leaderboard,
    and compiles summary reports.
    """
    checkpoints_dir = os.path.join(project_root, "experiments", "checkpoints")
    metrics_dir = os.path.join(project_root, "experiments", "metrics")
    logs_dir = os.path.join(project_root, "experiments", "logs")
    reports_dir = os.path.join(project_root, "reports", "summary")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    leaderboard_path = os.path.join(project_root, "experiments", "leaderboard.csv")
    
    # Store curves for comparison plot
    roc_curves = {}
    pr_curves = {}
    leaderboard_records = []

    datasets = ["Dataset A", "Dataset B"]
    models_to_train = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]

    for dataset in datasets:
        logger.info(f"\n{'='*70}")
        logger.info(f"LOADING DATASET: {dataset}")
        logger.info(f"{'='*70}")

        # Load splits
        with Timer(f"Loading {dataset} splits"):
            X_train, y_train = load_dataset(dataset, project_root, "train")
            X_val, y_val = load_dataset(dataset, project_root, "validation")
            X_test, y_test = load_dataset(dataset, project_root, "test")

        feature_names = list(X_train.columns)
        logger.info(f"  Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

        for model_name in models_to_train:
            run_id = f"{model_name}_{dataset.replace(' ', '_').lower()}"
            logger.info(f"\nTraining Model: {model_name} on {dataset}...")

            # Initialize model
            cfg = MODEL_CONFIGS[model_name]
            if model_name == "logistic_regression":
                model = LogisticRegression(**cfg)
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(**cfg)
            elif model_name == "random_forest":
                model = RandomForestClassifier(**cfg)
            elif model_name == "xgboost":
                model = XGBClassifier(**cfg)
            elif model_name == "lightgbm":
                model = LGBMClassifier(**cfg)
            else:
                logger.error(f"Unknown model name: {model_name}")
                continue

            # Train
            t0 = time.time()
            with Timer(f"  Fitting {model_name} on {dataset}"):
                model.fit(X_train, y_train)
            training_time = time.time() - t0

            # Predict probability and class on Validation & Test
            y_val_prob = model.predict_proba(X_val)[:, 1]
            y_val_pred = model.predict(X_val)

            y_test_prob = model.predict_proba(X_test)[:, 1]
            y_test_pred = model.predict(X_test)

            # Evaluate Metrics
            val_metrics = calculate_comprehensive_metrics(y_val, y_val_pred, y_val_prob)
            test_metrics = calculate_comprehensive_metrics(y_test, y_test_pred, y_test_prob)

            logger.info(f"  [Validation] AUROC: {val_metrics['auroc']:.4f} | AUPRC: {val_metrics['auprc']:.4f} | F1: {val_metrics['f1']:.4f} | Recall: {val_metrics['recall']:.4f}")
            logger.info(f"  [Test]       AUROC: {test_metrics['auroc']:.4f} | AUPRC: {test_metrics['auprc']:.4f} | F1: {test_metrics['f1']:.4f} | Recall: {test_metrics['recall']:.4f}")

            # Save curves comparison (use Test set for reporting)
            curve_key = f"{model_name.replace('_', ' ').title()} ({dataset})"
            roc_curves[curve_key] = (y_test, y_test_prob)
            pr_curves[curve_key] = (y_test, y_test_prob)

            # Save confusion matrix plot for validation and test
            cm_val_path = os.path.join(metrics_dir, f"{run_id}_confusion_val.png")
            cm_test_path = os.path.join(metrics_dir, f"{run_id}_confusion_test.png")
            
            save_confusion_matrix_plot(
                val_metrics["confusion_matrix"]["tn"], val_metrics["confusion_matrix"]["fp"],
                val_metrics["confusion_matrix"]["fn"], val_metrics["confusion_matrix"]["tp"],
                cm_val_path, title=f"Confusion Matrix - {model_name} ({dataset} Val)"
            )
            save_confusion_matrix_plot(
                test_metrics["confusion_matrix"]["tn"], test_metrics["confusion_matrix"]["fp"],
                test_metrics["confusion_matrix"]["fn"], test_metrics["confusion_matrix"]["tp"],
                cm_test_path, title=f"Confusion Matrix - {model_name} ({dataset} Test)"
            )

            # Save feature importance if tree-based (or coefficients if linear)
            imp_path = os.path.join(metrics_dir, f"{run_id}_importance.png")
            if model_name == "logistic_regression":
                coefs = np.abs(model.coef_[0])
                save_feature_importance_plot(coefs, feature_names, imp_path, title=f"Coef Importances - {model_name} ({dataset})")
            elif hasattr(model, "feature_importances_"):
                save_feature_importance_plot(model.feature_importances_, feature_names, imp_path, title=f"Feature Importances - {model_name} ({dataset})")

            # Save model checkpoint
            ds_suffix = "dataset_a" if dataset == "Dataset A" else "dataset_b"
            checkpoint_path = save_checkpoint(model, model_name, ds_suffix, checkpoints_dir)
            logger.info(f"  Model checkpoint saved to: {checkpoint_path}")

            # Append to leaderboard list (using Validation metrics for primary scores)
            leaderboard_records.append({
                "Experiment_ID": f"EXP_{ds_suffix.upper()}",
                "Model": model_name.replace("_", " ").title(),
                "Dataset": dataset,
                "Window_Size": "1h",
                "Horizon": "0h",
                "Learning_Rate": "N/A" if model_name not in ["xgboost", "lightgbm"] else str(cfg.get("learning_rate")),
                "Batch_Size": "N/A",
                "Epochs": "N/A",
                "Seed": 42,
                "AUROC": test_metrics["auroc"],
                "AUPRC": test_metrics["auprc"],
                "F1": test_metrics["f1"],
                "Recall": test_metrics["recall"],
                "Specificity": test_metrics["specificity"],
                "Training_Time": f"{training_time:.2f}s",
                "Git_Version": "v0.6",
                "Notes": f"Tabular classical ML run on {dataset}"
            })

            # Force garbage collection to manage RAM
            del model
            gc.collect()

    # Save leaderboard
    leaderboard_df = pd.DataFrame(leaderboard_records)
    leaderboard_df.to_csv(leaderboard_path, index=False)
    logger.info(f"\nLeaderboard CSV updated at: {leaderboard_path}")

    # Generate comparative ROC and PR plots on test sets
    logger.info("\nGenerating comparative ROC and Precision-Recall plots...")
    save_comparative_roc_and_pr_plots(roc_curves, pr_curves, metrics_dir)

    # Compile reports
    logger.info("\nCompiling baseline comparison reports...")
    compile_baseline_pdf(reports_dir, leaderboard_df, metrics_dir)
    compile_baseline_markdown_and_html(reports_dir, leaderboard_df)
    
    logger.info("\n" + "="*70)
    logger.info("Phase 6 — Baseline ML Models COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    run_baseline_pipeline()
