# XGBoost Hyperparameter Optimization Module
import os
import sys
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger
from optimization.config import XGB_PARAM_GRID, OUTPUTS_DIR, SEED
from optimization.data_loader import load_tabular_splits


def run_xgboost_optimization(num_trials=5):
    logger.info("Running hyperparameter optimization for XGBoost...")
    
    # Load splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_tabular_splits()

    # Subsample training data to 20% to speed up CPU tuning significantly
    # while maintaining representative patterns
    np.random.seed(SEED)
    train_size = int(len(X_train) * 0.20)
    train_indices = np.random.choice(len(X_train), size=train_size, replace=False)
    X_train_sub = X_train.iloc[train_indices]
    y_train_sub = y_train.iloc[train_indices]

    # Generate random combinations from XGB_PARAM_GRID
    trials = []
    
    # Force the default baseline parameters as Trial 1 for direct relative comparison
    trials.append({
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "colsample_bytree": 1.0
    })

    # Sample remaining trials randomly
    for trial_idx in range(1, num_trials):
        trial = {}
        for param, options in XGB_PARAM_GRID.items():
            trial[param] = float(np.random.choice(options)) if isinstance(options[0], float) else int(np.random.choice(options))
        trials.append(trial)

    results = []
    best_auprc = -1.0
    best_params = None

    for idx, params in enumerate(trials):
        logger.info(f"  Trial {idx+1}/{num_trials} — Parameters: {params}")
        
        start_time = time.time()
        
        # Instantiate XGBoost with balanced class weights
        model = XGBClassifier(
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            n_estimators=int(params["n_estimators"]),
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=54.5,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1
        )
        
        model.fit(X_train_sub, y_train_sub)
        
        # Predict on validation split
        val_probs = model.predict_proba(X_val)[:, 1]
        val_auprc = average_precision_score(y_val, val_probs)
        val_auroc = roc_auc_score(y_val, val_probs)
        
        elapsed = time.time() - start_time
        logger.info(f"    Validation AUPRC: {val_auprc:.4f} | Validation AUROC: {val_auroc:.4f} | Duration: {elapsed:.1f}s")
        
        trial_result = {
            "trial": idx + 1,
            "max_depth": int(params["max_depth"]),
            "learning_rate": params["learning_rate"],
            "n_estimators": int(params["n_estimators"]),
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "val_auprc": val_auprc,
            "val_auroc": val_auroc,
            "duration_sec": elapsed
        }
        results.append(trial_result)
        
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_params = params

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUTS_DIR, "xgboost_trials.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"  XGBoost trials logged to: {results_path}")
    logger.info(f"  Best XGBoost Parameters found: {best_params} (Val AUPRC: {best_auprc:.4f})")
    
    return best_params, best_auprc


if __name__ == "__main__":
    run_xgboost_optimization()
