# Comprehensive Evaluation Metrics for Sepsis Prediction
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score
)


def calculate_comprehensive_metrics(y_true, y_pred, y_prob):
    """
    Computes all standard research metrics for evaluation.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0 or 1).
    y_pred : array-like
        Binary predictions (0 or 1).
    y_prob : array-like
        Probability estimates of the positive class.

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Sensitivity/Recall
    sensitivity = recall_score(y_true, y_pred, zero_division=0)

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision
    precision = precision_score(y_true, y_pred, zero_division=0)

    # F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Area Under ROC Curve
    auroc = roc_auc_score(y_true, y_prob)

    # Area Under Precision-Recall Curve (AUPRC)
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recalls, precisions)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "precision": float(precision),
        "recall": float(sensitivity), # Sensitivity is equivalent to Recall
        "f1": float(f1),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
