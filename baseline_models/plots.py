# Performance Curves and Matrix Plotter
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

# Use a clean aesthetic style for plot generation
plt.style.use('ggplot')
sns.set_theme(style='whitegrid')


def save_confusion_matrix_plot(tn, fp, fn, tp, path, title="Confusion Matrix"):
    """
    Saves a styled confusion matrix heatmap.
    """
    cm = np.array([[tn, fp], [fn, tp]])
    labels = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"]
    ]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=False,
        xticklabels=["Non-Sepsis", "Sepsis"],
        yticklabels=["Non-Sepsis", "Sepsis"],
        annot_kws={"fontsize": 12, "weight": "bold"}
    )
    plt.title(title, fontsize=13, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Predicted Class", fontsize=10)
    plt.ylabel("True Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_feature_importance_plot(importances, feature_names, path, title="Feature Importance"):
    """
    Saves a horizontal bar plot showing the top 15 features by importance.
    """
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:15]

    plt.figure(figsize=(9, 6))
    sns.barplot(
        x=importances[top_indices],
        y=np.array(feature_names)[top_indices],
        palette="viridis",
        hue=np.array(feature_names)[top_indices],
        legend=False
    )
    plt.title(title, fontsize=13, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Relative Importance Score", fontsize=10)
    plt.ylabel("Clinical Feature Name", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparative_roc_and_pr_plots(roc_curves, pr_curves, output_dir):
    """
    Plots all ROC and PR curves in comparative charts to directly evaluate
    Dataset A (68 features) vs Dataset B (97 features) across algorithms.
    """
    # 1. ROC Curves Plot
    plt.figure(figsize=(8, 7))
    for name, (y_true, y_prob) in roc_curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=name, lw=1.5)
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Comparative ROC Curves (Baselines)", fontsize=13, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_roc.png"), dpi=150)
    plt.close()

    # 2. PR Curves Plot
    plt.figure(figsize=(8, 7))
    for name, (y_true, y_prob) in pr_curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=name, lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Comparative Precision-Recall Curves (Baselines)", fontsize=13, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Recall (Sensitivity)", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparative_pr.png"), dpi=150)
    plt.close()
