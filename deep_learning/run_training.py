# Main Orchestrator for Phase 7 Deep Temporal Learning Framework
import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from deep_learning.config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, POS_WEIGHT,
    EARLY_STOPPING_PATIENCE, DEVICE, MODEL_HYPERPARAMS, SCHEDULER_FACTOR, SCHEDULER_PATIENCE, WEIGHT_DECAY
)
from deep_learning.data_loader import get_sequence_dataloaders
from deep_learning.models.lstm import LSTMClassifier
from deep_learning.models.gru import GRUClassifier
from deep_learning.models.bilstm import BiLSTMClassifier
from deep_learning.models.transformer import TransformerClassifier
from deep_learning.trainer import SepsisDeepTrainer

# Re-use plots from classical baseline, or implement custom curve plotting here
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def set_all_seeds(seed=42):
    """
    Sets random seeds across numpy and PyTorch to guarantee reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_roc_and_pr_curves(results, save_dir):
    """
    Generates comparative ROC and PR curves for Deep Learning models on Test set.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. Comparative ROC Curves
    plt.figure(figsize=(8, 7))
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(data["test_labels"], data["test_probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUROC = {data['auroc']:.4f})", lw=1.5)
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Comparative Deep Learning ROC Curves (Test Set)", fontsize=13, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dl_comparative_roc.png"), dpi=150)
    plt.close()

    # 2. Comparative PR Curves
    plt.figure(figsize=(8, 7))
    for name, data in results.items():
        precision, recall, _ = precision_recall_curve(data["test_labels"], data["test_probs"])
        plt.plot(recall, precision, label=f"{name} (AUPRC = {data['auprc']:.4f})", lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Comparative Deep Learning PR Curves (Test Set)", fontsize=13, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Recall (Sensitivity)", fontsize=10)
    plt.ylabel("Precision", fontsize=10)
    plt.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dl_comparative_pr.png"), dpi=150)
    plt.close()


def save_confusion_matrix_heatmap(tn, fp, fn, tp, path, title):
    """
    Saves a styled confusion matrix heatmap for a deep learning model.
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
    plt.title(title, fontsize=12, pad=15, color='#0f766e', weight='bold')
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compile_deep_learning_reports(leaderboard_df, reports_dir):
    """
    Generates Markdown, HTML, and PDF Deep Learning benchmarking reports.
    """
    os.makedirs(reports_dir, exist_ok=True)
    md_path = os.path.join(reports_dir, "DeepLearning_Report.md")
    html_path = os.path.join(reports_dir, "DeepLearning_Report.html")
    pdf_path = os.path.join(reports_dir, "DeepLearning_Report.pdf")

    # 1. Markdown Report
    md_lines = [
        "# THAARU Sepsis AI — Deep Learning Evaluation Report",
        f"**Generated:** {time.strftime('%B %d, %Y')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "This report documents Phase 7: Deep Temporal Learning. We benchmarked 4 sequential neural architectures "
        "(LSTM, GRU, BiLSTM, and Transformer Encoder) using the baseline `w12_h0` sequence configurations. "
        "The best performing architecture was then evaluated across multiple observation window lengths and prediction horizons.",
        "",
        "## 2. Deep Learning Leaderboard",
        "",
        "| Model | Dataset Config | Window | Horizon | AUROC | AUPRC | F1-Score | Recall (Sens) | Specificity | Parameters |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for _, r in leaderboard_df.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['Config_ID']} | {r['Window_Size']} | {r['Horizon']} "
            f"| {r['AUROC']:.4f} | {r['AUPRC']:.4f} | {r['F1']:.4f} | {r['Recall']:.4f} "
            f"| {r['Specificity']:.4f} | {r['Parameters']:,} |"
        )
    
    md_lines += [
        "",
        "## 3. Comparative Findings",
        "* **Sequence Models vs XGBoost Baseline:** Deep sequence models leverage the multi-hour observation window to track trajectory changes. Direct comparison shows whether temporal representations outperform static feature boosting (XGBoost baseline Test AUROC 0.8381, AUPRC 0.1318).",
        "* **Best Architecture:** Bidirectional LSTM and Transformer models are compared on context retention, representing the most powerful components of our temporal pipeline.",
    ]
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    # 2. HTML Report
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAARU Sepsis AI — Deep Learning Evaluation Report</title>
    <style>
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 40px 20px; }
        .container { max-width: 950px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h1 { color: #0f766e; font-size: 28px; border-bottom: 3px solid #0f766e; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #0f766e; font-size: 20px; margin-top: 30px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; }
        .meta { color: #475569; font-style: italic; margin-bottom: 30px; }
        table { width: 100%%; border-collapse: collapse; margin: 20px 0; }
        th, td { text-align: left; padding: 12px; border: 1px solid #cbd5e1; }
        th { background: #0f766e; color: white; }
        tr:nth-child(even) { background: #f1f5f9; }
    </style>
</head>
<body>
<div class="container">
    <h1>THAARU Sepsis AI — Deep Learning Evaluation Report</h1>
    <div class="meta">Generated: %s</div>

    <h2>1. Executive Summary</h2>
    <p>This report documents Phase 7: Deep Temporal Learning benchmarks across sequential neural networks compared against our classical tree-based baselines.</p>

    <h2>2. Deep Learning Leaderboard</h2>
    <table>
        <tr><th>Model</th><th>Config ID</th><th>Window</th><th>Horizon</th><th>AUROC</th><th>AUPRC</th><th>F1-Score</th><th>Recall (Sens)</th><th>Specificity</th><th>Parameters</th></tr>
        %s
    </table>

    <h2>3. Comparative Findings</h2>
    <ul>
        <li><b>Attention vs Recurrence:</b> The Transformer model captures global context over sequence history via parallel multi-head attention, whereas BiLSTMs track recurrences.</li>
        <li><b>Comparison ceiling:</b> High-performance benchmarks directly evaluate deep learning against XGBoost classical baselines.</li>
    </ul>
</div>
</body>
</html>"""

    rows_html = ""
    for _, r in leaderboard_df.iterrows():
        rows_html += (
            f"<tr><td>{r['Model']}</td><td>{r['Config_ID']}</td>"
            f"<td>{r['Window_Size']}</td><td>{r['Horizon']}</td>"
            f"<td>{r['AUROC']:.4f}</td><td>{r['AUPRC']:.4f}</td>"
            f"<td>{r['F1']:.4f}</td><td>{r['Recall']:.4f}</td>"
            f"<td>{r['Specificity']:.4f}</td><td>{r['Parameters']:,}</td></tr>\n"
        )
    html_content = html_template % (time.strftime('%B %d, %Y'), rows_html)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 3. PDF Report (ReportLab)
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
        fontSize=18, textColor=colors.HexColor('#0f766e'), alignment=1, spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique',
        fontSize=10, textColor=colors.HexColor('#475569'), alignment=1, spaceAfter=25
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading2'], fontName='Helvetica-Bold',
        fontSize=13, textColor=colors.HexColor('#0f766e'), spaceBefore=12, spaceAfter=8, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyText', parent=styles['Normal'], fontName='Helvetica',
        fontSize=9.5, leading=13, textColor=colors.HexColor('#1e293b'), spaceAfter=8
    )
    th_style = ParagraphStyle(
        'TH', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=8.5, textColor=colors.white
    )
    td_style = ParagraphStyle(
        'TD', parent=styles['Normal'], fontName='Helvetica',
        fontSize=8, leading=10
    )

    story = [
        Spacer(1, 10),
        Paragraph("THAARU Sepsis AI — Deep Learning Benchmarks", title_style),
        Paragraph(f"Phase 7 Deep Learning Report &bull; Generated: {time.strftime('%B %d, %Y')}", subtitle_style),
        Paragraph("1. Executive Summary", h1_style),
        Paragraph(
            "We constructed sequential neural network frameworks including LSTM, GRU, BiLSTM, and "
            "Transformers. Models were optimized via weighted BCE loss to handle class imbalance, "
            "and early stopping monitored validation AUPRC directly to prevent overfitting.",
            body_style
        ),
        Paragraph("2. Deep Learning Leaderboard", h1_style)
    ]

    pdf_rows = [[
        Paragraph("Model", th_style), Paragraph("Config", th_style),
        Paragraph("AUROC", th_style), Paragraph("AUPRC", th_style),
        Paragraph("F1-Score", th_style), Paragraph("Recall", th_style),
        Paragraph("Parameters", th_style),
    ]]
    for _, r in leaderboard_df.iterrows():
        pdf_rows.append([
            Paragraph(str(r["Model"]), td_style),
            Paragraph(str(r["Config_ID"]), td_style),
            Paragraph(f"{r['AUROC']:.4f}", td_style),
            Paragraph(f"{r['AUPRC']:.4f}", td_style),
            Paragraph(f"{r['F1']:.4f}", td_style),
            Paragraph(f"{r['Recall']:.4f}", td_style),
            Paragraph(f"{r['Parameters']:,}", td_style),
        ])
    t = Table(pdf_rows, colWidths=[110, 80, 70, 70, 70, 70, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    doc.build(story)


def run_deep_learning_pipeline():
    set_all_seeds(SEED)
    
    sequences_root = os.path.join(project_root, "datasets", "sequences")
    checkpoints_dir = os.path.join(project_root, "experiments", "checkpoints")
    plots_dir = os.path.join(project_root, "experiments", "metrics")
    reports_dir = os.path.join(project_root, "reports", "summary")
    predictions_dir = os.path.join(project_root, "experiments", "predictions")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    leaderboard_records = []
    comparison_curves_data = {}

    # ── Milestone 7A: Train LSTM, GRU, BiLSTM, Transformer on w12_h0 ────
    base_config_id = "w12_h0"
    base_seq_dir = os.path.join(sequences_root, base_config_id)
    
    SAMPLE_PCT = 0.20  # Use 20% sample size to accelerate local CPU training
    train_loader, val_loader, test_loader = get_sequence_dataloaders(
        base_seq_dir, BATCH_SIZE, sample_pct=SAMPLE_PCT, logger=logger
    )

    models_to_test = {
        "LSTM": LSTMClassifier(**MODEL_HYPERPARAMS["lstm"]),
        "GRU": GRUClassifier(**MODEL_HYPERPARAMS["gru"]),
        "BiLSTM": BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"]),
        "Transformer": TransformerClassifier(**MODEL_HYPERPARAMS["transformer"])
    }

    best_model_name = None
    best_val_auprc = -1.0

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT], device=DEVICE))

    for name, model in models_to_test.items():
        logger.info(f"\n{'-'*70}")
        logger.info(f"Milestone 7A: Training {name} on {base_config_id}...")
        logger.info(f"{'-'*70}")

        model.print_summary(logger=logger)
        n_params = model.count_trainable_parameters()

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )

        trainer = SepsisDeepTrainer(
            model=model, device=DEVICE, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, early_stopping_patience=EARLY_STOPPING_PATIENCE,
            checkpoints_dir=checkpoints_dir, logger=logger
        )

        # Train model
        fit_history = trainer.fit(train_loader, val_loader, EPOCHS, f"{name.lower()}_{base_config_id}")

        # Evaluate on Test set
        test_loss, test_auroc, test_auprc, test_probs, test_labels = trainer.evaluate(test_loader)
        
        # Binary predictions thresholded at 0.5 probability
        test_preds = (test_probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * (tp / (tp + fp + fn + tp) if (tp + fp + fn + tp) > 0 else 0.0)  # quick F1 fallback
        from sklearn.metrics import f1_score
        f1 = f1_score(test_labels, test_preds, zero_division=0)

        logger.info(f"  [Test Results] {name} — AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

        # Save predictions for explainability
        pred_path = os.path.join(predictions_dir, f"{name.lower()}_{base_config_id}_test_preds.npz")
        np.savez_compressed(pred_path, probs=test_probs, labels=test_labels)

        # Save confusion matrix plot
        save_confusion_matrix_heatmap(
            tn, fp, fn, tp,
            os.path.join(plots_dir, f"{name.lower()}_{base_config_id}_confusion_test.png"),
            title=f"Confusion Matrix — {name} ({base_config_id} Test)"
        )

        # Track curves
        comparison_curves_data[name] = {
            "test_labels": test_labels,
            "test_probs": test_probs,
            "auroc": test_auroc,
            "auprc": test_auprc
        }

        # Track records
        leaderboard_records.append({
            "Model": name,
            "Config_ID": base_config_id,
            "Window_Size": "12h",
            "Horizon": "0h",
            "AUROC": test_auroc,
            "AUPRC": test_auprc,
            "F1": f1,
            "Recall": recall,
            "Specificity": specificity,
            "Parameters": n_params,
            "Notes": "Milestone 7A validation config"
        })

        # Tracking best validation AUPRC to select best model for Milestone 7B
        val_best_auprc = trainer.best_val_auprc
        if val_best_auprc > best_val_auprc:
            best_val_auprc = val_best_auprc
            best_model_name = name

        # Cleanup memory
        del model, trainer
        gc.collect()

    logger.info(f"\n{'='*70}")
    logger.info(f"Milestone 7A Complete! Top Deep Learning Model: {best_model_name}")
    logger.info(f"{'='*70}")

    # ── Milestone 7B: Train best model on remaining configurations ────────
    other_configs = ["w6_h0", "w24_h0", "w12_h3", "w12_h6"]
    
    for cfg_id in other_configs:
        logger.info(f"\n{'-'*70}")
        logger.info(f"Milestone 7B: Training best model ({best_model_name}) on config {cfg_id}...")
        logger.info(f"{'-'*70}")

        cfg_seq_dir = os.path.join(sequences_root, cfg_id)
        train_loader, val_loader, test_loader = get_sequence_dataloaders(
            cfg_seq_dir, BATCH_SIZE, sample_pct=SAMPLE_PCT, logger=logger
        )

        # Re-initialize the best model structure
        if best_model_name == "LSTM":
            model = LSTMClassifier(**MODEL_HYPERPARAMS["lstm"])
        elif best_model_name == "GRU":
            model = GRUClassifier(**MODEL_HYPERPARAMS["gru"])
        elif best_model_name == "BiLSTM":
            model = BiLSTMClassifier(**MODEL_HYPERPARAMS["bilstm"])
        elif best_model_name == "Transformer":
            model = TransformerClassifier(**MODEL_HYPERPARAMS["transformer"])
        else:
            raise ValueError(f"Unknown best model selection: {best_model_name}")

        n_params = model.count_trainable_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )

        trainer = SepsisDeepTrainer(
            model=model, device=DEVICE, criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, early_stopping_patience=EARLY_STOPPING_PATIENCE,
            checkpoints_dir=checkpoints_dir, logger=logger
        )

        trainer.fit(train_loader, val_loader, EPOCHS, f"{best_model_name.lower()}_{cfg_id}")

        # Test set evaluation
        test_loss, test_auroc, test_auprc, test_probs, test_labels = trainer.evaluate(test_loader)
        test_preds = (test_probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(test_labels, test_preds, zero_division=0)

        logger.info(f"  [Test Results] {best_model_name} ({cfg_id}) — AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

        # Save predictions
        pred_path = os.path.join(predictions_dir, f"{best_model_name.lower()}_{cfg_id}_test_preds.npz")
        np.savez_compressed(pred_path, probs=test_probs, labels=test_labels)

        # Save confusion matrix
        save_confusion_matrix_heatmap(
            tn, fp, fn, tp,
            os.path.join(plots_dir, f"{best_model_name.lower()}_{cfg_id}_confusion_test.png"),
            title=f"Confusion Matrix — {best_model_name} ({cfg_id} Test)"
        )

        # Parse config fields for leaderboard
        window_size = "6h" if "w6" in cfg_id else "24h" if "w24" in cfg_id else "12h"
        horizon = "+3h" if "h3" in cfg_id else "+6h" if "h6" in cfg_id else "0h"

        leaderboard_records.append({
            "Model": f"{best_model_name} (Selected)",
            "Config_ID": cfg_id,
            "Window_Size": window_size,
            "Horizon": horizon,
            "AUROC": test_auroc,
            "AUPRC": test_auprc,
            "F1": f1,
            "Recall": recall,
            "Specificity": specificity,
            "Parameters": n_params,
            "Notes": f"Milestone 7B experimental matrix on {cfg_id}"
        })

        # Track curves
        curve_key = f"{best_model_name} ({cfg_id})"
        comparison_curves_data[curve_key] = {
            "test_labels": test_labels,
            "test_probs": test_probs,
            "auroc": test_auroc,
            "auprc": test_auprc
        }

        # Cleanup memory
        del model, trainer
        gc.collect()

    # Save Deep Learning Leaderboard
    leaderboard_df = pd.DataFrame(leaderboard_records)
    dl_leaderboard_path = os.path.join(project_root, "experiments", "DeepLearning_Leaderboard.csv")
    leaderboard_df.to_csv(dl_leaderboard_path, index=False)
    logger.info(f"\nDeep Learning Leaderboard saved at: {dl_leaderboard_path}")

    # Generate comparative ROC and PR curves on Test set
    logger.info("\nPlotting comparative deep learning curves...")
    plot_roc_and_pr_curves(comparison_curves_data, plots_dir)

    # Compile reports
    logger.info("\nCompiling deep learning summary reports...")
    compile_deep_learning_reports(leaderboard_df, reports_dir)

    logger.info("\n" + "="*70)
    logger.info("Phase 7 — Deep Temporal Learning Framework COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    with Timer("Phase 7 — Deep Learning Framework Pipeline"):
        run_deep_learning_pipeline()
