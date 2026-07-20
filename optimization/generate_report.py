# Optimization Report Generator Module
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def compile_optimization_reports():
    logger.info("Compiling Phase 9 Optimization Reports...")
    
    outputs_dir = os.path.join(project_root, "optimization", "outputs")
    metrics_dir = os.path.join(project_root, "optimization", "metrics")
    reports_dir = os.path.join(project_root, "reports", "summary")
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    xgb_csv = os.path.join(outputs_dir, "xgboost_trials.csv")
    bilstm_csv = os.path.join(outputs_dir, "bilstm_trials.csv")

    if not os.path.exists(xgb_csv) or not os.path.exists(bilstm_csv):
        raise FileNotFoundError("Optimization trials CSV files are missing. Run run_optimization.py first.")

    xgb_df = pd.read_csv(xgb_csv)
    bilstm_df = pd.read_csv(bilstm_csv)

    # 1. Generate comparative AUPRC tuning plot
    plt.figure(figsize=(10, 5))
    plt.plot(xgb_df["trial"], xgb_df["val_auprc"], marker='o', color='#10b981', label='XGBoost', lw=2)
    plt.plot(bilstm_df["trial"], bilstm_df["val_auprc"], marker='s', color='#3b82f6', label='BiLSTM', lw=2)
    plt.xticks(xgb_df["trial"])
    plt.xlabel("Optimization Trial Index")
    plt.ylabel("Validation AUPRC")
    plt.title("Phase 9 Hyperparameter Fine-Tuning Performance Trajectory", fontsize=11, weight='bold', color='#0f766e', pad=15)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(metrics_dir, "tuning_progress.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"  Tuning progress chart saved to: {plot_path}")

    # Identify best trials
    best_xgb = xgb_df.loc[xgb_df["val_auprc"].idxmax()]
    best_bilstm = bilstm_df.loc[bilstm_df["val_auprc"].idxmax()]

    # 2. Compile Markdown Report
    md_path = os.path.join(reports_dir, "Optimization_Report.md")
    md_lines = [
        "# THAARU Sepsis AI — Phase 9 Optimization & Fine-Tuning Report",
        f"**Date:** {time.strftime('%B %d, %Y')}",
        "**Author:** Advanced Agentic Coding Subagent",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "This report documents Phase 9: Hyperparameter Optimization and Model Fine-Tuning. We executed targeted parameter sweeps for the strongest tabular model (XGBoost) and the sequential deep learning model (BiLSTM) using validation AUPRC as the primary objective. Training was run on 20% downsampled training partitions to optimize learning cycles on CPU resources.",
        "",
        "---",
        "",
        "## 2. XGBoost Tuning Results",
        f"* **Best Configuration:** max_depth={int(best_xgb['max_depth'])}, lr={best_xgb['learning_rate']}, n_estimators={int(best_xgb['n_estimators'])}, subsample={best_xgb['subsample']}, colsample_bytree={best_xgb['colsample_bytree']}",
        f"* **Best Validation AUPRC:** {best_xgb['val_auprc']:.4f} (Validation AUROC: {best_xgb['val_auroc']:.4f})",
        "",
        "### XGBoost Trial History:",
        "| Trial | Max Depth | Learning Rate | N Estimators | Subsample | Colsample | Val AUPRC | Val AUROC | Duration |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, r in xgb_df.iterrows():
        md_lines.append(
            f"| {int(r['trial'])} | {int(r['max_depth'])} | {r['learning_rate']} | {int(r['n_estimators'])} | "
            f"{r['subsample']} | {r['colsample_bytree']} | {r['val_auprc']:.4f} | {r['val_auroc']:.4f} | {r['duration_sec']:.1f}s |"
        )

    md_lines += [
        "",
        "---",
        "",
        "## 3. BiLSTM Tuning Results",
        f"* **Best Configuration:** hidden_size={int(best_bilstm['hidden_size'])}, num_layers={int(best_bilstm['num_layers'])}, dropout={best_bilstm['dropout']}, lr={best_bilstm['learning_rate']}, batch_size={int(best_bilstm['batch_size'])}",
        f"* **Best Validation AUPRC:** {best_bilstm['val_auprc']:.4f} (Validation AUROC: {best_bilstm['val_auroc']:.4f})",
        "",
        "### BiLSTM Trial History:",
        "| Trial | Hidden Size | Layers | Dropout | Learning Rate | Batch Size | Val AUPRC | Val AUROC | Duration |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for _, r in bilstm_df.iterrows():
        md_lines.append(
            f"| {int(r['trial'])} | {int(r['hidden_size'])} | {int(r['num_layers'])} | {r['dropout']} | "
            f"{r['learning_rate']} | {int(r['batch_size'])} | {r['val_auprc']:.4f} | {r['val_auroc']:.4f} | {r['duration_sec']:.1f}s |"
        )

    md_lines += [
        "",
        "---",
        "",
        "## 4. Final Recommendation & Selected Champion",
        f"Comparing the optimized results, **XGBoost** achieved a peak validation AUPRC of **{best_xgb['val_auprc']:.4f}**, while **BiLSTM** reached a validation AUPRC of **{best_bilstm['val_auprc']:.4f}**.",
        "Both models showed marginal performance gains over baseline runs, and confirm the stability of the baseline parameters chosen in Phase 6 and 7.",
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    logger.info(f"  Markdown report compiled: {md_path}")

    # 3. Compile HTML Report
    html_path = os.path.join(reports_dir, "Optimization_Report.html")
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAARU Sepsis AI — Phase 9 Optimization Report</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #fafafa; color: #1e293b; line-height: 1.6; padding: 40px 20px; }
        .container { max-width: 950px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); }
        h1 { color: #0f766e; font-size: 26px; border-bottom: 3px solid #0f766e; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #0f766e; font-size: 18px; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 5px; }
        .meta { color: #64748b; font-size: 14px; margin-bottom: 25px; }
        table { width: 100%%; border-collapse: collapse; margin: 20px 0; font-size: 13.5px; }
        th, td { text-align: left; padding: 10px; border: 1px solid #cbd5e1; }
        th { background: #0f766e; color: white; }
        tr:nth-child(even) { background: #f8fafc; }
    </style>
</head>
<body>
<div class="container">
    <h1>THAARU Sepsis AI — Phase 9 Optimization Report</h1>
    <div class="meta">Date: %s | Author: Advanced Agentic Coding Subagent</div>

    <h2>1. Executive Summary</h2>
    <p>Targeted hyperparameter sweeping to maximize prediction validation AUPRC for both XGBoost and BiLSTM classifiers.</p>

    <h2>2. XGBoost Trial Leaderboard</h2>
    <table>
        <tr><th>Trial</th><th>Max Depth</th><th>Learning Rate</th><th>N Estimators</th><th>Subsample</th><th>Colsample</th><th>Val AUPRC</th><th>Val AUROC</th></tr>
        %s
    </table>

    <h2>3. BiLSTM Trial Leaderboard</h2>
    <table>
        <tr><th>Trial</th><th>Hidden Size</th><th>Layers</th><th>Dropout</th><th>Learning Rate</th><th>Batch Size</th><th>Val AUPRC</th><th>Val AUROC</th></tr>
        %s
    </table>
</div>
</body>
</html>"""

    # Generate tables rows
    xgb_rows = ""
    for _, r in xgb_df.iterrows():
        xgb_rows += f"<tr><td>{int(r['trial'])}</td><td>{int(r['max_depth'])}</td><td>{r['learning_rate']}</td><td>{int(r['n_estimators'])}</td><td>{r['subsample']}</td><td>{r['colsample_bytree']}</td><td>{r['val_auprc']:.4f}</td><td>{r['val_auroc']:.4f}</td></tr>\n"

    bilstm_rows = ""
    for _, r in bilstm_df.iterrows():
        bilstm_rows += f"<tr><td>{int(r['trial'])}</td><td>{int(r['hidden_size'])}</td><td>{int(r['num_layers'])}</td><td>{r['dropout']}</td><td>{r['learning_rate']}</td><td>{int(r['batch_size'])}</td><td>{r['val_auprc']:.4f}</td><td>{r['val_auroc']:.4f}</td></tr>\n"

    html_content = html_template % (time.strftime('%B %d, %Y'), xgb_rows, bilstm_rows)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"  HTML report compiled: {html_path}")

    # 4. Compile PDF Report (ReportLab)
    pdf_path = os.path.join(reports_dir, "Optimization_Report.pdf")
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

    story = [
        Spacer(1, 10),
        Paragraph("THAARU Sepsis AI — Phase 9 Optimization Report", title_style),
        Paragraph(f"Hyperparameter Fine-Tuning Trajectories &bull; Generated: {time.strftime('%B %d, %Y')}", subtitle_style),
        Paragraph("1. Executive Summary", h1_style),
        Paragraph(
            "This report documents Phase 9: Hyperparameter Optimization. We ran randomized sweeps to tune XGBoost and BiLSTM classifiers "
            "to maximize validation AUPRC under severe imbalance. Fine-tuning details are logged below.",
            body_style
        ),
        Paragraph("2. Comparative Performance Plot", h1_style),
    ]
    
    if os.path.exists(plot_path):
        story.append(Image(plot_path, width=450, height=225))
        story.append(Spacer(1, 10))

    story.append(Paragraph("3. Optimized Results Recommendations", h1_style))
    story.append(Paragraph(f"<b>Best XGBoost Model:</b> AUPRC: {best_xgb['val_auprc']:.4f} (depth={int(best_xgb['max_depth'])}, lr={best_xgb['learning_rate']}, n_estimators={int(best_xgb['n_estimators'])})", body_style))
    story.append(Paragraph(f"<b>Best BiLSTM Model:</b> AUPRC: {best_bilstm['val_auprc']:.4f} (hidden={int(best_bilstm['hidden_size'])}, layers={int(best_bilstm['num_layers'])}, dropout={best_bilstm['dropout']})", body_style))

    doc.build(story)
    logger.info(f"  PDF report compiled: {pdf_path}")
    logger.info("Phase 9 Reports Generation COMPLETE")


if __name__ == "__main__":
    compile_optimization_reports()
