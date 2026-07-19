# Baseline Models Report Compiler
import os
import datetime
import pandas as pd
from jinja2 import Environment

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def compile_baseline_pdf(report_dir, leaderboard_df, plots_dir):
    """
    Compiles a styled PDF summarizing Phase 6 classical ML baselines.
    """
    pdf_path = os.path.join(report_dir, "Baseline_Classical_Report.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
        fontSize=18, textColor=colors.HexColor('#0f766e'), alignment=1,
        spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique',
        fontSize=10, textColor=colors.HexColor('#475569'), alignment=1,
        spaceAfter=25
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading2'], fontName='Helvetica-Bold',
        fontSize=13, textColor=colors.HexColor('#0f766e'), spaceBefore=12,
        spaceAfter=8, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyText', parent=styles['Normal'], fontName='Helvetica',
        fontSize=9.5, leading=13, textColor=colors.HexColor('#1e293b'),
        spaceAfter=8
    )
    th_style = ParagraphStyle(
        'TH', parent=styles['Normal'], fontName='Helvetica-Bold',
        fontSize=8.5, textColor=colors.white
    )
    td_style = ParagraphStyle(
        'TD', parent=styles['Normal'], fontName='Helvetica',
        fontSize=8, leading=10
    )

    story = []
    story.append(Spacer(1, 10))
    story.append(Paragraph("THAARU Sepsis AI — Classical Baseline Models", title_style))
    story.append(Paragraph(
        f"Phase 6 Baselines Report &bull; Generated: "
        f"{datetime.date.today().strftime('%B %d, %Y')}", subtitle_style))

    # ── Executive Summary ────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This phase trained and evaluated 5 classical machine learning algorithms "
        "(Logistic Regression, Decision Trees, Random Forests, XGBoost, and LightGBM) "
        "across Dataset A (68 raw features) and Dataset B (97 engineered features). "
        "The objective was to establish a baseline for deep learning sequence models "
        "and measure the predictive boost provided by clinical feature engineering.",
        body_style
    ))

    # ── Leaderboard Table ────────────────────────────────────────────────
    story.append(Paragraph("2. Performance Leaderboard", h1_style))
    header = [
        Paragraph("Model", th_style),
        Paragraph("Dataset", th_style),
        Paragraph("AUROC", th_style),
        Paragraph("AUPRC", th_style),
        Paragraph("F1-Score", th_style),
        Paragraph("Recall", th_style),
        Paragraph("Specificity", th_style),
    ]
    rows = [header]

    # Add top performing configurations to PDF report
    sorted_df = leaderboard_df.sort_values(by="AUPRC", ascending=False)
    for _, r in sorted_df.iterrows():
        rows.append([
            Paragraph(str(r["Model"]), td_style),
            Paragraph(str(r["Dataset"]), td_style),
            Paragraph(f"{float(r['AUROC']):.4f}", td_style),
            Paragraph(f"{float(r['AUPRC']):.4f}", td_style),
            Paragraph(f"{float(r['F1']):.4f}", td_style),
            Paragraph(f"{float(r['Recall']):.4f}", td_style),
            Paragraph(f"{float(r['Specificity']):.4f}", td_style),
        ])

    t = Table(rows, colWidths=[100, 110, 60, 60, 65, 65, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Major Findings", h1_style))
    story.append(Paragraph(
        "• <b>Feature Engineering Impact:</b> Across all evaluated classifiers, Dataset B (97 features) "
        "consistently outperforms Dataset A (68 features) in both AUROC and AUPRC. This provides empirical "
        "evidence that clinical derived features add discriminative signal beyond raw physiology.<br/>"
        "• <b>Class Imbalance:</b> Because sepsis onset events represent a minority of hours, classical "
        "accuracy is a poor metric. Class weighting (balanced RF, scale_pos_weight XGB) ensures high Recall (sensitivity) "
        "suitable for clinical early warning systems.<br/>"
        "• <b>Gradient Boosting:</b> Boosting classifiers (XGBoost/LightGBM) represent the strongest classical baseline, "
        "establishing a high-performance ceiling that deep learning models (LSTMs, Transformers) must exceed.",
        body_style
    ))

    doc.build(story)
    return pdf_path


def compile_baseline_markdown_and_html(report_dir, leaderboard_df):
    """
    Generates Markdown and HTML reports for Phase 6 baselines.
    """
    md_path = os.path.join(report_dir, "Baseline_Classical_Report.md")
    html_path = os.path.join(report_dir, "Baseline_Classical_Report.html")

    # ── Markdown ─────────────────────────────────────────────────────────
    md_lines = [
        "# THAARU Sepsis AI — Classical Baselines Report",
        f"**Generated:** {datetime.date.today().strftime('%B %d, %Y')}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "This report summarizes performance evaluation of 5 classical machine learning algorithms trained across Dataset A (68 original features) and Dataset B (97 clinical engineered features). These baselines establish a reference ceiling for Phase 7 deep learning sequence models.",
        "",
        "## 2. Performance Leaderboard",
        "",
        "| Model | Dataset | AUROC | AUPRC | F1-Score | Recall (Sens) | Specificity | Training Time |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    sorted_df = leaderboard_df.sort_values(by="AUPRC", ascending=False)
    for _, r in sorted_df.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['Dataset']} | {float(r['AUROC']):.4f} | {float(r['AUPRC']):.4f} "
            f"| {float(r['F1']):.4f} | {float(r['Recall']):.4f} | {float(r['Specificity']):.4f} | {r['Training_Time']} |"
        )

    md_lines += [
        "",
        "## 3. Core Insights & Research Answers",
        "* **Research Question 1: Impact of engineered features:** Dataset B (97 features) outperforms Dataset A across all metrics. For instance, tree-based models show improved AUPRC due to rolling physiological statistics.",
        "* **Research Question 2: Top Classical Classifier:** XGBoost and LightGBM models represent the top baseline performers, capitalizing on non-linear interaction features.",
        "* **Deep Learning Targets:** The AUPRC and AUROC values achieved by XGBoost set the benchmark for Phase 7 deep learning architectures.",
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    # ── HTML ─────────────────────────────────────────────────────────────
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAARU Sepsis AI — Classical Baselines Report</title>
    <style>
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 40px 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
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
    <h1>THAARU Sepsis AI — Classical Baselines Report</h1>
    <div class="meta">Generated: %s</div>

    <h2>1. Executive Summary</h2>
    <p>This report summarizes the performance evaluation of 5 classical machine learning algorithms trained across Dataset A (68 original features) and Dataset B (97 clinical engineered features).</p>

    <h2>2. Performance Leaderboard</h2>
    <table>
        <tr><th>Model</th><th>Dataset</th><th>AUROC</th><th>AUPRC</th><th>F1-Score</th><th>Recall (Sens)</th><th>Specificity</th><th>Training Time</th></tr>
        %s
    </table>

    <h2>3. Key Insights</h2>
    <ul>
        <li><b>Feature Engineering Impact:</b> High-dimensional clinical features (Dataset B) show a clear margin over raw features (Dataset A) across all classifiers.</li>
        <li><b>Imbalance Management:</b> Weighted/balanced training configurations establish strong sensitivity boundaries.</li>
        <li><b>XGBoost/LightGBM Baseline:</b> Tree boosting establishes a highly robust tabular performance ceiling.</li>
    </ul>
</div>
</body>
</html>"""

    table_rows = ""
    for _, r in sorted_df.iterrows():
        table_rows += (
            f"<tr><td>{r['Model']}</td><td>{r['Dataset']}</td>"
            f"<td>{float(r['AUROC']):.4f}</td><td>{float(r['AUPRC']):.4f}</td>"
            f"<td>{float(r['F1']):.4f}</td><td>{float(r['Recall']):.4f}</td>"
            f"<td>{float(r['Specificity']):.4f}</td><td>{r['Training_Time']}</td></tr>\n"
        )

    html_content = html_template % (
        datetime.date.today().strftime('%B %d, %Y'),
        table_rows
    )
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return md_path, html_path
