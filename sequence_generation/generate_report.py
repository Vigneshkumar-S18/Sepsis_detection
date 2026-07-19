# ─────────────────────────────────────────────────────────────────────────────
# Report Generator — Creates PDF, HTML, and Markdown sequence reports
# ─────────────────────────────────────────────────────────────────────────────
import os
import datetime
from jinja2 import Environment

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def compile_sequence_pdf(report_dir, all_stats):
    """
    Generates a styled PDF report summarizing all sequence experiment datasets.
    """
    pdf_path = os.path.join(report_dir, "Sequence_Report.pdf")

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
    story.append(Paragraph("THAARU Sepsis AI — Temporal Sequence Report", title_style))
    story.append(Paragraph(
        f"Sequence Generation Summary &bull; Generated: "
        f"{datetime.date.today().strftime('%B %d, %Y')}", subtitle_style))

    # ── Executive Summary ────────────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", h1_style))
    total_seqs = sum(
        s["train"]["total_sequences"] + s["validation"]["total_sequences"] +
        s["test"]["total_sequences"]
        for s in all_stats.values()
    )
    story.append(Paragraph(
        f"This phase generated <b>{total_seqs:,}</b> temporal sequences across "
        f"<b>{len(all_stats)}</b> experiment configurations. Each sequence is a "
        f"fixed-length window of standardized clinical features, paired with a "
        f"binary sepsis label aligned to the specified prediction horizon.",
        body_style
    ))

    # ── Experiment Dataset Summary Table ─────────────────────────────────
    story.append(Paragraph("2. Experiment Dataset Summary", h1_style))
    header = [
        Paragraph("Config", th_style),
        Paragraph("Window", th_style),
        Paragraph("Horizon", th_style),
        Paragraph("Train Seqs", th_style),
        Paragraph("Val Seqs", th_style),
        Paragraph("Test Seqs", th_style),
        Paragraph("Train +Rate", th_style),
    ]
    rows = [header]
    for cfg_id, stats in all_stats.items():
        rows.append([
            Paragraph(cfg_id, td_style),
            Paragraph(f"{stats['window_size']}h", td_style),
            Paragraph(f"+{stats['horizon']}h", td_style),
            Paragraph(f"{stats['train']['total_sequences']:,}", td_style),
            Paragraph(f"{stats['validation']['total_sequences']:,}", td_style),
            Paragraph(f"{stats['test']['total_sequences']:,}", td_style),
            Paragraph(f"{stats['train']['positive_rate']*100:.2f}%", td_style),
        ])

    t = Table(rows, colWidths=[80, 55, 55, 80, 70, 70, 75])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#f8fafc')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Validation Results ───────────────────────────────────────────────
    story.append(Paragraph("3. Validation Results", h1_style))
    for cfg_id, stats in all_stats.items():
        val = stats.get("validation_results", {})
        overall = val.get("overall", {}).get("passed", "N/A")
        status = "✅ ALL PASSED" if overall else "❌ SOME FAILED"
        story.append(Paragraph(
            f"<b>{cfg_id}</b>: {status}", body_style
        ))

    story.append(PageBreak())

    # ── Per-Config Detail Pages ──────────────────────────────────────────
    story.append(Paragraph("4. Per-Configuration Details", h1_style))
    for cfg_id, stats in all_stats.items():
        story.append(Paragraph(f"{cfg_id} — {stats.get('description', '')}", h1_style))
        detail_rows = [
            [Paragraph("Split", th_style), Paragraph("Sequences", th_style),
             Paragraph("Positive", th_style), Paragraph("Negative", th_style),
             Paragraph("+Rate", th_style), Paragraph("Imbalance", th_style)],
        ]
        for split in ["train", "validation", "test"]:
            s = stats[split]
            detail_rows.append([
                Paragraph(split.capitalize(), td_style),
                Paragraph(f"{s['total_sequences']:,}", td_style),
                Paragraph(f"{s['positive_count']:,}", td_style),
                Paragraph(f"{s['negative_count']:,}", td_style),
                Paragraph(f"{s['positive_rate']*100:.2f}%", td_style),
                Paragraph(f"{s['imbalance_ratio']:.1f}:1", td_style),
            ])
        dt = Table(detail_rows, colWidths=[90, 80, 75, 75, 65, 70])
        dt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#f8fafc')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(dt)
        story.append(Spacer(1, 15))

    doc.build(story)
    return pdf_path


def compile_sequence_markdown_and_html(report_dir, all_stats):
    """
    Generates Markdown and HTML reports for sequence generation.
    """
    md_path = os.path.join(report_dir, "Sequence_Report.md")
    html_path = os.path.join(report_dir, "Sequence_Report.html")

    # ── Markdown ─────────────────────────────────────────────────────────
    md_lines = [
        "# THAARU Sepsis AI — Temporal Sequence Generation Report",
        f"**Generated:** {datetime.date.today().strftime('%B %d, %Y')}",
        "",
        "---",
        "",
        "## 1. Experiment Dataset Summary",
        "",
        "| Config | Window | Horizon | Train Seqs | Val Seqs | Test Seqs | Train +Rate |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for cfg_id, stats in all_stats.items():
        md_lines.append(
            f"| {cfg_id} | {stats['window_size']}h | +{stats['horizon']}h "
            f"| {stats['train']['total_sequences']:,} "
            f"| {stats['validation']['total_sequences']:,} "
            f"| {stats['test']['total_sequences']:,} "
            f"| {stats['train']['positive_rate']*100:.2f}% |"
        )

    md_lines += ["", "---", "", "## 2. Per-Configuration Details", ""]
    for cfg_id, stats in all_stats.items():
        md_lines.append(f"### {cfg_id} — {stats.get('description', '')}")
        md_lines.append("")
        md_lines.append("| Split | Sequences | Positive | Negative | +Rate | Imbalance |")
        md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for split in ["train", "validation", "test"]:
            s = stats[split]
            md_lines.append(
                f"| {split.capitalize()} | {s['total_sequences']:,} "
                f"| {s['positive_count']:,} | {s['negative_count']:,} "
                f"| {s['positive_rate']*100:.2f}% | {s['imbalance_ratio']:.1f}:1 |"
            )
        md_lines.append("")

    md_lines += ["---", "", "## 3. Validation", ""]
    for cfg_id, stats in all_stats.items():
        val = stats.get("validation_results", {})
        overall = val.get("overall", {}).get("passed", "N/A")
        status = "✅ ALL PASSED" if overall else "❌ SOME FAILED"
        md_lines.append(f"- **{cfg_id}**: {status}")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    # ── HTML ─────────────────────────────────────────────────────────────
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAARU Sepsis AI — Temporal Sequence Report</title>
    <style>
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; padding: 40px 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h1 { color: #0f766e; font-size: 28px; border-bottom: 3px solid #0f766e; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #0f766e; font-size: 20px; margin-top: 30px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; }
        h3 { color: #334155; font-size: 16px; margin-top: 20px; }
        .meta { color: #475569; font-style: italic; margin-bottom: 30px; }
        table { width: 100%%; border-collapse: collapse; margin: 20px 0; }
        th, td { text-align: left; padding: 12px; border: 1px solid #cbd5e1; }
        th { background: #0f766e; color: white; }
        tr:nth-child(even) { background: #f1f5f9; }
        .pass { color: #16a34a; font-weight: bold; }
        .fail { color: #dc2626; font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <h1>THAARU Sepsis AI — Temporal Sequence Report</h1>
    <div class="meta">Generated: %s</div>

    <h2>1. Experiment Dataset Summary</h2>
    <table>
        <tr><th>Config</th><th>Window</th><th>Horizon</th><th>Train</th><th>Val</th><th>Test</th><th>Train +Rate</th></tr>
        %s
    </table>

    <h2>2. Per-Configuration Details</h2>
    %s

    <h2>3. Validation Results</h2>
    <ul>%s</ul>
</div>
</body>
</html>"""

    summary_rows = ""
    for cfg_id, stats in all_stats.items():
        summary_rows += (
            f"<tr><td>{cfg_id}</td><td>{stats['window_size']}h</td>"
            f"<td>+{stats['horizon']}h</td>"
            f"<td>{stats['train']['total_sequences']:,}</td>"
            f"<td>{stats['validation']['total_sequences']:,}</td>"
            f"<td>{stats['test']['total_sequences']:,}</td>"
            f"<td>{stats['train']['positive_rate']*100:.2f}%</td></tr>\n"
        )

    detail_html = ""
    for cfg_id, stats in all_stats.items():
        detail_html += f"<h3>{cfg_id} — {stats.get('description', '')}</h3>\n<table>\n"
        detail_html += "<tr><th>Split</th><th>Sequences</th><th>Positive</th><th>Negative</th><th>+Rate</th><th>Imbalance</th></tr>\n"
        for split in ["train", "validation", "test"]:
            s = stats[split]
            detail_html += (
                f"<tr><td>{split.capitalize()}</td>"
                f"<td>{s['total_sequences']:,}</td>"
                f"<td>{s['positive_count']:,}</td>"
                f"<td>{s['negative_count']:,}</td>"
                f"<td>{s['positive_rate']*100:.2f}%</td>"
                f"<td>{s['imbalance_ratio']:.1f}:1</td></tr>\n"
            )
        detail_html += "</table>\n"

    val_html = ""
    for cfg_id, stats in all_stats.items():
        val = stats.get("validation_results", {})
        overall = val.get("overall", {}).get("passed", False)
        cls = "pass" if overall else "fail"
        txt = "ALL PASSED ✅" if overall else "SOME FAILED ❌"
        val_html += f'<li><b>{cfg_id}</b>: <span class="{cls}">{txt}</span></li>\n'

    html_content = html_template % (
        datetime.date.today().strftime('%B %d, %Y'),
        summary_rows, detail_html, val_html
    )
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return md_path, html_path
