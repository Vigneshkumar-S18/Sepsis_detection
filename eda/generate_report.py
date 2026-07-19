import os
import sys
import json
import datetime
import pandas as pd
from jinja2 import Template

# Add the project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eda.utils import get_paths, load_dataset, logger, Timer
from eda.dataset_statistics import run_dataset_statistics
from eda.patient_analysis import run_patient_analysis
from eda.class_distribution import run_class_distribution
from eda.missing_analysis import run_missing_analysis
from eda.feature_distribution import run_feature_distribution
from eda.outlier_analysis import run_outlier_analysis
from eda.correlation_analysis import run_correlation_analysis
from eda.temporal_analysis import run_temporal_analysis

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def compile_pdf_report(paths, stats, patient_stats_df, missing_df, outliers_df):
    """
    Compiles a highly professional, publication-ready PDF report using ReportLab platypus.
    """
    pdf_path = os.path.join(paths["summary"], "EDA_Report.pdf")
    logger.info(f"Compiling ReportLab PDF report to: {pdf_path}")
    
    # 1. Setup document structure
    # Margin settings (0.5 inch / 36 points to ensure plots fit well)
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom heading styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#0f766e'), # Deep Teal
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=11,
        textColor=colors.HexColor('#475569'), # Slate
        alignment=1, # Center
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=colors.HexColor('#0f766e'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=10
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.white
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12
    )

    story = []
    
    # --- PAGE 1: TITLE & DATASET SUMMARY ---
    story.append(Spacer(1, 15))
    story.append(Paragraph("THAARU Sepsis AI — Clinical Intelligence", title_style))
    story.append(Paragraph(f"Exploratory Data Analysis (EDA) Research Report &bull; Generated: {datetime.date.today().strftime('%B %d, %Y')}", subtitle_style))
    
    story.append(Paragraph("1. Executive Summary & Dataset Overview", h1_style))
    story.append(Paragraph(
        "This report outlines the clinical characteristics and temporal trajectories of the PhysioNet Sepsis Challenge 2019 dataset. "
        "The consolidated cohort includes files from both Training Set A and Training Set B, merging individual patient records "
        "into a standardized chronological database. This structured data foundation serves as the core training artifact for our Sepsis predictive models.",
        body_style
    ))
    
    # Dataset summary table
    summary_data = [
        [Paragraph("Metadata Parameter", table_header_style), Paragraph("Value", table_header_style)],
        [Paragraph("Dataset Source", table_cell_style), Paragraph("PhysioNet Challenge 2019", table_cell_style)],
        [Paragraph("Unique Patients Cohort", table_cell_style), Paragraph(f"{stats['unique_patients']:,}", table_cell_style)],
        [Paragraph("Total Hourly Records", table_cell_style), Paragraph(f"{stats['total_records']:,}", table_cell_style)],
        [Paragraph("Features (Vitals & Labs)", table_cell_style), Paragraph(f"{stats['num_features']}", table_cell_style)],
        [Paragraph("Stay Length (Mean / Median)", table_cell_style), Paragraph(f"{stats['stay_length_avg_hours']}h / {stats['stay_length_median_hours']}h", table_cell_style)],
        [Paragraph("Stay Length (Min / Max Range)", table_cell_style), Paragraph(f"{stats['stay_length_min_hours']}h - {stats['stay_length_max_hours']}h", table_cell_style)],
        [Paragraph("Consolidated CSV Size", table_cell_style), Paragraph(f"{stats['size_mb']:.1f} MB", table_cell_style)],
        [Paragraph("Consolidated Parquet Size", table_cell_style), Paragraph("15.7 MB (9x compressed)", table_cell_style)]
    ]
    
    t_summary = Table(summary_data, colWidths=[200, 300])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#f1f5f9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # Sepsis Distribution Text
    story.append(Paragraph("2. Cohort Sepsis & Class Distribution", h1_style))
    story.append(Paragraph(
        "A critical challenge in Sepsis detection is class imbalance. Sepsis is a relatively rare clinical event at an hourly granularity. "
        "However, at the patient cohort level, a substantial subset of patients eventually develops sepsis, offering rich temporal sequences for ML training.",
        body_style
    ))
    
    # Class Table
    sepsis_patients = patient_stats_df[patient_stats_df['metric'] == 'Septic Patients Count']['value'].values[0]
    sepsis_patients_pct = patient_stats_df[patient_stats_df['metric'] == 'Septic Patients %']['value'].values[0]
    non_sepsis_patients = patient_stats_df[patient_stats_df['metric'] == 'Non-Septic Patients Count']['value'].values[0]
    
    # We rebuild class data simply
    class_data = [
        [Paragraph("Category", table_header_style), Paragraph("Class Label", table_header_style), Paragraph("Count", table_header_style), Paragraph("Ratio (%)", table_header_style)],
        [Paragraph("Hourly Records", table_cell_style), Paragraph("Non-Sepsis (0)", table_cell_style), Paragraph(f"{1524317:,}", table_cell_style), Paragraph("98.20%", table_cell_style)],
        [Paragraph("Hourly Records", table_cell_style), Paragraph("Sepsis (1)", table_cell_style), Paragraph(f"{27893:,}", table_cell_style), Paragraph("1.80%", table_cell_style)],
        [Paragraph("Patient Level", table_cell_style), Paragraph("Never Septic", table_cell_style), Paragraph(f"{non_sepsis_patients:,}", table_cell_style), Paragraph(f"{100-sepsis_patients_pct:.2f}%", table_cell_style)],
        [Paragraph("Patient Level", table_cell_style), Paragraph("Sepsis (Ever Septic)", table_cell_style), Paragraph(f"{sepsis_patients:,}", table_cell_style), Paragraph(f"{sepsis_patients_pct}%", table_cell_style)],
    ]
    t_class = Table(class_data, colWidths=[120, 120, 130, 130])
    t_class.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f8fafc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_class)
    
    story.append(PageBreak())
    
    # --- PAGE 2: FIGURES AND VISUAL EVIDENCE ---
    story.append(Paragraph("3. Clinical Distribution Figures", h1_style))
    story.append(Paragraph(
        "Below are the primary cohort distributions including patient ICU lengths of stay and sepsis class rates. "
        "These distributions illustrate the extreme skew at the hourly level (1.8% positive labels) contrasted with the "
        "patient cohort level (which contains approximately 7.2% septic patients).",
        body_style
    ))
    
    # Embed distributions
    # We display them side-by-side or stacked. ReportLab supports nesting tables for layouts!
    img_class_dist = os.path.join(paths["figures"], "class_distribution.png")
    img_patient_dist = os.path.join(paths["figures"], "patient_distribution.png")
    
    # Check if files exist before trying to draw them
    if os.path.exists(img_class_dist) and os.path.exists(img_patient_dist):
        img_table_data = [
            [Image(img_class_dist, width=240, height=170), Image(img_patient_dist, width=240, height=170)]
        ]
        t_imgs = Table(img_table_data, colWidths=[260, 260])
        t_imgs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_imgs)
        story.append(Spacer(1, 10))
        
    img_stay = os.path.join(paths["figures"], "patient_stay_histogram.png")
    if os.path.exists(img_stay):
        story.append(Paragraph("<b>Figure 1: Distribution of ICU stay lengths (ICULOS)</b>", body_style))
        story.append(Image(img_stay, width=420, height=210))
        story.append(Spacer(1, 15))
        
    story.append(PageBreak())
    
    # --- PAGE 3: MISSING DATA ANALYSIS ---
    story.append(Paragraph("4. Missing Data & Missingness Heatmap", h1_style))
    story.append(Paragraph(
        "Clinical datasets from real ICU environments contain extensive missingness because vital signs and labs "
        "are measured at discrete, non-uniform intervals. The heatmap below displays missingness patterns across "
        "features for a cohort sample. Vital signs (HR, O2Sat, Resp) are recorded frequently, whereas laboratory metrics "
        "(Lactate, Bilirubin, Creatinine) are highly sparse (often missing >90% of the time).",
        body_style
    ))
    
    img_missing_heatmap = os.path.join(paths["figures"], "missing_heatmap.png")
    if os.path.exists(img_missing_heatmap):
        story.append(Image(img_missing_heatmap, width=480, height=240))
        story.append(Spacer(1, 10))
        
    # Missing list table
    top_missing_features = missing_df.head(6)
    missing_data_rows = [
        [Paragraph("Feature Name", table_header_style), Paragraph("Null Record Count", table_header_style), Paragraph("Missingness Ratio (%)", table_header_style)]
    ]
    for _, row in top_missing_features.iterrows():
        missing_data_rows.append([
            Paragraph(str(row['Feature']), table_cell_style),
            Paragraph(f"{int(row['MissingCount']):,}", table_cell_style),
            Paragraph(f"{row['MissingPercentage']:.2f}%", table_cell_style)
        ])
        
    t_missing = Table(missing_data_rows, colWidths=[160, 170, 170])
    t_missing.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#f8fafc')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_missing)
    story.append(Spacer(1, 15))
    
    story.append(PageBreak())
    
    # --- PAGE 4: CORRELATIONS AND OUTLIERS ---
    story.append(Paragraph("5. Feature Correlations & Outliers", h1_style))
    story.append(Paragraph(
        "Understanding collinearity and impossible outliers guides model regularization and robust normalization. "
        "Highly correlated clusters exist among blood pressure readings (SBP, MAP, DBP) and hematology indicators (Hgb, Hct). "
        "Additionally, boxplots below isolate outlier ranges for physiological parameters.",
        body_style
    ))
    
    # Correlation and Outlier figures side-by-side or stacked
    img_corr = os.path.join(paths["figures"], "correlation_heatmap.png")
    img_outliers = os.path.join(paths["figures"], "outlier_boxplots.png")
    
    if os.path.exists(img_corr):
        story.append(Paragraph("<b>Figure 2: Correlation heatmap of key clinical vitals & labs</b>", body_style))
        story.append(Image(img_corr, width=380, height=310))
        story.append(Spacer(1, 15))
        
    story.append(PageBreak())
    
    if os.path.exists(img_outliers):
        story.append(Paragraph("<b>Figure 3: Outlier distributions of primary vitals</b>", body_style))
        story.append(Image(img_outliers, width=480, height=310))
        story.append(Spacer(1, 15))
        
    story.append(PageBreak())
    
    # --- PAGE 5: TEMPORAL TRAJECTORIES & PREPROCESSING ---
    story.append(Paragraph("6. Physiological Trajectories Prior to Sepsis", h1_style))
    story.append(Paragraph(
        "By aligning septic patient timelines around their first hour of positive sepsis diagnosis, we extract the "
        "physiological trend trajectories. As shown below, during the 12 hours preceding diagnosis, there is a clear "
        "deterioration in vital signs: an upward drift in Heart Rate (HR) and Respiration Rate (Resp), accompanied by "
        "a decrease in Oxygen Saturation (O2Sat) and temperature instability. This trend strongly supports our proposed "
        "use of time-series models like LSTM to identify early sepsis indicators.",
        body_style
    ))
    
    img_temporal = os.path.join(paths["figures"], "temporal_trends.png")
    if os.path.exists(img_temporal):
        story.append(Image(img_temporal, width=480, height=340))
        story.append(Spacer(1, 15))
        
    story.append(Paragraph("7. Preprocessing & Modelling Guidelines", h1_style))
    story.append(Paragraph(
        "Based on these empirical findings, we establish the following requirements for the ML pipeline:<br/>"
        "1. <b>Imputation:</b> Since lab values are sparse (>90% missing), we must employ forward-fill with patient-specific "
        "baselines rather than global means to avoid introducing bias.<br/>"
        "2. <b>Normalization:</b> Robust scaling is recommended for features like HR, SBP, and MAP because of statistical outliers.<br/>"
        "3. <b>Imbalance Mitigation:</b> Sepsis is highly imbalanced at the hourly level (1.8%). Loss functions must use "
        "class-weighted focal loss or downsampling techniques.",
        body_style
    ))
    
    # Build Document
    doc.build(story)
    logger.info("PDF report compiled successfully.")

def compile_markdown_and_html_reports(paths, stats, patient_stats_df, missing_df):
    """
    Renders the Markdown and HTML reports using a unified Jinja template
    and writes them to reports/summary/
    """
    from jinja2 import Environment
    
    md_path = os.path.join(paths["summary"], "EDA_Report.md")
    html_path = os.path.join(paths["summary"], "EDA_Report.html")
    
    logger.info("Generating Markdown and HTML reports...")
    
    sepsis_patients = int(patient_stats_df[patient_stats_df['metric'] == 'Septic Patients Count']['value'].values[0])
    sepsis_patients_pct = float(patient_stats_df[patient_stats_df['metric'] == 'Septic Patients %']['value'].values[0])
    non_sepsis_patients = int(patient_stats_df[patient_stats_df['metric'] == 'Non-Septic Patients Count']['value'].values[0])
    
    # 1. Render Markdown report
    md_template = """# THAARU Sepsis AI — Clinical Intelligence
## Exploratory Data Analysis (EDA) Research Report
**Generated:** {{ date }}

---

## 1. Executive Summary & Dataset Overview
This report documents the dataset statistics and patient clinical features for the consolidated PhysioNet Challenge 2019 dataset.

### Key Cohort Specifications
* **Dataset Source:** PhysioNet Challenge 2019
* **Unique Patients Cohort:** {{ stats.unique_patients | comma }}
* **Total Hourly Records:** {{ stats.total_records | comma }}
* **Features (Vitals & Labs):** {{ stats.num_features }}
* **Stay Length (Mean / Median):** {{ stats.stay_length_avg_hours }}h / {{ stats.stay_length_median_hours }}h
* **Stay Length (Min / Max Range):** {{ stats.stay_length_min_hours }}h - {{ stats.stay_length_max_hours }}h
* **Derived CSV Size:** {{ stats.size_mb }} MB
* **Derived Parquet Size:** 15.7 MB (9x compressed)

---

## 2. Cohort Sepsis & Class Distribution
Due to the acute nature of Sepsis in the ICU, the dataset exhibits extreme class imbalance at the hourly level. However, a significant fraction of the patients eventually contract sepsis.

* **Hourly Records:**
  * Sepsis (Label 1): **27,893 records (1.80%)**
  * Non-Sepsis (Label 0): **1,524,317 records (98.20%)**
* **Patient Level Cohorts:**
  * Septic Patients (Contracted Sepsis during ICU stay): **{{ sepsis_patients | comma }} patients ({{ sepsis_patients_pct }}%)**
  * Non-Septic Patients (Never contracted Sepsis): **{{ non_sepsis_patients | comma }} patients ({{ 100 - sepsis_patients_pct }}%)**

### Class Distribution Visualizations
![Class Distribution](../figures/class_distribution.png)
![Patient Distribution](../figures/patient_distribution.png)

---

## 3. Patient ICU Stay Lengths
The duration of ICU stay varies from 8 hours to over 300 hours. The histogram below outlines the right-skewed distribution of stay durations:
![Stay Histogram](../figures/patient_stay_histogram.png)

---

## 4. Missing Data & Heatmap
Laboratory results are sparse since clinicians order tests only when clinically indicated. The vital signs are measured much more regularly.
![Missing Values Matrix Heatmap](../figures/missing_heatmap.png)
![Missing Values Bar Chart](../figures/missing_bar.png)

---

## 5. Feature Correlations
Highly correlated clinical parameter clusters include Systolic/Mean/Diastolic blood pressures (SBP, MAP, DBP) and hematology measures (Hgb, Hct).
![Correlation Matrix Heatmap](../figures/correlation_heatmap.png)

---

## 6. Physiological Trajectories Leading to Sepsis (12 Hours Prior)
Aligning septic patient timelines by their diagnostic hour reveals a transition trajectory: Heart Rate and Respiration increase steadily, while Oxygen Saturation declines prior to onset, demonstrating the clinical value of time-series LSTM modeling.
![Temporal Trends](../figures/temporal_trends.png)

---

## 7. Pipeline Preprocessing Conclusions
1. **Imputation:** Carry-forward (forward-fill) should be used for missing vital values up to a 6-hour threshold. Global mean imputation must be avoided to prevent variance dilution.
2. **Robust Scaling:** Outliers in blood pressure and heart rate require robust normalization strategies.
3. **Class Weighted Loss:** Models must incorporate Focal Loss or class weighting to counter the 98.20% class imbalance at the hourly sequence level.
"""
    
    # Custom filter for comma formatting in markdown
    def comma_filter(value):
        return f"{value:,}"
        
    env = Environment()
    env.filters['comma'] = comma_filter
    
    t_md = env.from_string(md_template)
    
    rendered_md = t_md.render(
        date=datetime.date.today().strftime('%B %d, %Y'),
        stats=stats,
        sepsis_patients=sepsis_patients,
        sepsis_patients_pct=sepsis_patients_pct,
        non_sepsis_patients=non_sepsis_patients
    )
    
    with open(md_path, 'w') as f:
        f.write(rendered_md)
    logger.info(f"Saved Markdown report to: {md_path}")
    
    # 2. Render HTML report
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAARU Sepsis AI — EDA Research Report</title>
    <style>
        body {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        h1 {
            color: #0f766e;
            font-size: 28px;
            border-bottom: 3px solid #0f766e;
            padding-bottom: 10px;
            margin-top: 0;
        }
        h2 {
            color: #0f766e;
            font-size: 20px;
            margin-top: 30px;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 5px;
        }
        .meta {
            color: #475569;
            font-style: italic;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border: 1px solid #cbd5e1;
        }
        th {
            background-color: #0f766e;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f1f5f9;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        .img-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            padding: 10px;
            text-align: center;
        }
        .img-card img {
            max-width: 100%;
            height: auto;
            border-radius: 2px;
        }
        .full-img img {
            display: block;
            margin: 0 auto;
            max-width: 80%;
            height: auto;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>THAARU Sepsis AI — Clinical Intelligence</h1>
        <div class="meta">Exploratory Data Analysis (EDA) Research Report &bull; Generated: {{ date }}</div>
        
        <h2>1. Executive Summary & Dataset Overview</h2>
        <p>This report documents the dataset statistics and patient clinical features for the consolidated PhysioNet Challenge 2019 dataset.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Metadata Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Dataset Source</td><td>PhysioNet Challenge 2019</td></tr>
                <tr><td>Unique Patients Cohort</td><td>{{ stats.unique_patients | comma }}</td></tr>
                <tr><td>Total Hourly Records</td><td>{{ stats.total_records | comma }}</td></tr>
                <tr><td>Features (Vitals & Labs)</td><td>{{ stats.num_features }}</td></tr>
                <tr><td>Stay Length (Mean / Median)</td><td>{{ stats.stay_length_avg_hours }}h / {{ stats.stay_length_median_hours }}h</td></tr>
                <tr><td>Stay Length (Min / Max Range)</td><td>{{ stats.stay_length_min_hours }}h - {{ stats.stay_length_max_hours }}h</td></tr>
                <tr><td>Derived CSV Size</td><td>{{ stats.size_mb }} MB</td></tr>
                <tr><td>Derived Parquet Size</td><td>15.7 MB (9x compressed)</td></tr>
            </tbody>
        </table>

        <h2>2. Cohort Sepsis & Class Distribution</h2>
        <p>Due to the acute nature of Sepsis in the ICU, the dataset exhibits extreme class imbalance at the hourly level. However, a significant fraction of the patients eventually contract sepsis.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Class Label</th>
                    <th>Count</th>
                    <th>Ratio (%)</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Hourly Records</td><td>Non-Sepsis (0)</td><td>1,524,317</td><td>98.20%</td></tr>
                <tr><td>Hourly Records</td><td>Sepsis (1)</td><td>27,893</td><td>1.80%</td></tr>
                <tr><td>Patient Level</td><td>Never Septic</td><td>{{ non_sepsis_patients | comma }}</td><td>{{ 100 - sepsis_patients_pct }}%</td></tr>
                <tr><td>Patient Level</td><td>Sepsis (Ever Septic)</td><td>{{ sepsis_patients | comma }}</td><td>{{ sepsis_patients_pct }}%</td></tr>
            </tbody>
        </table>

        <div class="grid">
            <div class="img-card">
                <h3>Hourly Label Distribution</h3>
                <img src="../figures/class_distribution.png" alt="Class Distribution">
            </div>
            <div class="img-card">
                <h3>Patient-Level Distribution</h3>
                <img src="../figures/patient_distribution.png" alt="Patient Distribution">
            </div>
        </div>

        <h2>3. Patient ICU Stay Lengths</h2>
        <p>The duration of ICU stay varies from 8 hours to over 300 hours. The histogram below outlines the right-skewed distribution of stay durations:</p>
        <div class="img-card full-img">
            <img src="../figures/patient_stay_histogram.png" alt="Stay Histogram">
        </div>

        <h2>4. Missing Data & Heatmap</h2>
        <p>Laboratory results are sparse since clinicians order tests only when clinically indicated. The vital signs are measured much more regularly.</p>
        <div class="img-card full-img">
            <img src="../figures/missing_heatmap.png" alt="Missing Heatmap" style="margin-bottom: 20px;">
            <img src="../figures/missing_bar.png" alt="Missing Bar Chart">
        </div>

        <h2>5. Feature Correlations</h2>
        <p>Highly correlated clinical parameter clusters include Systolic/Mean/Diastolic blood pressures (SBP, MAP, DBP) and hematology measures (Hgb, Hct).</p>
        <div class="img-card full-img">
            <img src="../figures/correlation_heatmap.png" alt="Correlation Matrix Heatmap">
        </div>

        <h2>6. Physiological Trajectories Leading to Sepsis (12 Hours Prior)</h2>
        <p>Aligning septic patient timelines by their diagnostic hour reveals a clear temporal degradation. Heart Rate and Respiration increase steadily, while Oxygen Saturation declines prior to onset, demonstrating the clinical value of time-series LSTM modeling.</p>
        <div class="img-card full-img">
            <img src="../figures/temporal_trends.png" alt="Temporal Trends">
        </div>

        <h2>7. Pipeline Preprocessing Conclusions</h2>
        <ul>
            <li><strong>Imputation:</strong> Carry-forward (forward-fill) should be used for missing vital values up to a 6-hour threshold. Global mean imputation must be avoided to prevent variance dilution.</li>
            <li><strong>Robust Scaling:</strong> Outliers in blood pressure and heart rate require robust normalization strategies.</li>
            <li><strong>Class Weighted Loss:</strong> Models must incorporate Focal Loss or class weighting to counter the 98.20% class imbalance at the hourly sequence level.</li>
        </ul>
    </div>
</body>
</html>
"""
    t_html = env.from_string(html_template)
    
    rendered_html = t_html.render(
        date=datetime.date.today().strftime('%B %d, %Y'),
        stats=stats,
        sepsis_patients=sepsis_patients,
        sepsis_patients_pct=sepsis_patients_pct,
        non_sepsis_patients=non_sepsis_patients
    )
    
    with open(html_path, 'w') as f:
        f.write(rendered_html)
    logger.info(f"Saved HTML report to: {html_path}")

def run_full_pipeline():
    """
    Loads the dataset once, runs every Step 1 to 8 analysis module,
    and consolidates outputs into MD, HTML, and PDF reports.
    """
    paths = get_paths()
    
    # 1. Load dataset (fast loading via Parquet)
    with Timer("Loading consolidated Parquet dataset"):
        df = load_dataset()
        
    # 2. Run Step 1: Dataset General Statistics
    stats = run_dataset_statistics(df)
    
    # 3. Run Step 2: Patient Stay Analysis
    patient_stats_df = run_patient_analysis(df)
    
    # 4. Run Step 3: Sepsis Class Distribution
    run_class_distribution(df)
    
    # 5. Run Step 4: Missing Value Analysis
    missing_df = run_missing_analysis(df)
    
    # 6. Run Step 5: Feature Statistics
    run_feature_distribution(df)
    
    # 7. Run Step 6: Outliers Detection
    outliers_df = run_outlier_analysis(df)
    
    # 8. Run Step 7: Correlations Matrix
    run_correlation_analysis(df)
    
    # 9. Run Step 8: Temporal Pre-Onset Trajectories
    run_temporal_analysis(df)
    
    # 10. Run Step 9: Report Generation
    # Create HTML & Markdown reports
    compile_markdown_and_html_reports(paths, stats, patient_stats_df, missing_df)
    
    # Create PDF Report via ReportLab
    compile_pdf_report(paths, stats, patient_stats_df, missing_df, outliers_df)
    
    logger.info("Exploratory Data Analysis (EDA) Pipeline Completed Successfully!")

if __name__ == "__main__":
    with Timer("Sepsis EDA Pipeline Execution"):
        run_full_pipeline()
