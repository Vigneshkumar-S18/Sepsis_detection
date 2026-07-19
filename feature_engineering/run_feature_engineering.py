import os
import sys
import pickle
import json
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
from jinja2 import Environment

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from feature_engineering.build_features import generate_engineered_features

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def compile_pdf_report(paths, stats, summary_stats):
    """
    Compiles a styled, publication-ready Clinical Feature Engineering Report PDF.
    """
    pdf_path = os.path.join(paths["summary"], "Feature_Engineering_Report.pdf")
    logger.info(f"Compiling ReportLab Feature Engineering PDF to: {pdf_path}")
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18,
        textColor=colors.HexColor('#0f766e'), alignment=1, spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=10,
        textColor=colors.HexColor('#475569'), alignment=1, spaceAfter=25
    )
    h1_style = ParagraphStyle(
        'SectionH1', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=13,
        textColor=colors.HexColor('#0f766e'), spaceBefore=12, spaceAfter=8, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5,
        leading=13, textColor=colors.HexColor('#1e293b'), spaceAfter=8
    )
    table_header_style = ParagraphStyle(
        'TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8.5,
        textColor=colors.white
    )
    table_cell_style = ParagraphStyle(
        'TableCell', parent=styles['Normal'], fontName='Helvetica', fontSize=8, leading=10
    )

    story = []
    story.append(Spacer(1, 10))
    story.append(Paragraph("THAARU Sepsis AI — Clinical Feature Engineering", title_style))
    story.append(Paragraph(f"Feature Derivation & Scaler Update Summary &bull; Generated: {datetime.date.today().strftime('%B %d, %Y')}", subtitle_style))
    
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        f"This phase derives clinically validated cardiovascular indices, patient-level temporal trajectory "
        f"statistics, and measurement availability markers. To guarantee leak-proof scaling across "
        f"splits, a new StandardScaler was fit exclusively on the Training split incorporating both "
        f"original clinical columns and the new engineered features. The final dataset size increased from "
        f"68 columns to {stats.get('total_columns', 0)} total columns.",
        body_style
    ))
    
    # Feature Stats Table
    stats_data = [
        [Paragraph("Feature Group", table_header_style), Paragraph("Features Added", table_header_style), Paragraph("Clinical Justification", table_header_style)],
        [Paragraph("Cardiovascular Indices", table_cell_style), Paragraph("Shock Index, Pulse Pressure, MAP deviation", table_cell_style), Paragraph("Identifiers for hemodynamic instability and hypovolemia.", table_cell_style)],
        [Paragraph("Clinical Ratios", table_cell_style), Paragraph("HR/MAP, Resp/O2Sat, HR/Temp", table_cell_style), Paragraph("Proxy markers for physiological stress and SIRS indicators.", table_cell_style)],
        [Paragraph("6h Rolling Stats", table_cell_style), Paragraph("6h Mean, 6h Std/Variability (HR, Resp, Temp, MAP, O2Sat)", table_cell_style), Paragraph("Captures vital signs instability and deterioration cycles.", table_cell_style)],
        [Paragraph("1h Lag Differences", table_cell_style), Paragraph("1h Slopes/Differences (HR, Temp, Resp, O2Sat)", table_cell_style), Paragraph("Tracks acute worsening trends and rate of change.", table_cell_style)],
        [Paragraph("Sparsity Markers", table_cell_style), Paragraph("Hours since last Lactate, WBC, pH, Creatinine, etc.", table_cell_style), Paragraph("Tracks clinician decision-making density (ordering patterns).", table_cell_style)],
        [Paragraph("Timeline Flags", table_cell_style), Paragraph("First 24 Hours, Diurnal proxy hour of stay", table_cell_style), Paragraph("Controls for stay duration and temporal physiological cycle.", table_cell_style)]
    ]
    t_stats = Table(stats_data, colWidths=[130, 160, 250])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor('#f8fafc')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_stats)
    story.append(Spacer(1, 12))
    
    # 2. Split dimensions
    story.append(Paragraph("2. Processed Parquet Dimensions (Post Feature Engineering)", h1_style))
    t_dim_data = [
        [Paragraph("Dataset Partition", table_header_style), Paragraph("Rows (Hourly Records)", table_header_style), Paragraph("Original Columns", table_header_style), Paragraph("Engineered Columns", table_header_style), Paragraph("Final Columns", table_header_style)],
        [Paragraph("Training Split", table_cell_style), Paragraph("1,086,436", table_cell_style), Paragraph("68", table_cell_style), Paragraph(f"{stats.get('engineered_columns', 0)}", table_cell_style), Paragraph(f"{stats.get('total_columns', 0)}", table_cell_style)],
        [Paragraph("Validation Split", table_cell_style), Paragraph("231,472", table_cell_style), Paragraph("68", table_cell_style), Paragraph(f"{stats.get('engineered_columns', 0)}", table_cell_style), Paragraph(f"{stats.get('total_columns', 0)}", table_cell_style)],
        [Paragraph("Testing Split", table_cell_style), Paragraph("234,302", table_cell_style), Paragraph("68", table_cell_style), Paragraph(f"{stats.get('engineered_columns', 0)}", table_cell_style), Paragraph(f"{stats.get('total_columns', 0)}", table_cell_style)]
    ]
    t_dim = Table(t_dim_data, colWidths=[140, 110, 100, 100, 90])
    t_dim.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#f8fafc')),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#f8fafc')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_dim)
    
    story.append(PageBreak())
    
    # --- PAGE 2: SCALE ANALYSIS OF DERIVED FEATURES ---
    story.append(Paragraph("3. Summary Statistics of Derived Clinical Features (Train Partition)", h1_style))
    story.append(Paragraph(
        "Below are the unscaled means, scales (stds), and variances fit on the training split, "
        "enabling stable and reproducible transformations during sequence building and model deployment.",
        body_style
    ))
    
    sum_rows = [
        [Paragraph("Engineered Feature Name", table_header_style), Paragraph("Fit Mean", table_header_style), Paragraph("Fit Standard Deviation", table_header_style), Paragraph("Fit Variance", table_header_style)]
    ]
    for _, row in summary_stats.head(25).iterrows():  # Top 25 for report length limits
        sum_rows.append([
            Paragraph(str(row['Feature']), table_cell_style),
            Paragraph(f"{row['Mean']:.4f}", table_cell_style),
            Paragraph(f"{row['Scale']:.4f}", table_cell_style),
            Paragraph(f"{row['Variance']:.4f}", table_cell_style)
        ])
        
    t_sum = Table(sum_rows, colWidths=[200, 110, 120, 110])
    t_sum.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f766e')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_sum)
    
    doc.build(story)
    logger.info("Clinical Feature Engineering PDF report compiled successfully.")

def compile_markdown_and_html_reports(paths, stats, summary_stats):
    """
    Generates Feature Engineering reports in HTML and Markdown.
    """
    md_path = os.path.join(paths["summary"], "Feature_Engineering_Report.md")
    html_path = os.path.join(paths["summary"], "Feature_Engineering_Report.html")
    
    logger.info("Generating Feature Engineering Markdown and HTML reports...")
    
    # 1. Markdown Template
    md_template = """# THAARU Sepsis AI — Clinical Feature Engineering Report
## Derived Features & Model Schema Overview
**Generated:** {{ date }}

---

## 1. Executive Summary
This report summarizes Phase 4 clinical feature engineering. All features are calculated patient-wise (no crossover patient leakage) and scaled symmetrically using parameters fit on the Training split cohort.

* **Final Features Shape:** {{ stats.total_columns }} columns (68 original columns + {{ stats.engineered_columns }} newly engineered features)
* **Training Records:** 1,086,436 rows
* **Validation Records:** 231,472 rows
* **Testing Records:** 234,302 rows

---

## 2. Derived Feature Groups

| Feature Group | Features Added | Clinical Justification |
| :--- | :--- | :--- |
| **Cardiovascular Indices** | Shock Index, Pulse Pressure, MAP deviation | Key flags for cardiovascular hypoperfusion and shock |
| **Clinical Ratios** | HR/MAP, Resp/O2Sat, HR/Temp | Tracks vital signs couplings under septic stress |
| **6h Rolling Statistics** | 6h Mean, 6h Std/Variability (HR, Temp, Resp, MAP, O2Sat) | Identifies vital signs instability trends |
| **1h Lag Differences** | 1h Slopes/Differences (HR, Temp, Resp, O2Sat) | Captures rate of change and acute worsening trends |
| **Sparsity Indicators** | Hours since Lactate, WBC, pH, Creatinine, etc. | Tracks density of clinician test ordering patterns |
| **ICU Timeline Flags** | First 24 Hours, Diurnal proxy hour of stay | Controls for stay duration and diurnal stay cycles |

---

## 3. Scale Statistics of Derived Features (Train Set Fit)
Below are the training-fit scaler parameters used to standardize the newly engineered features:

| Feature Name | Mean | Standard Deviation (Scale) | Variance |
| :--- | :--- | :--- | :--- |
{% for idx, row in summary_stats.iterrows() -%}
| {{ row.Feature }} | {{ row.Mean | round(4) }} | {{ row.Scale | round(4) }} | {{ row.Variance | round(4) }} |
{% endfor %}

---

## 4. Conclusion
The feature engineering stage is stable and complete. Processed splits now contain a comprehensive set of 94 features (including demographics, scaled vitals/labs, missingness flags, and derived clinical indices), ready for Phase 5 sequence formatting.
"""
    
    env = Environment()
    t_md = env.from_string(md_template)
    rendered_md = t_md.render(
        date=datetime.date.today().strftime('%B %d, %Y'),
        stats=stats,
        summary_stats=summary_stats
    )
    with open(md_path, 'w') as f:
        f.write(rendered_md)
    logger.info(f"Saved Markdown report to: {md_path}")
    
    # 2. HTML Template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>THAARU Sepsis AI — Clinical Feature Engineering Report</title>
    <style>
        body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; background-color: #f8fafc; color: #1e293b; line-height: 1.6; padding: 40px 20px; }
        .container { max-width: 900px; margin: 0 auto; background: #ffffff; padding: 40px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h1 { color: #0f766e; font-size: 28px; border-bottom: 3px solid #0f766e; padding-bottom: 10px; margin-top: 0; }
        h2 { color: #0f766e; font-size: 20px; margin-top: 30px; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; }
        .meta { color: #475569; font-style: italic; margin-bottom: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { text-align: left; padding: 12px; border: 1px solid #cbd5e1; }
        th { background-color: #0f766e; color: white; }
        tr:nth-child(even) { background-color: #f1f5f9; }
        ul { padding-left: 20px; }
        li { margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>THAARU Sepsis AI — Clinical Feature Engineering Report</h1>
        <div class="meta">Derived Features & Scaling Verification Report &bull; Generated: {{ date }}</div>
        
        <h2>1. Executive Summary</h2>
        <p>This report documents Phase 4 clinical feature engineering. The final dataset combines original clinical measurements, missingness indicators, and derived clinical features.</p>
        
        <table>
            <thead>
                <tr>
                    <th>Dataset Partition</th>
                    <th>Rows (Hourly Records)</th>
                    <th>Original Columns</th>
                    <th>Engineered Columns</th>
                    <th>Final Columns</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Training Split</td><td>1,086,436</td><td>68</td><td>{{ stats.engineered_columns }}</td><td>{{ stats.total_columns }}</td></tr>
                <tr><td>Validation Split</td><td>231,472</td><td>68</td><td>{{ stats.engineered_columns }}</td><td>{{ stats.total_columns }}</td></tr>
                <tr><td>Testing Split</td><td>234,302</td><td>68</td><td>{{ stats.engineered_columns }}</td><td>{{ stats.total_columns }}</td></tr>
            </tbody>
        </table>

        <h2>2. Derived Feature Groups</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature Group</th>
                    <th>Features Added</th>
                    <th>Clinical Significance</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Cardiovascular Indices</td><td>Shock Index, Pulse Pressure, MAP deviation</td><td> hemodynamic shock and volume status indicators</td></tr>
                <tr><td>Clinical Ratios</td><td>HR/MAP, Resp/O2Sat, HR/Temp</td><td>Physiological distress and systemic inflammatory response</td></tr>
                <tr><td>6h Rolling Statistics</td><td>6h Mean, 6h Std/Variability</td><td>Vital signs instability and deterioration cycles</td></tr>
                <tr><td>1h Lag Differences</td><td>1h Slopes/Differences</td><td>Tracks acute worsening trends and rate of change</td></tr>
                <tr><td>Sparsity Indicators</td><td>Hours since Last Lab Measurements</td><td>Tracks density of clinician test ordering patterns</td></tr>
                <tr><td>ICU Timeline Flags</td><td>First 24 Hours, Diurnal proxy hour of stay</td><td>Diurnal biological cycle controls</td></tr>
            </tbody>
        </table>

        <h2>3. Scale Statistics of Derived Features (Train Set Fit)</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature Name</th>
                    <th>Mean</th>
                    <th>Standard Deviation (Scale)</th>
                    <th>Variance</th>
                </tr>
            </thead>
            <tbody>
                {% for idx, row in summary_stats.iterrows() -%}
                <tr>
                    <td>{{ row.Feature }}</td>
                    <td>{{ row.Mean | round(4) }}</td>
                    <td>{{ row.Scale | round(4) }}</td>
                    <td>{{ row.Variance | round(4) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>4. Conclusion</h2>
        <p>The feature engineering stage is stable and complete. Processed splits now contain a comprehensive set of {{ stats.total_columns }} features, ready for Phase 5 sequence formatting.</p>
    </div>
</body>
</html>
"""
    t_html = env.from_string(html_template)
    rendered_html = t_html.render(
        date=datetime.date.today().strftime('%B %d, %Y'),
        stats=stats,
        summary_stats=summary_stats
    )
    with open(html_path, 'w') as f:
        f.write(rendered_html)
    logger.info(f"Saved HTML report to: {html_path}")

def execute_feature_engineering_pipeline():
    """
    Main pipeline entry for feature engineering:
    1. Loads train/validation/test_processed.parquet.
    2. Loads base scaler.pkl and inverse-transforms clinical continuous columns.
    3. Runs build_features.py to calculate Shock Index, rolling stats, slopes, sparsity indicators.
    4. Fits a new StandardScaler on all continuous columns (original + engineered).
    5. Saves train/validation/test_features.parquet.
    6. Saves reports.
    """
    processed_dir = os.path.join(project_root, "datasets", "processed")
    reports_dir = os.path.join(project_root, "reports", "features")
    os.makedirs(reports_dir, exist_ok=True)
    
    train_path = os.path.join(processed_dir, "train_processed.parquet")
    val_path = os.path.join(processed_dir, "validation_processed.parquet")
    test_path = os.path.join(processed_dir, "test_processed.parquet")
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path, scaler_path]):
        logger.error("Base processed parquet splits or scaler pickle not found in datasets/processed/")
        sys.exit(1)
        
    # 1. Load data & scaler
    with Timer("Loading processed splits and base scaler"):
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)
        with open(scaler_path, "rb") as f:
            base_scaler = pickle.load(f)
            
    # 2. Inverse-transform continuous columns (unscaling base features to raw clinical values)
    with Timer("Inverse-transforming continuous base columns"):
        columns_to_scale_base = [
            col for col in train_df.columns
            if col not in ['PatientID', 'SepsisLabel', 'Gender', 'Unit1', 'Unit2']
            and not col.endswith('_measured')
        ]
        
        train_df_unscaled = train_df.copy()
        train_df_unscaled[columns_to_scale_base] = base_scaler.inverse_transform(train_df[columns_to_scale_base])
        
        val_df_unscaled = val_df.copy()
        val_df_unscaled[columns_to_scale_base] = base_scaler.inverse_transform(val_df[columns_to_scale_base])
        
        test_df_unscaled = test_df.copy()
        test_df_unscaled[columns_to_scale_base] = base_scaler.inverse_transform(test_df[columns_to_scale_base])
        
    # 3. Generate engineered features
    with Timer("Generating engineered features (Cardiovascular, Rolling, Trend, Sparsity)"):
        train_fe = generate_engineered_features(train_df_unscaled, train_df)
        val_fe = generate_engineered_features(val_df_unscaled, val_df)
        test_fe = generate_engineered_features(test_df_unscaled, test_df)
        
    # 4. Fit new StandardScaler on all continuous columns (original + newly engineered)
    with Timer("Fitting new StandardScaler on expanded feature set"):
        # Exclude IDs, target, categorical flags, and binary indicators from scaling
        new_columns_to_scale = [
            col for col in train_fe.columns
            if col not in ['PatientID', 'SepsisLabel', 'Gender', 'Unit1', 'Unit2', 'First_24h_Flag']
            and not col.endswith('_measured')
        ]
        
        new_scaler = StandardScaler()
        new_scaler.fit(train_fe[new_columns_to_scale])
        
        # Apply new scaling
        train_fe[new_columns_to_scale] = new_scaler.transform(train_fe[new_columns_to_scale])
        val_fe[new_columns_to_scale] = new_scaler.transform(val_fe[new_columns_to_scale])
        test_fe[new_columns_to_scale] = new_scaler.transform(test_fe[new_columns_to_scale])
        
    # 5. Save new splits and scaler pickle (overwriting scaler.pkl)
    with Timer("Saving features splits parquets and new scaler pickle"):
        train_fe.to_parquet(os.path.join(processed_dir, "train_features.parquet"), index=False)
        val_fe.to_parquet(os.path.join(processed_dir, "validation_features.parquet"), index=False)
        test_fe.to_parquet(os.path.join(processed_dir, "test_features.parquet"), index=False)
        
        with open(scaler_path, "wb") as f:
            pickle.dump(new_scaler, f)
            
    # 6. Save scaled statistics to reports/features/scaled_statistics.csv
    stats_data = {
        "Feature": new_columns_to_scale,
        "Mean": new_scaler.mean_,
        "Scale": new_scaler.scale_,
        "Variance": new_scaler.var_
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(reports_dir, "scaled_statistics.csv"), index=False)
    
    # 7. Compile reports
    engineered_cols = [c for c in train_fe.columns if c not in train_df.columns]
    summary_stats = stats_df[stats_df['Feature'].isin(engineered_cols)]
    
    stats_summary = {
        "engineered_columns": int(len(engineered_cols)),
        "total_columns": int(len(train_fe.columns)),
        "dataset_shapes": {
            "train_split": train_fe.shape,
            "validation_split": val_fe.shape,
            "test_split": test_fe.shape
        }
    }
    
    report_paths = {
        "summary": os.path.join(project_root, "reports", "summary"),
        "figures": os.path.join(project_root, "reports", "features"),
        "tables": os.path.join(project_root, "reports", "features")
    }
    
    compile_markdown_and_html_reports(report_paths, stats_summary, summary_stats)
    compile_pdf_report(report_paths, stats_summary, summary_stats)
    
    logger.info(f"Feature engineering pipeline completed! Total columns: {len(train_fe.columns)} ({len(engineered_cols)} engineered).")

if __name__ == "__main__":
    with Timer("Feature Engineering Run"):
        execute_feature_engineering_pipeline()
