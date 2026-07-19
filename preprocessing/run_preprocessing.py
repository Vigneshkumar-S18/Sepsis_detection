import os
import sys

# Add the project root to python path to support running from any directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Logger file handler setup is performed dynamically inside execute_preprocessing_pipeline

from preprocessing.utils import logger, Timer
from preprocessing.split_data import split_and_prepare_dataset
from preprocessing.validate_ranges import run_range_validation
from preprocessing.impute_missing import run_adaptive_imputation
from preprocessing.handle_outliers import run_outlier_handling
from preprocessing.scale_features import run_feature_scaling
from preprocessing.validate_processed import run_processed_validation
from preprocessing.save_processed import run_save_processed
from preprocessing.generate_report import compile_markdown_and_html_reports, compile_pdf_report

def execute_preprocessing_pipeline():
    """
    Orchestrates the entire preprocessing workflow:
    Runs patient-wise splits, cleans ranges, imputes missing values,
    handles outliers, scales continuous numerical features, validates quality,
    saves the final output Parquets, and compiles Preprocessing reports.
    """
    log_dir = os.path.join(project_root, "datasets", "processed")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "preprocessing_log.txt")
    
    # Setup logging file handler to write logs to preprocessing_log.txt
    import logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    
    logger.info("================================================================")
    logger.info("            SEPSIS DATA PREPROCESSING PIPELINE RUNNER            ")
    logger.info("================================================================")
    
    # 1. Step 3.1: Stratified Split & Indicator columns
    split_and_prepare_dataset()
    
    # Reload split parquets
    train_path = os.path.join(log_dir, "train_split.parquet")
    val_path = os.path.join(log_dir, "val_split.parquet")
    test_path = os.path.join(log_dir, "test_split.parquet")
    
    import pandas as pd
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    # Remove split parquet temp files to keep disk clean (optional but splits are now replaced by processed)
    
    # 2. Step 3.2: Clinical Range Validation
    train_df, val_df, test_df, range_val_summary = run_range_validation(train_df, val_df, test_df)
    
    # 3. Step 3.3: Adaptive Imputation
    train_df, val_df, test_df, imputation_summary = run_adaptive_imputation(train_df, val_df, test_df)
    
    # 4. Step 3.5: Outlier Detection & Winsorization
    train_df, val_df, test_df, outlier_summary = run_outlier_handling(train_df, val_df, test_df)
    
    # 5. Step 3.6: Standard Scaling
    train_df, val_df, test_df, scaler = run_feature_scaling(train_df, val_df, test_df)
    
    # 6. Step 3.7: Quality Assurance Validation
    qa_passed, qa_report = run_processed_validation(train_df, val_df, test_df)
    
    if not qa_passed:
        logger.error("Quality checks FAILED. Pipeline aborted before saving processed files.")
        return
        
    # 7. Step 3.8: Save Final Processed Datasets & Metadata
    stats = run_save_processed(train_df, val_df, test_df)
    
    # 8. Step 3.9: Report Compilation
    report_paths = {
        "summary": os.path.join(project_root, "reports", "summary"),
        "figures": os.path.join(project_root, "reports", "preprocessing"),
        "tables": os.path.join(project_root, "reports", "preprocessing")
    }
    
    range_val_df = pd.DataFrame(range_val_summary)
    
    compile_markdown_and_html_reports(report_paths, stats, range_val_df, qa_report)
    compile_pdf_report(report_paths, stats, range_val_df, outlier_summary, qa_report)
    
    # Clean up temp split files from processed directory to keep it tidy
    for path in [train_path, val_path, test_path]:
        if os.path.exists(path):
            os.remove(path)
            
    logger.info("================================================================")
    logger.info("            PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY      ")
    logger.info("================================================================")
    
    # Remove file handler to release the log file
    logger.removeHandler(file_handler)
    file_handler.close()

if __name__ == "__main__":
    with Timer("Full Preprocessing Pipeline Execution"):
        execute_preprocessing_pipeline()
