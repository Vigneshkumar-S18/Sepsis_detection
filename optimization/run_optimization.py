# Orchestrator for Phase 9 Optimization Pipeline
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from optimization.xgboost_opt import run_xgboost_optimization
from optimization.bilstm_opt import run_bilstm_optimization
from optimization.generate_report import compile_optimization_reports


def run_optimization_pipeline():
    logger.info("="*75)
    logger.info("Initializing THAARU Sepsis AI — Phase 9 Optimization Framework")
    logger.info("="*75)
    
    with Timer("Phase 9 Hyperparameter Optimization Pipeline"):
        # Step 1: XGBoost Hyperparameter Sweep
        with Timer("Step 1: XGBoost Hyperparameter Sweep"):
            run_xgboost_optimization(num_trials=5)
            
        # Step 2: BiLSTM Hyperparameter Sweep
        with Timer("Step 2: BiLSTM Hyperparameter Sweep"):
            run_bilstm_optimization(num_trials=5)
            
        # Step 3: Compile Reports
        with Timer("Step 3: Compiling Optimization Reports"):
            compile_optimization_reports()

    logger.info("="*75)
    logger.info("Phase 9 Hyperparameter Optimization Complete!")
    logger.info("="*75)


if __name__ == "__main__":
    run_optimization_pipeline()
