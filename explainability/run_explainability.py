# Orchestrator for Phase 8 Explainable AI Pipeline
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from explainability.shap_analysis import run_shap_explanations
from explainability.integrated_gradients import run_integrated_gradients
from explainability.attention_visualization import run_attention_visualization
from explainability.error_analysis import run_error_analysis
from explainability.feature_ranking import compare_feature_rankings
from explainability.patient_case_studies import generate_clinician_dashboards
from explainability.generate_report import compile_explainability_reports


def run_explainability_pipeline():
    logger.info("="*75)
    logger.info("Initializing THAARU Sepsis AI — Phase 8 Explainable AI (XAI) Framework")
    logger.info("="*75)
    
    with Timer("Phase 8 XAI Framework Pipeline"):
        # Step 1: SHAP Explanations on XGBoost
        with Timer("Step 1: SHAP Explanations"):
            run_shap_explanations()

        # Step 2: Integrated Gradients on BiLSTM
        with Timer("Step 2: Integrated Gradients Attributions"):
            run_integrated_gradients()

        # Step 3: Transformer Attention Heatmaps
        with Timer("Step 3: Attention Visualization"):
            run_attention_visualization()

        # Step 4: Clinical Error Analysis
        with Timer("Step 4: Error Cohort Segmentation"):
            res = run_error_analysis()
            tps, tns, fps, fns, probs, labels, pids = (
                res[1], res[2], res[3], res[4], res[5], res[6], res[7]
            )

        # Step 5: Feature Ranking Comparisons
        with Timer("Step 5: Feature Rankings Comparison"):
            compare_feature_rankings()

        # Step 6: Patient Case Studies & Clinician Dashboard
        with Timer("Step 6: Clinician Dashboard & Case Studies"):
            generate_clinician_dashboards(tps, tns, fps, fns, probs, labels, pids)

        # Step 7: Compile Reports
        with Timer("Step 7: Compiling Summary PDF, HTML, and MD Reports"):
            compile_explainability_reports()

    logger.info("="*75)
    logger.info("Phase 8 Explainable AI Framework Pipeline Execution Complete!")
    logger.info("="*75)


if __name__ == "__main__":
    run_explainability_pipeline()
