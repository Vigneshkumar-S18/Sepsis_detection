# ─────────────────────────────────────────────────────────────────────────────
# Run Sequence Generation — Main orchestrator for Phase 5
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import gc
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from preprocessing.utils import logger, Timer
from sequence_generation.config import (
    EXPERIMENT_CONFIGS, EXCLUDE_COLUMNS, LABEL_COLUMN,
    PATIENT_ID_COLUMN, STRIDE
)
from sequence_generation.sequence_builder import build_split_sequences
from sequence_generation.label_generator import compute_label_statistics
from sequence_generation.validation import validate_sequences
from sequence_generation.save_sequences import save_sequence_dataset
from sequence_generation.generate_report import (
    compile_sequence_pdf, compile_sequence_markdown_and_html
)


def run_sequence_generation_pipeline():
    """
    Main pipeline:
    1. Loads feature-engineered parquet splits.
    2. Iterates through each experiment config (window × horizon).
    3. Builds sequences per split, validates, saves .npz + metadata.
    4. Compiles summary reports.
    """
    processed_dir = os.path.join(project_root, "datasets", "processed")
    sequences_dir = os.path.join(project_root, "datasets", "sequences")
    report_dir = os.path.join(project_root, "reports", "summary")
    os.makedirs(sequences_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # ── 1. Load parquet splits ───────────────────────────────────────────
    with Timer("Loading feature-engineered parquet splits"):
        splits = {}
        for split_name, fname in [("train", "train_features.parquet"),
                                   ("validation", "validation_features.parquet"),
                                   ("test", "test_features.parquet")]:
            path = os.path.join(processed_dir, fname)
            if not os.path.exists(path):
                logger.error(f"Missing: {path}")
                sys.exit(1)
            splits[split_name] = pd.read_parquet(path)
            logger.info(f"  Loaded {split_name}: {splits[split_name].shape}")

    # ── 2. Determine feature columns ─────────────────────────────────────
    feature_cols = [
        col for col in splits["train"].columns
        if col not in EXCLUDE_COLUMNS
    ]
    n_features = len(feature_cols)
    logger.info(f"Feature columns for sequences: {n_features} features")

    # ── 3. Iterate over experiment configurations ────────────────────────
    all_stats = {}

    for cfg_id, cfg in EXPERIMENT_CONFIGS.items():
        window_size = cfg["window_size"]
        horizon = cfg["horizon"]
        description = cfg["description"]

        logger.info(f"\n{'='*70}")
        logger.info(f"Experiment: {cfg_id} | Window={window_size}h | "
                     f"Horizon=+{horizon}h | {description}")
        logger.info(f"{'='*70}")

        cfg_output_dir = os.path.join(sequences_dir, cfg_id)
        os.makedirs(cfg_output_dir, exist_ok=True)

        cfg_stats = {
            "window_size": window_size,
            "horizon": horizon,
            "description": description,
            "n_features": n_features,
        }

        for split_name, split_df in splits.items():
            with Timer(f"Building {split_name} sequences for {cfg_id}"):
                X, y, patient_ids, metadata_df = build_split_sequences(
                    split_df, feature_cols, window_size, horizon, STRIDE,
                    split_name, logger=logger
                )

            # Label statistics
            label_stats = compute_label_statistics(y)
            cfg_stats[split_name] = label_stats
            logger.info(f"  {split_name} label stats: "
                         f"total={label_stats['total_sequences']:,}, "
                         f"positive={label_stats['positive_count']:,} "
                         f"({label_stats['positive_rate']*100:.2f}%), "
                         f"imbalance={label_stats['imbalance_ratio']:.1f}:1")

            # Validation (run on train split only to save time)
            if split_name == "train":
                with Timer(f"Validating {split_name} sequences for {cfg_id}"):
                    val_results = validate_sequences(
                        X, y, patient_ids, metadata_df,
                        window_size, n_features,
                        source_df=split_df, feature_cols=feature_cols,
                        horizon=horizon, logger=logger
                    )
                cfg_stats["validation_results"] = val_results
            else:
                # Quick shape + NaN check for val/test
                shape_ok = X.shape == (len(y), window_size, n_features)
                nan_ok = int(np.isnan(X).sum()) == 0
                logger.info(f"  {split_name} quick check: shape={'PASS' if shape_ok else 'FAIL'}, "
                             f"nan={'PASS' if nan_ok else 'FAIL'}")

            # Save
            with Timer(f"Saving {split_name} sequences for {cfg_id}"):
                save_sequence_dataset(
                    X, y, patient_ids, metadata_df,
                    cfg_output_dir, split_name, logger=logger
                )

            # Free memory after saving each split
            del X, y, patient_ids, metadata_df
            gc.collect()

        all_stats[cfg_id] = cfg_stats

    # ── 4. Save aggregate statistics JSON ────────────────────────────────
    stats_path = os.path.join(sequences_dir, "sequence_statistics.json")

    # Convert any non-serializable types
    def _make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(i) for i in obj]
        return obj

    with open(stats_path, 'w') as f:
        json.dump(_make_serializable(all_stats), f, indent=2)
    logger.info(f"\nSaved aggregate statistics to: {stats_path}")

    # ── 5. Compile reports ───────────────────────────────────────────────
    with Timer("Compiling sequence generation reports"):
        pdf_path = compile_sequence_pdf(report_dir, all_stats)
        md_path, html_path = compile_sequence_markdown_and_html(report_dir, all_stats)
        logger.info(f"  PDF:  {pdf_path}")
        logger.info(f"  MD:   {md_path}")
        logger.info(f"  HTML: {html_path}")

    logger.info("\n" + "=" * 70)
    logger.info("Phase 5 — Temporal Sequence Generation COMPLETE")
    logger.info(f"  Configs: {len(all_stats)}")
    total = sum(
        s["train"]["total_sequences"] + s["validation"]["total_sequences"] +
        s["test"]["total_sequences"]
        for s in all_stats.values()
    )
    logger.info(f"  Total sequences generated: {total:,}")
    logger.info("=" * 70)


if __name__ == "__main__":
    with Timer("Phase 5 — Sequence Generation Pipeline"):
        run_sequence_generation_pipeline()
