"""
NSE MOMENTUM 5™ — CLI Tool: Machine Learning Model Training Pipeline
================================================================================
Command-line utility to train, validate, and serialize probabilistic classifiers
for short-term research upside targets (+5%/3D, +10%/5D, +15%/5D, +20%/5D).

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --model gradient_boosting --target all
    python scripts/train_models.py --model random_forest --target target_15pct_5d
================================================================================
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.initialize import init_database
from database.db import get_db_session
from database.models import ModelRunRecord
from data.downloader import MarketDataDownloader
from features import build_feature_pipeline
from ml.targets import create_research_targets
from ml.train import ModelTrainer
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("CLI_ML_TRAINER")


def main() -> int:
    parser = argparse.ArgumentParser(description="NSE MOMENTUM 5™ Machine Learning Trainer")
    parser.add_argument(
        "--model",
        type=str,
        default="gradient_boosting",
        choices=["gradient_boosting", "random_forest", "logistic_regression"],
        help="Machine learning algorithm to train (default: gradient_boosting)."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        help="Target horizon to train: 'all', 'target_5pct_3d', 'target_10pct_5d', 'target_15pct_5d', 'target_20pct_5d'."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=CONFIG.ml.test_size_chronological,
        help="Chronological out-of-sample test fraction (default: 0.20)."
    )
    args = parser.parse_args()

    # 1. Guarantee database is initialized
    init_database()

    # 2. Load universe and compute features and forward targets
    logger.info("Loading market data from database to construct pooled training dataset...")
    downloader = MarketDataDownloader()
    bench_df = downloader.load_ohlcv_from_db(CONFIG.universe.benchmark_symbol)

    symbols = CONFIG.universe.default_symbols
    training_frames = []

    for sym in symbols:
        df = downloader.load_ohlcv_from_db(sym)
        if df.empty or len(df) < 60:
            continue

        # Extract features and append forward-looking targets
        f_df = build_feature_pipeline(df, benchmark_df=bench_df)
        t_df = create_research_targets(df)
        combined = pd.concat([f_df, t_df], axis=1)
        training_frames.append(combined)

    if not training_frames:
        logger.error("No historical data available for training. Run `python scripts/download_data.py` first.")
        return 1

    pooled_df = pd.concat(training_frames, axis=0).sort_index()
    logger.info(f"Pooled training matrix assembled: {len(pooled_df):,} total observation bars.")

    # 3. Train target models
    trainer = ModelTrainer(model_type=args.model)
    targets_to_train = CONFIG.ml.all_targets if args.target.lower() == "all" else [args.target]

    print("\n" + "=" * 70)
    print(f"       NSE MOMENTUM 5™ — ML TRAINING AUDIT ({args.model.upper()})")
    print("=" * 70)

    for tgt in targets_to_train:
        logger.info(f"Training {args.model} on {tgt}...")
        report = trainer.train_target_model(pooled_df, target_col=tgt, test_size=args.test_size)

        if report.get("trained", False):
            print(f"  Target Horizon       : {tgt}")
            print(f"  Out-of-Sample ROC-AUC: {report['roc_auc']:.3f}")
            print(f"  Brier Calibration    : {report['brier_score']:.4f}")
            print(f"  Test Accuracy        : {report.get('accuracy', 0)*100:.1f}%")
            print(f"  Training Samples     : {report['train_samples']:,}")
            print(f"  Test Samples (OOS)   : {report['test_samples']:,}")
            print(f"  Top Predictive Features: {list(report.get('top_features', {}).keys())[:4]}")
            print("-" * 70)

            # Persist Model Run in Database
            try:
                with get_db_session() as session:
                    model_rec = ModelRunRecord(
                        model_name=args.model.upper(),
                        target_name=tgt,
                        train_start=datetime.utcnow(),
                        train_end=datetime.utcnow(),
                        test_start=datetime.utcnow(),
                        test_end=datetime.utcnow(),
                        roc_auc=report["roc_auc"],
                        brier_score=report["brier_score"],
                        feature_importance_json=json.dumps(report.get("top_features", {}))
                    )
                    session.add(model_rec)
                    session.commit()
            except Exception as exc:
                logger.warning(f"Could not persist model run record to database: {exc}")
        else:
            logger.warning(f"Training skipped for {tgt}: {report.get('notes', 'Insufficient samples')}")

    print("=" * 70 + "\n")
    logger.info("Machine learning model training and artifact serialization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
