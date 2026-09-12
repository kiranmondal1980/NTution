"""
NSE MOMENTUM 5™ — Machine Learning Training & Model Governance Engine
================================================================================
Trains calibrated probabilistic classifiers to estimate upside realization probabilities:
- Target 1: +5% within 3 trading sessions (`target_5pct_3d`)
- Target 2: +10% within 5 trading sessions (`target_10pct_5d`)
- Target 3: +15% within 5 trading sessions (`target_15pct_5d`)
- Target 4: +20% within 5 trading sessions (`target_20pct_5d`)

REPRODUCIBILITY & ARCHITECTURE:
- Models: Logistic Regression, Random Forest, Gradient Boosting
- Fixed Random Seed: `random_state = 42`
- Scaler: StandardScaler fit strictly on in-sample training data
- Validation: Chronological out-of-sample split with purged boundary
- Metrics: ROC-AUC, Brier Score (probability calibration), Precision, Log-Loss
- Artifact Persistence: Models, scalers, and feature metadata saved to `MODEL_DIR`
================================================================================
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, log_loss,
    accuracy_score, precision_score
)
from sklearn.preprocessing import StandardScaler

from ml.features import prepare_training_dataset
from ml.validation import chronological_train_test_split
from config import CONFIG, MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger("ML_TRAIN_ENGINE")


class ModelTrainer:
    """Trains, validates, and serializes probabilistic target classifiers."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = CONFIG.ml.random_state
    ):
        """
        Args:
            model_type: "gradient_boosting", "random_forest", or "logistic_regression".
            random_state: Deterministic seed for reproducibility.
        """
        self.model_type = model_type.lower()
        self.random_state = random_state

    def _instantiate_model(self) -> Any:
        """Instantiates the specified classification algorithm."""
        if self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced"
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=10,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=-1
            )
        else:
            # Default: Gradient Boosting Classifier
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=3,
                min_samples_leaf=15,
                random_state=self.random_state
            )

    def train_target_model(
        self,
        df_combined: pd.DataFrame,
        target_col: str = "target_15pct_5d",
        test_size: float = CONFIG.ml.test_size_chronological
    ) -> Dict[str, Any]:
        """
        Trains and validates a calibrated classifier for an upside target horizon.

        Args:
            df_combined: DataFrame containing features and research targets.
            target_col: Binary target column to train against.
            test_size: Out-of-sample chronological test ratio.

        Returns:
            Dictionary with training metadata, metrics, and feature importances.
        """
        X, y, feature_names = prepare_training_dataset(df_combined, target_col=target_col)

        if len(X) < 100 or y.nunique() < 2:
            logger.warning(f"Insufficient training samples or single class in {target_col}.")
            return {
                "trained": False,
                "target": target_col,
                "model_type": self.model_type,
                "roc_auc": 0.50,
                "brier_score": 1.0,
                "notes": "Insufficient samples or single class distribution."
            }

        # 1. Chronological Split (No Shuffling)
        X_train, X_test, y_train, y_test = chronological_train_test_split(
            X, y, test_size=test_size, embargo_bars=CONFIG.ml.purge_embargo_bars
        )

        if len(y_train) < 30 or len(y_test) < 10 or y_test.nunique() < 2:
            logger.warning("Split produced insufficient class variance in test slice.")
            return {
                "trained": False,
                "target": target_col,
                "model_type": self.model_type,
                "roc_auc": 0.50,
                "brier_score": 1.0,
                "notes": "Class imbalance in chronological test partition."
            }

        # 2. In-Sample Feature Normalization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 3. Model Training
        model = self._instantiate_model()
        model.fit(X_train_scaled, y_train)

        # 4. Out-of-Sample Evaluation
        probs_test = model.predict_proba(X_test_scaled)[:, 1]
        preds_test = (probs_test >= 0.5).astype(int)

        auc = float(roc_auc_score(y_test, probs_test))
        brier = float(brier_score_loss(y_test, probs_test))
        acc = float(accuracy_score(y_test, preds_test))
        prec = float(precision_score(y_test, preds_test, zero_division=0))

        # 5. Extract Feature Importances
        feature_importance: Dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance = {
                name: round(float(imp), 4)
                for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
            }
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0]
            feature_importance = {
                name: round(float(abs(c)), 4)
                for name, c in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:10]
            }

        # 6. Save Artifacts to Disk
        model_filename = MODEL_DIR / f"{target_col}_{self.model_type}.joblib"
        scaler_filename = MODEL_DIR / f"{target_col}_{self.model_type}_scaler.joblib"
        meta_filename = MODEL_DIR / f"{target_col}_{self.model_type}_meta.json"

        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)

        metadata = {
            "target": target_col,
            "model_type": self.model_type,
            "trained_at": datetime.utcnow().isoformat(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_start": str(X_train.index[0].date()),
            "train_end": str(X_train.index[-1].date()),
            "test_start": str(X_test.index[0].date()),
            "test_end": str(X_test.index[-1].date()),
            "roc_auc": round(auc, 3),
            "brier_score": round(brier, 4),
            "accuracy": round(acc, 3),
            "features_used": feature_names,
            "top_features": feature_importance
        }

        with open(meta_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Trained {self.model_type} on {target_col}: OOS AUC = {auc:.3f}, Brier = {brier:.4f}"
        )

        return {
            "trained": True,
            "target": target_col,
            "model_type": self.model_type,
            "roc_auc": round(auc, 3),
            "brier_score": round(brier, 4),
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "top_features": feature_importance
        }

    def train_all_targets(
        self,
        df_combined: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Trains models across all configured research upside horizons.

        Returns:
            Dictionary of {target_name: training_report}.
        """
        reports: Dict[str, Dict[str, Any]] = {}
        targets = CONFIG.ml.all_targets

        for tgt in targets:
            if tgt in df_combined.columns:
                reports[tgt] = self.train_target_model(df_combined, target_col=tgt)

        return reports
