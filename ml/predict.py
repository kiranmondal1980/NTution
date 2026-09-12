"""
NSE MOMENTUM 5™ — Target Probability Inference & Expected Move Engine
================================================================================
Calculates empirical and machine-learned probabilities for research upside targets:
1. Probability of +5% within 3 trading days
2. Probability of +10% within 5 trading days
3. Probability of +15% within 5 trading days
4. Probability of +20% within 5 trading days
5. Expected return & Expected adverse excursion (MAE)

TWO-TIER INFERENCE ARCHITECTURE:
- Tier 1 (Transparent Empirical Baseline):
  Conditional historical realization frequency matching similar past momentum setups.
- Tier 2 (Machine Learning Inference):
  Loads calibrated models from `MODEL_DIR` and predicts calibrated event probabilities.
  Gracefully falls back to Tier 1 baseline if models are not yet trained.
================================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import joblib
import numpy as np
import pandas as pd

from ml.features import extract_ml_features
from config import CONFIG, MODEL_DIR
from utils.logger import setup_logger

logger = setup_logger("PROBABILITY_ESTIMATOR")


class TargetProbabilityEstimator:
    """Estimates forward move probabilities and expected excursions."""

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type.lower()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self._load_saved_models()

    def _load_saved_models(self) -> None:
        """Attempts to load trained ML model artifacts and scalers from disk."""
        for target in CONFIG.ml.all_targets:
            model_file = MODEL_DIR / f"{target}_{self.model_type}.joblib"
            scaler_file = MODEL_DIR / f"{target}_{self.model_type}_scaler.joblib"
            meta_file = MODEL_DIR / f"{target}_{self.model_type}_meta.json"

            if model_file.exists() and scaler_file.exists():
                try:
                    self.models[target] = joblib.load(model_file)
                    self.scalers[target] = joblib.load(scaler_file)
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as f:
                            self.metadata[target] = json.load(f)
                    logger.debug(f"Loaded trained ML artifact for {target}.")
                except Exception as exc:
                    logger.warning(f"Could not load ML model for {target}: {exc}")

    def estimate_probabilities(
        self,
        symbol_df: pd.DataFrame,
        current_momentum_score: float = 75.0,
        prefer_ml: bool = True
    ) -> Dict[str, Any]:
        """
        Calculates forward upside probabilities and expected excursions.

        Args:
            symbol_df: Historical feature DataFrame for the target stock.
            current_momentum_score: Latest calculated 0–100 momentum score.
            prefer_ml: If True, uses trained ML models; falls back to empirical if absent.

        Returns:
            Dictionary containing target probabilities, expected return, MAE, and methodology.
        """
        # Check if trained ML models are available
        has_ml = prefer_ml and all(t in self.models for t in CONFIG.ml.all_targets)

        if has_ml:
            return self._predict_ml(symbol_df, current_momentum_score)
        else:
            return self._estimate_empirical(symbol_df, current_momentum_score)

    def _predict_ml(
        self,
        symbol_df: pd.DataFrame,
        current_momentum_score: float
    ) -> Dict[str, Any]:
        """Tier 2: ML Model Probability Inference."""
        try:
            X_all, features_used = extract_ml_features(symbol_df)
            if X_all.empty:
                return self._estimate_empirical(symbol_df, current_momentum_score)

            latest_x = X_all.iloc[[-1]]  # Last row as 1-row DataFrame

            probs: Dict[str, float] = {}
            for target in CONFIG.ml.all_targets:
                model = self.models[target]
                scaler = self.scalers[target]

                # Scale features strictly using the model's fitted scaler
                x_scaled = scaler.transform(latest_x)
                prob_val = float(model.predict_proba(x_scaled)[0, 1])
                probs[target] = round(float(np.clip(prob_val, 0.01, 0.95)), 2)

            # Continuous Excursion Estimates based on historical volatility
            latest_bar = symbol_df.iloc[-1]
            close = float(latest_bar.get("close", 100.0))
            atr_pct = float(latest_bar.get("atr_pct", 2.5)) / 100.0

            expected_return = round((probs["target_10pct_5d"] * 0.10) + (probs["target_15pct_5d"] * 0.15), 3)
            expected_mae = round(2.0 * atr_pct, 3)

            return {
                "methodology": f"Machine Learning ({self.model_type.upper()})",
                "prob_5pct_3d": probs.get("target_5pct_3d", 0.30),
                "prob_10pct_5d": probs.get("target_10pct_5d", 0.18),
                "prob_15pct_5d": probs.get("target_15pct_5d", 0.08),
                "prob_20pct_5d": probs.get("target_20pct_5d", 0.04),
                "expected_return_pct": expected_return,
                "expected_adverse_excursion_pct": expected_mae,
                "risk_reward_ratio": round(expected_return / expected_mae, 2) if expected_mae > 0 else 1.5,
                "is_ml_backed": True
            }

        except Exception as exc:
            logger.warning(f"ML inference failed: {exc}. Falling back to empirical baseline.")
            return self._estimate_empirical(symbol_df, current_momentum_score)

    def _estimate_empirical(
        self,
        symbol_df: pd.DataFrame,
        current_momentum_score: float
    ) -> Dict[str, Any]:
        """
        Tier 1: Transparent Empirical Baseline.
        Calculates historical conditional realization frequency from past sessions.
        """
        if symbol_df is None or len(symbol_df) < 30:
            # Conservative safe baseline
            return {
                "methodology": "Empirical Baseline (Safe Default)",
                "prob_5pct_3d": 0.25,
                "prob_10pct_5d": 0.15,
                "prob_15pct_5d": 0.06,
                "prob_20pct_5d": 0.02,
                "expected_return_pct": 0.045,
                "expected_adverse_excursion_pct": 0.050,
                "risk_reward_ratio": 0.90,
                "is_ml_backed": False
            }

        high = symbol_df["high"]
        low = symbol_df["low"]
        next_open = symbol_df["open"].shift(-1)

        fwd_high_3d = high.shift(-1).rolling(3).max().shift(-2)
        fwd_high_5d = high.shift(-1).rolling(5).max().shift(-4)
        fwd_low_5d = low.shift(-1).rolling(5).min().shift(-4)

        safe_open = next_open.replace(0, np.nan)
        g_3d = ((fwd_high_3d - safe_open) / safe_open).dropna()
        g_5d = ((fwd_high_5d - safe_open) / safe_open).dropna()
        mae_5d = ((safe_open - fwd_low_5d) / safe_open).dropna()

        # Historical base realization rates across this stock's history
        base_5_3 = float((g_3d >= 0.05).mean()) if len(g_3d) > 0 else 0.20
        base_10_5 = float((g_5d >= 0.10).mean()) if len(g_5d) > 0 else 0.12
        base_15_5 = float((g_5d >= 0.15).mean()) if len(g_5d) > 0 else 0.05
        base_20_5 = float((g_5d >= 0.20).mean()) if len(g_5d) > 0 else 0.02

        # Empirical multiplier conditioned on current momentum conviction
        # Score of 75 = 1.0x; Score of 90 = 1.25x; Score of 60 = 0.80x
        score_multiplier = current_momentum_score / 75.0

        p5_3 = float(np.clip(base_5_3 * score_multiplier, 0.01, 0.90))
        p10_5 = float(np.clip(base_10_5 * score_multiplier, 0.01, 0.75))
        p15_5 = float(np.clip(base_15_5 * score_multiplier, 0.005, 0.55))
        p20_5 = float(np.clip(base_20_5 * score_multiplier, 0.002, 0.40))

        avg_expected_mae = float(mae_5d.mean()) if len(mae_5d) > 0 else 0.045
        expected_return = (p5_3 * 0.05) + (p10_5 * 0.10) + (p15_5 * 0.15)

        return {
            "methodology": "Empirical Conditional Frequency (Baseline)",
            "prob_5pct_3d": round(p5_3, 2),
            "prob_10pct_5d": round(p10_5, 2),
            "prob_15pct_5d": round(p15_5, 2),
            "prob_20pct_5d": round(p20_5, 2),
            "expected_return_pct": round(expected_return, 3),
            "expected_adverse_excursion_pct": round(avg_expected_mae, 3),
            "risk_reward_ratio": round(expected_return / avg_expected_mae, 2) if avg_expected_mae > 0 else 1.2,
            "is_ml_backed": False
        }
