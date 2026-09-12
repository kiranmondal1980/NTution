"""
NSE MOMENTUM 5™ — Machine Learning & Research Probability Suite
================================================================================
Exposes all machine learning target builders, feature curators, purged time-series
cross-validation splitters, model training pipelines, and probability estimators.

Modules Exported:
- targets: create_research_targets (+5%/3D, +10%/5D, +15%/5D, +20%/5D targets)
- features: extract_ml_features, prepare_training_dataset, ML_FEATURE_NAMES
- validation: PurgedTimeSeriesSplit, chronological_train_test_split (Zero leakage)
- train: ModelTrainer (Gradient Boosting, Random Forest, Logistic Regression)
- predict: TargetProbabilityEstimator (ML inference + Empirical baseline)
================================================================================
"""

from .targets import create_research_targets
from .features import (
    extract_ml_features,
    prepare_training_dataset,
    ML_FEATURE_NAMES
)
from .validation import (
    PurgedTimeSeriesSplit,
    chronological_train_test_split
)
from .train import ModelTrainer
from .predict import TargetProbabilityEstimator

__all__ = [
    "create_research_targets",
    "extract_ml_features",
    "prepare_training_dataset",
    "ML_FEATURE_NAMES",
    "PurgedTimeSeriesSplit",
    "chronological_train_test_split",
    "ModelTrainer",
    "TargetProbabilityEstimator"
]
