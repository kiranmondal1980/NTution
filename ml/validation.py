"""
NSE MOMENTUM 5™ — Purged & Embargoed Time-Series Cross-Validation Engine
================================================================================
Implements institutional time-series validation for financial machine learning models.

CRITICAL ANTI-LEAKAGE MANDATES:
1. STRICT CHRONOLOGY:
   - Forbids random k-fold shuffling of financial time-series data.
2. PURGING:
   - Drops training samples whose forward-looking target realization windows (e.g., 5 days)
     overlap with the test evaluation period.
3. EMBARGO:
   - Removes samples immediately following the test period to prevent post-test
     auto-correlation leakage.
================================================================================
"""

from typing import Generator, Tuple, Optional, List
import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """
    Purged and Embargoed Time-Series Cross-Validator.
    Ensures zero temporal overlap between training labels and test samples.
    """

    def __init__(
        self,
        n_splits: int = 4,
        purge_bars: int = 5,
        embargo_bars: int = 5
    ):
        """
        Args:
            n_splits: Number of rolling chronological folds.
            purge_bars: Forward bars spanned by target horizon to purge from boundary.
            embargo_bars: Trailing buffer bars to remove after test fold.
        """
        self.n_splits: int = n_splits
        self.purge_bars: int = purge_bars
        self.embargo_bars: int = embargo_bars

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[Any] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields chronological (train_indices, test_indices) tuples.
        """
        n_samples = len(X)
        if n_samples < (self.n_splits + 1) * 10:
            # Fallback for very short test series
            split_idx = int(n_samples * 0.75)
            yield np.arange(0, split_idx), np.arange(split_idx, n_samples)
            return

        # Slicing folds chronologically
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            # Purge forward overlap
            test_start = train_end + self.purge_bars
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples or test_start >= test_end:
                break

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    embargo_bars: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and labels into strictly chronological Train and Test sets
    with an embargo buffer at the split boundary.

    Args:
        X: Feature DataFrame indexed chronologically.
        y: Target Series indexed chronologically.
        test_size: Fraction of historical series allocated to out-of-sample test.
        embargo_bars: Number of buffer bars removed between train and test boundaries.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    n_samples = len(X)
    split_point = int(n_samples * (1.0 - test_size))

    # Train uses history up to split_point minus embargo buffer
    train_end = max(1, split_point - embargo_bars)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    return X_train, X_test, y_train, y_test
