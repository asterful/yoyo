import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from yoyo.models import BaseModel


class PurgedKFold:
    """Contiguous folds that purge overlapping labels and embargo feature rows.

    ``label_horizon`` is the number of future bars used to make each label.
    ``embargo`` is an extra gap after the test labels, in bars; set it to the
    largest feature lookback to keep future training features away from test data.
    """

    def __init__(self, label_horizon: int, n_splits: int = 5, embargo: int = 0):
        self.label_horizon = label_horizon
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, data: pd.DataFrame):
        indices = np.arange(len(data))

        for _, test_indices in KFold(n_splits=self.n_splits, shuffle=False).split(data):
            test_start = test_indices[0]
            test_end = test_indices[-1]
            purge_start = max(0, test_start - self.label_horizon)
            purge_stop = min(
                len(data), test_end + self.label_horizon + self.embargo + 1
            )
            train_indices = np.concatenate((indices[:purge_start], indices[purge_stop:]))
            yield train_indices, test_indices


class PurgedKFoldCrossValidator:
    """Fit a model and return predictions for each purged fold."""

    def __init__(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        label_horizon: int,
        n_splits: int = 5,
        embargo: int = 0,
    ):
        self.model = model
        self.data = data
        self.splitter = PurgedKFold(label_horizon, n_splits, embargo)

    @staticmethod
    def _split_features_target(data: pd.DataFrame, target_col: str):
        return data.drop(columns=[target_col]), data[target_col]

    def run(self, target_col: str = "target") -> list[dict]:
        results = []
        for fold, (train_idx, test_idx) in enumerate(self.splitter.split(self.data), 1):
            X_train, y_train = self._split_features_target(
                self.data.iloc[train_idx], target_col
            )
            X_test, y_test = self._split_features_target(
                self.data.iloc[test_idx], target_col
            )

            self.model.fit(X_train, y_train)
            y_pred = pd.Series(self.model.predict(X_test), index=y_test.index)
            results.append({"fold": fold, "y_true": y_test, "y_pred": y_pred})

        return results
