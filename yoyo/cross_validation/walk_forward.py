import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from yoyo.models import BaseModel


class WalkForwardCrossValidator:
    def __init__(self, model: BaseModel, data: pd.DataFrame, n_splits: int = 5, gap: int = 0):
        self.model = model
        self.data = data
        self.splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    def _split_features_target(
        self, df: pd.DataFrame, target_col: str, sample_weight_col: str | None = None
    ):
        columns_to_drop = [target_col]
        if sample_weight_col is not None:
            columns_to_drop.append(sample_weight_col)
        X = df.drop(columns=columns_to_drop)
        y = df[target_col]
        return X, y

    def run(
        self, target_col: str = "target", sample_weight_col: str | None = None
    ) -> list[dict]:
        results = []

        for fold, (train_idx, test_idx) in enumerate(self.splitter.split(self.data)):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            X_train, y_train = self._split_features_target(
                train_data, target_col, sample_weight_col
            )
            X_test, y_test = self._split_features_target(
                test_data, target_col, sample_weight_col
            )
            sample_weight = (
                train_data[sample_weight_col] if sample_weight_col is not None else None
            )

            self.model.fit(X_train, y_train, sample_weight=sample_weight)

            predictions = pd.Series(self.model.predict(X_test), index=y_test.index)

            results.append({
                "fold": fold + 1,
                "y_true": y_test,
                "y_pred": predictions
            })

        return results
