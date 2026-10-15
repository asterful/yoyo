import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

class WalkForwardCrossValidator:
    def __init__(self, model: BaseModel, data: pd.DataFrame, n_splits: int = 5, gap: int = 0):
        self.model = model
        self.data = data
        self.splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)


    def _split_features_target(self, df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y


    def run(self, target_col: str = "target") -> list[dict]:
        results = []

        for fold, (train_idx, test_idx) in enumerate(self.splitter.split(self.data)):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            X_train, y_train = self._split_features_target(train_data, target_col)
            X_test, y_test = self._split_features_target(test_data, target_col)

            self.model.fit(X_train, y_train)
            
            predictions = pd.Series(self.model.predict(X_test), index=y_test.index)

            results.append({
                "fold": fold + 1,
                "y_true": y_test,
                "y_pred": predictions
            })

        return results