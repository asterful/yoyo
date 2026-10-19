import json
import numpy as np
import pandas as pd
from purgedcv import PurgedKFold as _PurgedKFold
from yoyo.models import BaseModel


def export_fold_metadata_json(
    fold: int,
    total_len: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    index_values: pd.Index,
    all_folds_data: list,
    filepath: str
) -> None:
    """Computes contiguous block segments (train, test, gap/purge/embargo) 
    and appends them to a JSON export structure.
    """
    status = np.zeros(total_len, dtype=int)
    status[train_idx] = 1
    status[test_idx] = 2

    segments = []
    if total_len > 0:
        current_type_code = status[0]
        type_map = {0: "gap", 1: "train", 2: "test"}
        current_type = type_map[current_type_code]
        start_idx = 0

        for i in range(1, total_len):
            t_code = status[i]
            t = type_map[t_code]
            if t != current_type:
                seg = {
                    "type": current_type,
                    "start_idx": int(start_idx),
                    "end_idx": int(i - 1),
                    "start_time": str(index_values[start_idx]),
                    "end_time": str(index_values[i - 1]),
                }
                segments.append(seg)
                current_type = t
                start_idx = i

        # Final segment
        segments.append({
            "type": current_type,
            "start_idx": int(start_idx),
            "end_idx": int(total_len - 1),
            "start_time": str(index_values[start_idx]),
            "end_time": str(index_values[total_len - 1]),
        })

    fold_info = {
        "fold": fold,
        "total_rows": total_len,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "gap_rows": total_len - len(train_idx) - len(test_idx),
        "segments": segments
    }
    
    all_folds_data.append(fold_info)

    # Write out to JSON file
    with open(filepath, "w") as f:
        json.dump({"total_rows": total_len, "folds": all_folds_data}, f, indent=2)


class PurgedKFold:
    """Contiguous folds that automatically purge overlapping labels from a `t1` column 
    and apply a percentage-based embargo.
    """

    def __init__(self, t1_col: str = "t1", n_splits: int = 5, embargo: int = 0):
        self.t1_col = t1_col
        self.n_splits = n_splits
        self.embargo = embargo


    def split(self, data: pd.DataFrame):
        if self.t1_col not in data.columns:
            raise ValueError(f"Column '{self.t1_col}' not found in data.")

        # prediction_times (t0): when each observation/feature row starts
        pred_times = pd.Series(data.index, index=data.index)
        
        # evaluation_times (t1): when each observation's label resolves
        eval_times = data[self.t1_col]

        # Initialize the underlying purgedcv splitter
        cv = _PurgedKFold(
            n_splits=self.n_splits,
            prediction_times=pred_times,
            evaluation_times=eval_times,
            embargo_observations=self.embargo if self.embargo > 0 else None
        )

        for train_idx, test_idx in cv.split(data):
            yield train_idx, test_idx


class PurgedKFoldCrossValidator:
    """Fit a model and return predictions for each purged fold, supporting 
    variable `t1` label intervals and sample weights.
    """

    def __init__(self, model: BaseModel, data: pd.DataFrame, t1_col: str = "t1", n_splits: int = 5, embargo: int = 0, export_json_path: str | None = "purged_cv_metadata.json"):
        self.model = model
        self.data = data
        self.t1_col = t1_col
        self.export_json_path = export_json_path
        self.splitter = PurgedKFold(t1_col=t1_col, n_splits=n_splits, embargo=embargo)


    def _split_features_target(self, df: pd.DataFrame, target_col: str, sample_weight_col: str | None = None):
        columns_to_drop = [target_col, self.t1_col]
        if sample_weight_col is not None:
            columns_to_drop.append(sample_weight_col)
        
        # Keep only columns that actually exist in the dataframe slice
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        X = df.drop(columns=columns_to_drop)
        y = df[target_col]
        return X, y


    def run(self, target_col: str = "target", sample_weight_col: str | None = None) -> list[dict]:
        results = []
        all_folds_metadata = []
        total_len = len(self.data)
        index_values = self.data.index

        for fold, (train_idx, test_idx) in enumerate(self.splitter.split(self.data), 1):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            # Save fold breakdown to JSON if path provided
            if self.export_json_path:
                export_fold_metadata_json(
                    fold, total_len, train_idx, test_idx, index_values, 
                    all_folds_metadata, self.export_json_path
                )

            X_train, y_train = self._split_features_target(
                train_data, target_col, sample_weight_col
            )
            X_test, y_test = self._split_features_target(
                test_data, target_col, sample_weight_col
            )
            
            sample_weight = (
                train_data[sample_weight_col] if sample_weight_col is not None else None
            )

            # Fit model using sample weights if provided
            self.model.fit(X_train, y_train, sample_weight=sample_weight)

            predictions = pd.Series(self.model.predict(X_test), index=y_test.index)

            results.append({
                "fold": fold,
                "y_true": y_test,
                "y_pred": predictions
            })

        return results