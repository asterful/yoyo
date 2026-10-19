import pandas as pd
from yoyo.features import BaseFeature


class ClassBalancedWeightFeature(BaseFeature):
    def __init__(self, name: str, label_col: str):
        super().__init__(name)
        self.label_col = label_col


    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.label_col not in df.columns:
            raise KeyError(
                f"ClassBalancedWeightFeature missing required column: '{self.label_col}'"
            )

        labels = df[self.label_col]
        valid_labels = labels.dropna()
        class_counts = valid_labels.value_counts()
        n_samples = len(valid_labels)
        n_classes = len(class_counts)

        if n_classes == 0:
            df[self.name] = float("nan")
            return df

        class_weights = n_samples / (n_classes * class_counts)
        df[self.name] = labels.map(class_weights)
        return df
