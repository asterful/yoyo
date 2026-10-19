from typing import List

import pandas as pd

from yoyo.features.base import BaseFeature


class MultiplyFeature(BaseFeature):
    def __init__(self, name: str, columns: List[str]):
        super().__init__(name)
        if len(columns) < 2:
            raise ValueError("MultiplyFeature requires at least two columns.")
        self.columns = columns


    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        missing_columns = [column for column in self.columns if column not in df.columns]
        if missing_columns:
            raise KeyError(f"MultiplyFeature missing required columns: {missing_columns}")

        df[self.name] = df[self.columns].prod(
            axis=1, min_count=len(self.columns)
        )
        return df
