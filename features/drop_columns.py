import pandas as pd
from typing import List
from yoyo.features.base import BaseFeature

class DropColumnsFeature(BaseFeature):
    def __init__(self, columns_to_drop: List[str]):
        super().__init__("drop_raw_columns")
        self.columns_to_drop = columns_to_drop

    def warmup_bars(self, modules: dict) -> int:
        return 0

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_to_drop = [col for col in self.columns_to_drop if col in df.columns]
        return df.drop(columns=existing_to_drop)