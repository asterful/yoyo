import pandas as pd
from typing import List

from yoyo.features.base import BaseFeature


class SelectColumnsFeature(BaseFeature):
    def __init__(self, columns_to_select: List[str]):
        super().__init__("select_columns")
        self.columns_to_select = columns_to_select

    def warmup_bars(self, modules: dict) -> int:
        return 0

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        existing_to_select = [col for col in self.columns_to_select if col in df.columns]
        return df.loc[:, existing_to_select]
