import pandas as pd
from yoyo.features import BaseFeature

class ReturnFeature(BaseFeature):
    def __init__(self, name: str, close_column: str = 'close'):
        super().__init__(name)
        self.close_column = close_column

    def warmup_bars(self, modules: dict) -> int:
        return 1

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.close_column not in df.columns:
            raise KeyError(
                f"Dependency error: Feature '{self.name}' requires source "
                f"column '{self.close_column}', but it does not exist in the DataFrame."
            )
            
        df[self.name] = df[self.close_column].pct_change()
        return df