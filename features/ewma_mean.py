import pandas as pd
from yoyo.features.base import BaseFeature


class EWMAMeanFeature(BaseFeature):
    def __init__(self, name: str, span: int = 24, price_col: str = 'close'):
        super().__init__(name)
        self.span = span
        self.price_col = price_col

    def warmup_bars(self, modules: dict) -> int:
        dependency = modules.get(self.price_col)
        dependency_warmup = dependency.warmup_bars(modules) if dependency else 0
        return dependency_warmup + self.span

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise KeyError(f"Feature '{self.name}' requires '{self.price_col}' column.")

        df[self.name] = df[self.price_col].ewm(span=self.span, min_periods=self.span).mean()
        return df
