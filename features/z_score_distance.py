import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class ZScoreDistanceFeature(BaseFeature):
    """
    Measures the rolling EWMA Z-score of a price or feature column, 
    representing how many standard deviations the current value is away 
    from its exponential moving average.
    """
    def __init__(self, name: str, span: int = 24, price_col: str = 'close'):
        super().__init__(name)
        self.span = span
        self.price_col = price_col


    def warmup_bars(self, modules: dict) -> int:
        dep = modules.get(self.price_col)
        dep_warmup = dep.warmup_bars(modules) if dep else 0
        return dep_warmup + (self.span)


    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise KeyError(f"Feature '{self.name}' requires '{self.price_col}' column.")

        price = df[self.price_col]
        
        # Calculate EWMA mean and standard deviation
        ewma_mean = price.ewm(span=self.span, min_periods=self.span).mean()
        ewma_std = price.ewm(span=self.span, min_periods=self.span).std()

        # Compute z-score with a small epsilon to avoid division by zero
        z_score = (price - ewma_mean) / (ewma_std + 1e-8)

        # Assign directly to the dataframe column
        df[self.name] = z_score

        return df