import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class VWAPDistanceFeature(BaseFeature):
    def __init__(self, name: str, price_column: str = 'close', method: str = 'percentage'):
        super().__init__(name)
        self.price_column = price_column
        self.method = method


    def warmup_bars(self, modules: dict) -> int:
        return 0 


    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.price_column not in df.columns:
            raise KeyError(f"Missing price column '{self.price_column}'")
            
        if 'vwap' in df.columns:
            vwap_series = df['vwap']
        elif 'dollar_volume' in df.columns and 'volume' in df.columns:
            vwap_series = df['dollar_volume'] / df['volume']
        else:
            raise KeyError("Feature requires either a 'vwap' column or 'dollar_volume' and 'volume' columns.")

        # Calculate distance
        if self.method == 'percentage':
            df[self.name] = (df[self.price_column] - vwap_series) / vwap_series
        elif self.method == 'absolute':
            df[self.name] = df[self.price_column] - vwap_series
        else:
            raise ValueError(f"Invalid method '{self.method}'. Choose 'percentage' or 'absolute'.")

        return df