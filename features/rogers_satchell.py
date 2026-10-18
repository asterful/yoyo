import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class RogersSatchellFeature(BaseFeature):
    def __init__(self, name: str, window: int = 20):
        super().__init__(name)
        self.window = window

    def warmup_bars(self, modules: dict) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Dependency error: Feature '{self.name}' requires '{col}' column.")

        log_ho = np.log(df['high'] / df['open'])
        log_hc = np.log(df['high'] / df['close'])
        log_lo = np.log(df['low'] / df['open'])
        log_lc = np.log(df['low'] / df['close'])

        rs_single_bar = log_hc * log_ho + log_lc * log_lo

        rolling_var = rs_single_bar.rolling(window=self.window).sum() / self.window
        df[self.name] = np.sqrt(np.maximum(0, rolling_var))

        return df