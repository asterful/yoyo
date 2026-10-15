import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class BarSpeedFeature(BaseFeature):
    def __init__(self, name: str, window: int = 24):
        super().__init__(name)
        self.window = window

    @property
    def warmup_bars(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError(f"Feature '{self.name}' requires a DatetimeIndex.")

        # Calculate duration in seconds
        duration = df.index.to_series().diff().dt.total_seconds()
        duration = duration.fillna(duration.median())

        # Compute rolling median duration
        rolling_median_duration = duration.rolling(window=self.window).median()

        # Speed Ratio: Historical Median / Current Duration
        # If normal duration is 3600s and current bar took 300s, speed is 12.0 (Urgent breakout)
        # If current bar took 7200s, speed is 0.5 (Dead market)
        df[self.name] = rolling_median_duration / duration

        return df