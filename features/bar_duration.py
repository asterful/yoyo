import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class BarDurationFeature(BaseFeature):
    def __init__(self, name: str, time_unit: str = 'seconds', log_transform: bool = True):
        super().__init__(name)
        self.time_unit = time_unit
        self.log_transform = log_transform

    @property
    def warmup_bars(self) -> int:
        return 2  # Needs at least 2 bars to calculate a time difference

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError(f"Feature '{self.name}' requires a DatetimeIndex representing bar timestamps.")

        # Calculate time delta between consecutive dollar bars
        time_diff = df.index.to_series().diff()

        if self.time_unit == 'seconds':
            duration = time_diff.dt.total_seconds()
        elif self.time_unit == 'minutes':
            duration = time_diff.dt.total_seconds() / 60.0
        else:
            duration = time_diff.dt.total_seconds()

        # Handle the first row NaN caused by diff() using the median of the series
        duration = duration.fillna(duration.median())

        # Log transform is recommended because time deltas can be heavily skewed 
        # (e.g., a bar takes 10 seconds normally, but 6 hours overnight)
        if self.log_transform:
            # Use log1p to safely handle any zero-second anomalies
            df[self.name] = np.log1p(duration)
        else:
            df[self.name] = duration

        return df