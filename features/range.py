import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class RangePositionFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        high_column: str = 'high',
        low_column: str = 'low',
        close_column: str = 'close',
        window: int = 20,
        epsilon: float = 1e-8
    ):
        super().__init__(name)
        self.high_column = high_column
        self.low_column = low_column
        self.close_column = close_column
        self.window = window
        self.epsilon = epsilon

    def warmup_bars(self, modules: dict) -> int:
        # Requires 'window' bars to calculate the rolling min/max
        return self.window

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        missing_cols = [col for col in [self.high_column, self.low_column, self.close_column] if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Dependency error: Feature '{self.name}' requires columns {missing_cols}, "
                f"but they haven't been computed yet."
            )

        if self.window <= 0 or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("The range window and denominator epsilon must be positive")

        rolling_high = df[self.high_column].rolling(window=self.window, min_periods=self.window).max()
        rolling_low = df[self.low_column].rolling(window=self.window, min_periods=self.window).min()
        
        # Calculate range position (0 to 1 scale)
        df[self.name] = (df[self.close_column] - rolling_low) / (rolling_high - rolling_low + self.epsilon)
        
        return df