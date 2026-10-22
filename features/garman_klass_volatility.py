import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature


class GarmanKlassVolatilityFeature(BaseFeature):
    def __init__(self, name: str, window: int = 24):
        super().__init__(name)
        if window <= 0:
            raise ValueError("The Garman-Klass rolling window must be positive.")
        self.window = window

    def warmup_bars(self, modules: dict) -> int:
        return self.window

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise KeyError(
                f"Feature '{self.name}' requires columns {missing_columns}."
            )

        log_high_low = np.log(df['high'] / df['low'])
        log_close_open = np.log(df['close'] / df['open'])
        bar_variance = (
            0.5 * log_high_low.pow(2)
            - (2 * np.log(2) - 1) * log_close_open.pow(2)
        )
        rolling_variance = bar_variance.rolling(
            window=self.window,
            min_periods=self.window,
        ).mean()
        df[self.name] = np.sqrt(rolling_variance.clip(lower=0))
        return df
