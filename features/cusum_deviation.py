import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class CUSUMDeviationFeature(BaseFeature):
    def __init__(self, name: str, window: int = 20):
        super().__init__(name)
        self.window = window

    def warmup_bars(self, modules: dict) -> int:
        return self.window

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if 'close' not in df.columns:
            raise KeyError(f"Dependency error: Feature '{self.name}' requires 'close' column.")

        close = df['close'].astype(np.float64).values
        n = len(close)

        if n == 0:
            df[self.name] = np.array([], dtype=np.float64)
            return df

        # Compute log returns / price differences
        diff = np.diff(np.log(close), prepend=np.log(close[0]))

        # Compute rolling volatility threshold h (using .bfill() instead of .fillna(method='bfill'))
        vol = pd.Series(diff).rolling(window=self.window).std().bfill().values

        pos_sum = 0.0
        neg_sum = 0.0
        cusum_stat = np.zeros(n, dtype=np.float64)

        for i in range(n):
            h = max(1e-6, vol[i])  # Prevent division by zero
            pos_sum = max(0.0, pos_sum + diff[i])
            neg_sum = min(0.0, neg_sum + diff[i])

            if pos_sum > h:
                cusum_stat[i] = pos_sum / h  # Normalized positive departure strength
                pos_sum = 0.0
            elif neg_sum < -h:
                cusum_stat[i] = neg_sum / h  # Normalized negative departure strength
                neg_sum = 0.0
            else:
                # Store normalized net accumulation between resets
                cusum_stat[i] = (pos_sum + neg_sum) / h

        df[self.name] = cusum_stat
        return df