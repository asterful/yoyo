import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class OrderFlowImbalanceFeature(BaseFeature):
    def __init__(self, name: str, window: int = 20):
        super().__init__(name)
        self.window = window

    def warmup_bars(self, modules: dict) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ['close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Dependency error: Feature '{self.name}' requires '{col}' column.")

        close = df['close'].astype(np.float64).values
        volume = df['volume'].astype(np.float64).values
        n = len(close)

        if n == 0:
            df[self.name] = np.array([], dtype=np.float64)
            return df

        # 1. Compute price changes (delta)
        price_diff = np.diff(close, prepend=close[0])

        # 2. Compute Tick Rule Sign sequence b_t
        b = np.zeros(n, dtype=np.float64)
        last_sign = 1.0  # Default initial sign

        for i in range(n):
            if price_diff[i] > 0:
                last_sign = 1.0
            elif price_diff[i] < 0:
                last_sign = -1.0
            # If price_diff == 0, keep last_sign (López de Prado tick rule)
            b[i] = last_sign

        # 3. Calculate signed volume = b_t * v_t
        signed_volume = b * volume

        # 4. Compute rolling order flow imbalance ratio = sum(signed_volume) / sum(volume)
        s_vol_series = pd.Series(signed_volume)
        tot_vol_series = pd.Series(volume)

        rolling_signed_vol = s_vol_series.rolling(window=self.window).sum()
        rolling_total_vol = tot_vol_series.rolling(window=self.window).sum()

        # Prevent division by zero
        ofi_ratio = rolling_signed_vol / np.maximum(1e-12, rolling_total_vol)

        df[self.name] = ofi_ratio.values
        return df