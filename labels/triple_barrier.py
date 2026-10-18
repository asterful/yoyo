import pandas as pd
import numpy as np
from yoyo.features import BaseFeature

class TripleBarrierLabel(BaseFeature):
    def __init__(self, name: str, horizon: int, scaler: float, ratio: float, volatility_col: str):
        super().__init__(name)
        self.horizon = horizon
        self.scaler = scaler
        self.ratio = ratio
        self.volatility_col = volatility_col


    def lookahead_bars(self, modules: dict) -> int:
        """Tells the FeaturePipeline how many future bars to fetch and sync."""
        return self.horizon


    def compute(self, df: pd.DataFrame, start_idx, end_idx) -> pd.DataFrame:
        required_cols = ['close', 'high', 'low', self.volatility_col]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"TripleBarrierLabel missing required column: '{col}'")

        close = df['close'].to_numpy()
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        vol = df[self.volatility_col].to_numpy()
        index_vals = df.index  # Timeframes / timestamps
        
        n = len(df)
        labels = np.full(n, np.nan)
        t1_times = np.full(n, np.datetime64('NaT'), dtype='datetime64[ns]')

        # Derive pt_mult and sl_mult from scaler and ratio
        # scaler: baseline volatility multiplier for stop-loss
        # ratio: profit-taking distance relative to stop-loss (e.g. ratio=2.0 means PT is 2x wider than SL)
        sl_mult = self.scaler
        pt_mult = self.scaler * self.ratio

        # Iterate through the timeline up to the point where the lookahead window fits
        for i in range(start_idx, n - self.horizon):
            entry_price = close[i]
            current_vol = vol[i]
            
            if pd.isna(current_vol) or current_vol <= 0 or pd.isna(entry_price):
                continue

            # Calculate absolute price levels for the profit-taking and stop-loss barriers
            upper_barrier = entry_price * (1.0 + pt_mult * current_vol)
            lower_barrier = entry_price * (1.0 - sl_mult * current_vol)

            # Scan the future price path window: [i + 1 : i + self.horizon + 1]
            future_highs = high[i + 1 : i + self.horizon + 1]
            future_lows = low[i + 1 : i + self.horizon + 1]

            # Find the indices of the first breach for each barrier
            hit_upper = np.where(future_highs >= upper_barrier)[0]
            hit_lower = np.where(future_lows <= lower_barrier)[0]

            idx_upper = hit_upper[0] if len(hit_upper) > 0 else 999999
            idx_lower = hit_lower[0] if len(hit_lower) > 0 else 999999

            # Determine which barrier was hit first
            if idx_upper < idx_lower:
                labels[i] = 1.0   # Upper barrier hit first (Profit)
                exit_idx = i + 1 + idx_upper
            elif idx_lower < idx_upper:
                labels[i] = -1.0  # Lower barrier hit first (Stop Loss)
                exit_idx = i + 1 + idx_lower
            else:
                labels[i] = 0.0   # Time expired / Vertical barrier hit
                exit_idx = i + self.horizon 

            # Store the actual timeframe/timestamp from the DataFrame index
            t1_times[i] = index_vals[min(exit_idx, n - 1)]

        df[self.name] = labels
        df[f"{self.name}_t1"] = t1_times
        return df