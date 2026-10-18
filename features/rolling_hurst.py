import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class RollingHurstFeature(BaseFeature):
    """
    Calculates the rolling Hurst Exponent (H) over a specified window 
    using Rescaled Range (R/S) analysis.
    
    H < 0.5 -> Mean-reverting regime
    H ~ 0.5 -> Random walk
    H > 0.5 -> Trending regime
    """
    def __init__(self, name: str, window: int = 100, price_col: str = 'log_close'):
        super().__init__(name)
        self.window = window
        self.price_col = price_col

    def warmup_bars(self, modules: dict) -> int:
        return self.window

    @staticmethod
    def _compute_hurst(series: np.ndarray) -> float:
        """
        Computes the Hurst exponent for a 1D numpy array of prices/log-prices.
        """
        N = len(series)
        if N < 20:
            return 0.5
        
        # Calculate log returns inside the window
        returns = np.diff(series)
        if len(returns) == 0:
            return 0.5

        # Define sub-window sizes (lags) to compute R/S over
        max_lag = N // 2
        lags = np.unique(np.int32(np.linspace(8, max_lag, num=6)))
        
        rs_values = []
        for lag in lags:
            # Divide returns into non-overlapping chunks of length 'lag'
            num_chunks = len(returns) // lag
            if num_chunks == 0:
                continue
                
            chunks = returns[:num_chunks * lag].reshape(num_chunks, lag)
            
            # 1. Mean of each chunk
            chunk_means = np.mean(chunks, axis=1, keepdims=True)
            
            # 2. Cumulative mean-adjusted deviations
            cum_dev = np.cumsum(chunks - chunk_means, axis=1)
            
            # 3. Range (R) = max - min cumulative deviation
            r = np.max(cum_dev, axis=1) - np.min(cum_dev, axis=1)
            
            # 4. Standard Deviation (S) of each chunk
            s = np.std(chunks, axis=1, ddof=1)
            
            # Avoid division by zero
            valid_mask = s > 1e-8
            if not np.any(valid_mask):
                continue
                
            # Average R/S across valid chunks
            rs = np.mean(r[valid_mask] / s[valid_mask])
            if rs > 0:
                rs_values.append(rs)

        if len(rs_values) < 2:
            return 0.5

        # Hurst is the slope of log(R/S) vs log(lags)
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Linear regression slope
        slope, _ = np.polyfit(log_lags, log_rs, 1)
        
        # Clip Hurst between 0 and 1 for numerical stability
        return float(np.clip(slope, 0.0, 1.0))

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise KeyError(f"Feature '{self.name}' requires '{self.price_col}' column.")

        prices = df[self.price_col].to_numpy()
        n_samples = len(prices)
        hurst_series = np.full(n_samples, np.nan)

        # Rolling calculation using a sliding window
        for i in range(self.window, n_samples + 1):
            window_data = prices[i - self.window : i]
            hurst_series[i - 1] = self._compute_hurst(window_data)

        # Handle NaNs during warmup by forward filling or setting default 0.5
        df[self.name] = pd.Series(hurst_series, index=df.index).ffill().fillna(0.5)

        return df