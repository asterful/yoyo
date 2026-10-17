import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class TrendScanningTValueFeature(BaseFeature):
    def __init__(self, name: str, window: int = 20):
        super().__init__(name)
        self.window = window
        
        # Precompute regressors for Ordinary Least Squares (OLS)
        # X matrix shape: (N, 2) where column 0 is intercept (1s) and column 1 is time index t [0, 1, ..., N-1]
        t = np.arange(self.window, dtype=np.float64)
        X = np.column_stack([np.ones(self.window), t])
        
        # Precompute the OLS pseudo-inverse: (X^T * X)^(-1) * X^T
        # beta_hat = XtX_inv_Xt @ y
        XtX_inv = np.linalg.inv(X.T @ X)
        self._XtX_inv_Xt = XtX_inv @ X.T  # Shape: (2, N)
        
        # Extract variance multiplier for the slope coefficient (index 1)
        self._var_beta1_factor = XtX_inv[1, 1]
        self._degrees_of_freedom = self.window - 2

    @property
    def warmup_bars(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in df.columns:
            raise KeyError(f"Dependency error: Feature '{self.name}' requires 'close' column.")

        # Take log of close prices to convert multiplicative returns to additive space
        log_close = np.log(df['close'].astype(np.float64)).values
        
        # Extract sliding windows of shape (num_samples, window)
        # Handle initial warm-up bars using NaN padding
        n_samples = len(log_close)
        t_values = np.full(n_samples, np.nan, dtype=np.float64)

        if n_samples < self.window:
            df[self.name] = t_values
            return df

        # Construct rolling 2D array of windows using stride tricks
        shape = (n_samples - self.window + 1, self.window)
        strides = (log_close.strides[0], log_close.strides[0])
        windows = np.lib.stride_tricks.as_strided(log_close, shape=shape, strides=strides)

        # 1. Compute linear regression slope (beta_1) across all windows
        # shape: (n_samples - window + 1,)
        beta_1 = np.dot(windows, self._XtX_inv_Xt[1, :])

        # 2. Compute fitted values Y_hat for each window
        # beta_0 = dot(windows, self._XtX_inv_Xt[0, :])
        beta_0 = np.dot(windows, self._XtX_inv_Xt[0, :])
        t_grid = np.arange(self.window, dtype=np.float64)
        
        # Y_hat shape: (n_samples - window + 1, window)
        y_hat = beta_0[:, np.newaxis] + np.outer(beta_1, t_grid)

        # 3. Compute Residual Sum of Squares (RSS) and Standard Error of beta_1
        residuals = windows - y_hat
        rss = np.sum(residuals ** 2, axis=1)
        
        # Residual variance (sigma_squared)
        sigma_sq = rss / self._degrees_of_freedom
        
        # Standard Error of beta_1 = sqrt(sigma_squared * (X^T * X)^(-1)[1,1])
        se_beta_1 = np.sqrt(np.maximum(1e-12, sigma_sq * self._var_beta1_factor))

        # 4. Compute t-value = beta_1 / se(beta_1)
        computed_t_stats = beta_1 / se_beta_1

        # Place output into the results array at offset index corresponding to window end
        t_values[self.window - 1:] = computed_t_stats

        df[self.name] = t_values
        return df