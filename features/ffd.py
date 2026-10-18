import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from yoyo.features.base import BaseFeature

class FFDFeature(BaseFeature):
    def __init__(self, name: str, source_column: str = 'close', d: float = 0.4, thresh: float = 1e-4, print_p_value: bool = True):
        super().__init__(name)
        self.source_column = source_column
        self.d = d
        self.thresh = thresh
        self.print_p_value = print_p_value
        self.weights = self._get_weights()

    def _get_weights(self) -> np.ndarray:
        """Calculates FFD weights based on order d and weight cutoff threshold."""
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (self.d - k + 1)
            if abs(w_k) < self.thresh:
                break
            w.append(w_k)
            k += 1
        return np.array(w[::-1])

    def warmup_bars(self, modules: dict) -> int:
        dependency = modules.get(self.source_column)
        dependency_warmup = dependency.warmup_bars(modules) if dependency else 0
        return dependency_warmup + len(self.weights) - 1

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.source_column not in df.columns:
            raise KeyError(
                f"Dependency error: Feature '{self.name}' requires "
                f"price column '{self.source_column}', but it hasn't been computed yet."
            )
            
        series = df[self.source_column]
        width = len(self.weights)
        
        # Calculate Fractional Differentiation
        df[self.name] = series.rolling(window=width).apply(
            lambda window_vals: np.dot(window_vals, self.weights), 
            raw=True
        )
        
        # Calculate and print p-value using the Augmented Dickey-Fuller test
        if self.print_p_value:
            clean_series = df[self.name].dropna()
            if len(clean_series) > 10:
                p_value = adfuller(clean_series, regression='c', autolag='AIC', result_object=False)[1]
                print(f"[FFDFeature: {self.name}] d={self.d} | ADF p-value: {p_value:.10f}")
            else:
                print(f"[FFDFeature: {self.name}] Not enough data points to compute ADF test.")
        
        return df