import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class FFDFeature(BaseFeature):
    def __init__(self, name: str, price_column: str = 'close', d: float = 0.4, thresh: float = 1e-4):
        super().__init__(name)
        self.price_column = price_column
        self.d = d
        self.thresh = thresh
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

    @property
    def warmup_bars(self) -> int:
        return len(self.weights) - 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_column not in df.columns:
            raise KeyError(
                f"Dependency error: Feature '{self.name}' requires "
                f"price column '{self.price_column}', but it hasn't been computed yet."
            )
            
        series = df[self.price_column]
        width = len(self.weights)
        
        df[self.name] = series.rolling(window=width).apply(
            lambda window_vals: np.dot(window_vals, self.weights), 
            raw=True
        )
        
        return df