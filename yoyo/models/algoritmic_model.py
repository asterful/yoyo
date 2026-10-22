import numpy as np
import pandas as pd
import joblib
from .base import BaseModel

class HeuristicsModel(BaseModel):
    def __init__(
        self, 
        lower_thresh: float = 0.15, 
        upper_thresh: float = 0.85, 
        hurst_max: float = 0.52,
        max_vol_ratio: float = 1.8,
        min_dollar_velocity: float = None  # Optional: set based on your data's distribution
    ):
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.hurst_max = hurst_max
        self.max_vol_ratio = max_vol_ratio
        self.min_dollar_velocity = min_dollar_velocity

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None) -> None:
        """
        Optional dynamic fitting for velocity thresholds if not manually specified.
        """
        if self.min_dollar_velocity is None and 'dollar_velocity' in X.columns:
            # Set minimum velocity floor to the 10th percentile to avoid dead-liquidity zones
            self.min_dollar_velocity = X['dollar_velocity'].quantile(0.10)
        elif self.min_dollar_velocity is None:
            self.min_dollar_velocity = 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # 1. Extract Features Safely
        b_col = next((c for c in X.columns if 'bollinger_percent_b' in c), 'bollinger_percent_b_24b')
        h_col = 'hurst_100b' if 'hurst_100b' in X.columns else None
        v_ratio_col = 'garman_klass_vol_ratio_24b_100b' if 'garman_klass_vol_ratio_24b_100b' in X.columns else None
        vel_col = 'dollar_velocity' if 'dollar_velocity' in X.columns else None

        b = X[b_col].values
        h = X[h_col].values if h_col else np.zeros(len(X))
        v_ratio = X[v_ratio_col].values if v_ratio_col else np.ones(len(X))
        velocity = X[vel_col].values if vel_col else np.ones(len(X))

        # Default everything to Class 1 (Neutral / No Trade)
        preds = np.ones(len(X), dtype=int)

        # 2. Build Confluence Filters
        # Filter A: Hurst indicates mean-reverting regime (not trending)
        regime_ok = (h <= self.hurst_max) if h_col else np.ones(len(X), dtype=bool)

        # Filter B: Volatility isn't exploding into a runaway breakout
        vol_ok = (v_ratio <= self.max_vol_ratio) if v_ratio_col else np.ones(len(X), dtype=bool)

        # Filter C: Market has sufficient trading activity/liquidity (avoid dead zones)
        liquidity_ok = (velocity >= self.min_dollar_velocity) if vel_col else np.ones(len(X), dtype=bool)

        # Master filter combining market structure and regime checks
        valid_environment = regime_ok & vol_ok & liquidity_ok

        # 3. Apply Mean-Reversion Triggers
        # Oversold + Healthy Environment -> Class 2 (Expect bounce up)
        preds[(b < self.lower_thresh) & valid_environment] = 2
        
        # Overbought + Healthy Environment -> Class 0 (Expect drop down)
        preds[(b > self.upper_thresh) & valid_environment] = 0
        
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        proba = np.zeros((len(X), 3))
        for i, p in enumerate(preds):
            proba[i, int(p)] = 1.0
        return proba

    def save(self, filepath: str) -> None:
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return joblib.load(filepath)