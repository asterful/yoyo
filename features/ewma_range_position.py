import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class EWMARangePositionFeature(BaseFeature):
    """
    Measures price position within an EWMA-based band (mean +/- N standard deviations),
    clipped to a [0, 1] scale. Replaces hard-window rolling min/max with exponential decay.
    """
    def __init__(self, name: str, span: int = 20, num_std: float = 2.0, price_col: str = 'close'):
        super().__init__(name)
        self.span = span
        self.num_std = num_std
        self.price_col = price_col

    @property
    def warmup_bars(self) -> int:
        return self.span

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_col not in df.columns:
            raise KeyError(f"Feature '{self.name}' requires '{self.price_col}' column.")

        price = df[self.price_col]
        ewma_mean = price.ewm(span=self.span, min_periods=self.span).mean()
        ewma_std = price.ewm(span=self.span, min_periods=self.span).std()

        # Dynamic upper and lower bands based on EWMA volatility
        upper_band = ewma_mean + (self.num_std * ewma_std)
        lower_band = ewma_mean - (self.num_std * ewma_std)

        # Scale position between 0 and 1
        raw_position = (price - lower_band) / (upper_band - lower_band + 1e-8)
        
        # Clip values to keep them strictly in [0, 1] range
        df[self.name] = np.clip(raw_position, 0.0, 1.0)

        return df