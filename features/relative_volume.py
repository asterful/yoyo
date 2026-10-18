import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class RelativeVolumeFeature(BaseFeature):
    def __init__(self, name: str, window: int = 24, volume_column: str = 'volume'):
        super().__init__(name)
        self.window = window
        self.volume_column = volume_column

    def warmup_bars(self, modules: dict) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.volume_column not in df.columns:
            raise KeyError(f"Dependency error: Feature '{self.name}' requires '{self.volume_column}' column.")

        # Calculate the rolling average of volume
        rolling_vol_mean = df[self.volume_column].rolling(window=self.window).mean()

        # Compute Relative Volume (current bar volume / historical rolling mean)
        # Using np.where or letting pandas handle division by zero (results in inf/nan naturally during zero volume blocks)
        df[self.name] = df[self.volume_column] / rolling_vol_mean

        return df