import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class LogFeature(BaseFeature):
    def __init__(self, name: str, target_column: str = 'close'):
        super().__init__(name)
        self.target_column = target_column

    def warmup_bars(self, modules: dict) -> int:
        dependency = modules.get(self.target_column)
        return dependency.warmup_bars(modules) if dependency else 0

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.target_column not in df.columns:
            raise KeyError(f"Dependency error: Column '{self.target_column}' not found.")
        
        df[self.name] = np.log(df[self.target_column])
        return df