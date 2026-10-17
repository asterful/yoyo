import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature

class EWMAVolatilityFeature(BaseFeature):
    def __init__(self, name: str, span: int = 24, return_column: str = 'return'):
        super().__init__(name)
        self.span = span
        self.return_column = return_column


    @property
    def warmup_bars(self) -> int:
        return self.span


    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure the return column exists or calculate it on the fly
        if self.return_column not in df.columns:
            if 'close' not in df.columns:
                raise KeyError(f"Dependency error: Feature '{self.name}' requires 'close' or '{self.return_column}' column.")
            returns = df['close'].pct_change()
        else:
            returns = df[self.return_column]

        # Calculate EWMA standard deviation (Marcos López de Prado's core volatility method)
        df[self.name] = returns.ewm(span=self.span).std()

        return df