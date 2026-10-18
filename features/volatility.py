import pandas as pd
from yoyo.features import BaseFeature

class VolatilityFeature(BaseFeature):
    def __init__(self, name: str, window: int, return_column: str):
        super().__init__(name)
        self.window = window
        self.return_column = return_column

    def warmup_bars(self, modules: dict) -> int:
        dependency = modules.get(self.return_column)
        dependency_warmup = dependency.warmup_bars(modules) if dependency else 0
        return dependency_warmup + self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.return_column not in df.columns:
            raise KeyError(
                f"Dependency error: Feature '{self.name}' requires "
                f"column '{self.return_column}', but it hasn't been computed yet."
            )
            
        df[self.name] = df[self.return_column].rolling(window=self.window).std()
        return df