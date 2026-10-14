import pandas as pd
from yoyo.features import BaseFeature

class ShiftTribleBarrierLabel(BaseFeature):
    def __init__(self, target_col: str):
        super().__init__("ShiftTribleBarrierLabel")
        self.target_col = target_col


    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_col not in df.columns:
            raise KeyError(f"EncodedLabelFeature missing input column: '{self.target_col}'")

        # Map [-1.0, 0.0, 1.0] to [0.0, 1.0, 2.0] for XGBoost classifier compatibility
        mapping = {-1: 0, 0: 1, 1: 2}
        
        df[self.target_col] = df[self.target_col].map(mapping)
        return df