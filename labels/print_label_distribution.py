import pandas as pd
import numpy as np
from yoyo.features import BaseFeature

class PrintLabelDistribution(BaseFeature):
    def __init__(self, label_col: str):
        """
        A diagnostic feature that prints the distribution of valid label values 
        (excluding NaNs) without modifying or adding columns to the DataFrame.
        """
        super().__init__("PrintLabelDistribution")
        self.label_col = label_col


    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.label_col not in df.columns:
            raise KeyError(f"PrintLabelDistribution missing required column: '{self.label_col}'")

        series = df[self.label_col]
        
        # Compute counts and percentages, dropping NaNs
        counts = series.value_counts(dropna=True)
        percentages = series.value_counts(normalize=True, dropna=True) * 100

        # Pretty print the distribution
        print(f"\n[Label Distribution Analysis: {self.name}]")
        print(f"Target Column: '{self.label_col}' | Valid Rows: {counts.sum()} (Total Rows: {len(df)})")
        print("-" * 50)
        
        # Define friendly names for TripleBarrierLabel outputs
        label_mapping = {
            1.0: "Profit (1.0)",
            -1.0: "Stop Loss (-1.0)",
            0.0: "Time Expired / Flat (0.0)"
        }

        for val, count in counts.items():
            pct = percentages[val]
            display_name = label_mapping.get(val, str(val))
            print(f"  {display_name:<30}: {count} ({pct:.2f}%)")
            
        print("-" * 50)

        return df