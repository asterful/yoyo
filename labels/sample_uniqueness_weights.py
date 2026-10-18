import pandas as pd
import numpy as np
from yoyo.features import BaseFeature

class UniquenessWeightFeature(BaseFeature):
    def __init__(self, name: str, t1_col: str):
        super().__init__(name)
        self.t1_col = t1_col


    def lookahead_bars(self, modules: dict) -> int:
        """Uniqueness weights are computed retrospectively from existing t1 timestamps."""
        return 0


    def _compute_concurrency(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns the number of overlapping trades (c_t) at each bar.
        Equivalent to Marcos López de Prado's mpNumCoEvents logic.
        """
        close_idx = df.index
        
        # We only care about rows that have a valid t1 exit timestamp
        valid_mask = df[self.t1_col].notna()
        t1 = df.loc[valid_mask, self.t1_col]

        if len(t1) == 0:
            return pd.Series(0.0, index=close_idx)

        # Unclosed events still impact other weights up to the end of the series
        t1 = t1.fillna(close_idx[-1])

        # Initialize count series for all bars
        count = pd.Series(0.0, index=close_idx)
        for t_in, t_out in t1.items():
            count.loc[t_in:t_out] += 1.0

        return count


    def _compute_average_uniqueness(self, df: pd.DataFrame, num_co_events: pd.Series) -> pd.Series:
        """
        Returns the average uniqueness for each bar/event.
        Equivalent to Marcos López de Prado's mpSampleTW logic.
        """
        valid_mask = df[self.t1_col].notna()
        t1 = df.loc[valid_mask, self.t1_col]

        weights = pd.Series(index=df.index, dtype=float)

        for t_in, t_out in t1.items():
            event_co_events = num_co_events.loc[t_in:t_out]
            if len(event_co_events) > 0 and (event_co_events > 0).all():
                weights.loc[t_in] = (1.0 / event_co_events).mean()
            else:
                weights.loc[t_in] = np.nan

        return weights


    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        if self.t1_col not in df.columns:
            raise KeyError(f"UniquenessWeightFeature missing required column: '{self.t1_col}'")

        # Step 1: Compute concurrency per bar using only t1 timestamps
        num_co_events = self._compute_concurrency(df)

        # Step 2: Compute average uniqueness weights per event
        weights = self._compute_average_uniqueness(df, num_co_events)

        df[f"{self.name}_concurrency"] = num_co_events
        df[self.name] = weights
        return df