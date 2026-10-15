import pandas as pd

from yoyo.features.pipeline import FeaturePipeline
from yoyo.loaders.base import BackwardsLoader

class BarBacktestSplitter:
    def __init__(
        self,
        pipeline: FeaturePipeline,
        loader: BackwardsLoader,
        ends_at: pd.Timestamp,
        in_sample_bars: int,
        out_of_sample_bars: int,
        n_splits: int = 1
    ):
        self.pipeline = pipeline
        self.loader = loader
        self.ends_at = ends_at
        self.in_sample_bars = in_sample_bars
        self.out_of_sample_bars = out_of_sample_bars
        self.n_splits = n_splits
        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        # Total bars needed = training bars + (n_splits * test bars)
        embargo_bars = self.pipeline.max_lookahead()
        total_bars = self.in_sample_bars + (self.n_splits * self.out_of_sample_bars) + embargo_bars
        return self.pipeline.generate_backward(
            ends_at=self.ends_at,
            target_bars=total_bars,
            loader=self.loader
        )

    def split(self):
        embargo_bars = self.pipeline.max_lookahead()
        for i in range(self.n_splits):
            # Calculate integer positional indices for walk-forward splits
            test_end_idx = len(self._data) - (self.n_splits - 1 - i) * self.out_of_sample_bars
            test_start_idx = test_end_idx - self.out_of_sample_bars

            train_end_idx = test_start_idx - embargo_bars
            train_start_idx = train_end_idx - self.in_sample_bars

            if train_start_idx < 0:
                raise ValueError(
                    f"Training window start index is negative ({train_start_idx}). "
                    "Reduce in_sample_bars or ensure sufficient historical data is fetched."
                )

            # Use .iloc for precise row-count slicing on dollar bars
            train_df = self._data.iloc[train_start_idx:train_end_idx]
            embargo_df = self._data.iloc[train_end_idx:test_start_idx]
            test_df = self._data.iloc[test_start_idx:test_end_idx]

            yield train_df, embargo_df, test_df

    def get_last_split(self):
        return list(self.split())[-1]