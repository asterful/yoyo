import pandas as pd

from yoyo.features.pipeline import FeaturePipeline
from yoyo.loaders.base import BaseLoader

class BacktestSplitter:
    def __init__(
        self,
        pipeline: FeaturePipeline,
        loader: BaseLoader,
        ends_at: pd.Timestamp,
        in_sample_window: pd.Timedelta,
        out_of_sample_window: pd.Timedelta,
        n_splits: int = 1
    ):
        self.pipeline = pipeline
        self.loader = loader
        self.ends_at = ends_at
        self.in_sample_window = in_sample_window
        self.out_of_sample_window = out_of_sample_window
        self.n_splits = n_splits
        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        start_date = self.ends_at - (self.n_splits * self.out_of_sample_window) - self.in_sample_window
        return self.pipeline.generate(
            start_date=start_date,
            end_date=self.ends_at,
            loader=self.loader
        )

    def split(self):
        for i in range(self.n_splits):
            test_end = self.ends_at - (self.n_splits - 1 - i) * self.out_of_sample_window
            test_start = test_end - self.out_of_sample_window
            train_end = test_start
            train_start = train_end - self.in_sample_window

            train_mask = (self._data.index >= train_start) & (self._data.index < test_start)
            test_mask = (self._data.index >= test_start) & (self._data.index <= test_end)

            yield self._data.loc[train_mask], self._data.loc[test_mask]

    def get_last_split(self):
        return list(self.split())[-1]