"""Load precomputed ETHUSDT dollar bars from Parquet."""

from pathlib import Path

import pandas as pd

from yoyo.loaders.base import BackwardsLoader


class ProcessedDollarBarsLoader(BackwardsLoader):
    def __init__(self, file_path: str | Path | None = None) -> None:
        if file_path is None:
            file_path = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "processed_dollarBars"
                / "ETHUSDT-dyn-dollarBars-2020-01_2026-09.parquet"
            )
        self.file_path = Path(file_path)
        self._data: pd.DataFrame | None = None


    def load(
        self,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return bars in an optional range of naive UTC timestamps."""
        if start_date is not None and end_date is not None and start_date > end_date:
            raise ValueError("start_date must be less than or equal to end_date")

        if self._data is None:
            data = pd.read_parquet(self.file_path)
            data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
            self._data = data.set_index("timestamp").sort_index(kind="stable")

        data = self._data
        if start_date is not None:
            data = data.loc[start_date:]
        if end_date is not None:
            data = data.loc[:end_date]
        return data.copy()


    def load_backward(
        self, ends_at: pd.Timestamp, target_bars: int
    ) -> pd.DataFrame:
        if target_bars <= 0:
            raise ValueError("target_bars must be a positive integer")

        data = self.load(end_date=ends_at)
        if len(data) < target_bars:
            raise ValueError(
                f"Requested {target_bars} bars through {ends_at}; "
                f"only {len(data)} are available."
            )
        return data.tail(target_bars).copy()
