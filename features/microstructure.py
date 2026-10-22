import numpy as np
import pandas as pd
from yoyo.features.base import BaseFeature


def _require_columns(df: pd.DataFrame, name: str, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column not in df.columns:
            raise KeyError(
                f"Dependency error: Feature '{name}' requires '{column}' column."
            )


def _signed_ratio(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    total = positive + negative
    numerator = positive - negative
    return np.divide(
        numerator,
        total,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=total != 0,
    )


class VolumeImbalanceRatioFeature(BaseFeature):
    """Per-bar signed aggressor-volume imbalance in [-1, 1]."""

    def __init__(
        self,
        name: str,
        buy_volume_column: str = "buy_volume",
        sell_volume_column: str = "sell_volume",
    ):
        super().__init__(name)
        self.buy_volume_column = buy_volume_column
        self.sell_volume_column = sell_volume_column

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        _require_columns(
            df, self.name, (self.buy_volume_column, self.sell_volume_column)
        )
        buy_volume = df[self.buy_volume_column].to_numpy(dtype=np.float64)
        sell_volume = df[self.sell_volume_column].to_numpy(dtype=np.float64)
        df[self.name] = _signed_ratio(buy_volume, sell_volume)
        return df


class TradeCountImbalanceFeature(BaseFeature):
    """Per-bar signed aggressor trade-count imbalance in [-1, 1]."""

    def __init__(
        self,
        name: str,
        buy_trade_count_column: str = "buy_trade_count",
        sell_trade_count_column: str = "sell_trade_count",
    ):
        super().__init__(name)
        self.buy_trade_count_column = buy_trade_count_column
        self.sell_trade_count_column = sell_trade_count_column

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        _require_columns(
            df,
            self.name,
            (self.buy_trade_count_column, self.sell_trade_count_column),
        )
        buy_count = df[self.buy_trade_count_column].to_numpy(dtype=np.float64)
        sell_count = df[self.sell_trade_count_column].to_numpy(dtype=np.float64)
        df[self.name] = _signed_ratio(buy_count, sell_count)
        return df


class AverageTradeNotionalFeature(BaseFeature):
    """Average USD notional per underlying trade in the dollar bar."""

    def __init__(
        self,
        name: str,
        dollar_volume_column: str = "dollar_volume",
        trade_count_column: str = "trade_count",
    ):
        super().__init__(name)
        self.dollar_volume_column = dollar_volume_column
        self.trade_count_column = trade_count_column

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        _require_columns(
            df, self.name, (self.dollar_volume_column, self.trade_count_column)
        )
        dollar_volume = df[self.dollar_volume_column].to_numpy(dtype=np.float64)
        trade_count = df[self.trade_count_column].to_numpy(dtype=np.float64)
        df[self.name] = np.divide(
            dollar_volume,
            trade_count,
            out=np.full(trade_count.shape, np.nan, dtype=np.float64),
            where=trade_count != 0,
        )
        return df


class DollarVelocityFeature(BaseFeature):
    """Per-bar USD notional velocity, expressed in USD per second."""

    def __init__(
        self,
        name: str,
        dollar_volume_column: str = "dollar_volume",
        duration_column: str = "duration",
    ):
        super().__init__(name)
        self.dollar_volume_column = dollar_volume_column
        self.duration_column = duration_column

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        _require_columns(
            df, self.name, (self.dollar_volume_column, self.duration_column)
        )
        dollar_volume = df[self.dollar_volume_column].to_numpy(dtype=np.float64)
        duration_seconds = (
            df[self.duration_column].to_numpy(dtype=np.float64) / 1000.0
        )
        df[self.name] = np.divide(
            dollar_volume,
            duration_seconds,
            out=np.full(duration_seconds.shape, np.nan, dtype=np.float64),
            where=duration_seconds > 0,
        )
        return df
