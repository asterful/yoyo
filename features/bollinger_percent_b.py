from math import isfinite

import pandas as pd
from yoyo.features.base import BaseFeature


class BollingerPercentBFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        mean_column: str,
        std_span: int = 24,
        num_std: float = 2.0,
        price_col: str = 'close',
    ):
        super().__init__(name)
        self.mean_column = mean_column
        self.std_span = std_span
        self.num_std = num_std
        self.price_col = price_col

    def warmup_bars(self, modules: dict) -> int:
        mean_feature = modules.get(self.mean_column)
        mean_warmup = mean_feature.warmup_bars(modules) if mean_feature else 0
        price_feature = modules.get(self.price_col)
        price_warmup = price_feature.warmup_bars(modules) if price_feature else 0
        return max(mean_warmup, price_warmup + self.std_span)

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        missing_columns = [
            column for column in (self.price_col, self.mean_column)
            if column not in df.columns
        ]
        if missing_columns:
            raise KeyError(f"Feature '{self.name}' requires columns {missing_columns}.")
        if self.std_span <= 1 or not isfinite(self.num_std) or self.num_std <= 0:
            raise ValueError("The EWMA standard-deviation span must exceed 1 and its multiplier must be positive.")

        price = df[self.price_col]
        ewma_std = price.ewm(span=self.std_span, min_periods=self.std_span).std()
        band_width = 2 * self.num_std * ewma_std
        lower_band = df[self.mean_column] - self.num_std * ewma_std

        percent_b = (price - lower_band) / band_width
        df[self.name] = percent_b.where(band_width.ne(0), 0.5)
        return df
