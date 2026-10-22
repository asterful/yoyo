import pandas as pd
from yoyo.features.base import BaseFeature


class DivideFeature(BaseFeature):
    """Divide two feature columns, returning NaN for zero denominators."""

    def __init__(self, name: str, numerator_column: str, denominator_column: str):
        super().__init__(name)
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column

    def warmup_bars(self, modules: dict) -> int:
        dependencies = (
            modules.get(self.numerator_column),
            modules.get(self.denominator_column),
        )
        return max(
            (dependency.warmup_bars(modules) for dependency in dependencies if dependency),
            default=0,
        )

    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        required_columns = [self.numerator_column, self.denominator_column]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise KeyError(f"DivideFeature missing required columns: {missing_columns}")

        denominator = df[self.denominator_column]
        df[self.name] = df[self.numerator_column].div(denominator.where(denominator.ne(0)))
        return df
