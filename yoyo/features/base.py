from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    def warmup_bars(self) -> int:
        return 0

    @property
    def lookahead_bars(self) -> int:
        return 0

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass