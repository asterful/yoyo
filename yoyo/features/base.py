from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def warmup_bars(self) -> int:
        pass

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass