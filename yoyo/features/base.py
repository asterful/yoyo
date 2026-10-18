from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    def __init__(self, name: str):
        self.name = name


    def warmup_bars(self, modules: dict) -> int:
        """
        Maximum bars required to build this feature.
        Receives modules dict from the pipeline to let each feature figure out their own dependencies
        """
        return 0


    def lookahead_bars(self, modules: dict) -> int:
        """
        Maximum bars required to build this feature.
        Receives modules dict from the pipeline to let each feature figure out their own dependencies
        """
        return 0


    @abstractmethod
    def compute(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Compute features; the requested training window is [start_idx, end_idx)."""
        pass