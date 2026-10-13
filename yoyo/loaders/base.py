from abc import ABC, abstractmethod
import pandas as pd

class BaseLoader(ABC):
    @abstractmethod
    def load(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str) -> pd.DataFrame:
        """
        Fetches raw data for the given timeframe and frequency.
        Must return a DataFrame with a datetime index named 'timestamp' (or similar) 
        and sorted chronologically.
        """
        pass