from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model on historical data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate hard class predictions (0 or 1)."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities for confidence thresholding."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save the trained model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str):
        """Load a trained model from disk."""
        pass