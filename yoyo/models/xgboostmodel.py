import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from .base import BaseModel

class XGBoostQuantModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the XGBoost classifier with any passed hyperparameters
        self.model = xgb.XGBClassifier(**self.params)


    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the XGBoost model on historical feature data and labels."""
        self.model.fit(X, y)


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate hard class predictions (0 or 1)."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(X)


    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate class probabilities for confidence thresholding."""
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(X)


    def save(self, filepath: str) -> None:
        """Save the trained model wrapper (params + fitted model) to disk."""
        joblib.dump(self, filepath)


    @classmethod
    def load(cls, filepath: str):
        """Load a trained model instance from disk."""
        return joblib.load(filepath)