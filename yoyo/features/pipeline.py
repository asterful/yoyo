import pandas as pd
from yoyo.features.base import BaseFeature

class FeaturePipeline:
    def __init__(self):
        self.modules = []


    def register(self, feature_module: BaseFeature):
        """Registers a feature module into the feature set."""
        self.modules.append(feature_module)
        return self


    def max_warmup(self) -> int:
        """Calculates the maximum warmup bars required across all registered modules."""
        return max((m.warmup_bars for m in self.modules), default=0)


    def generate(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str, loader: BaseLoader) -> pd.DataFrame:
        """Executes the feature pipeline, returning a DataFrame with all computed features for the specified date range and frequency."""
        print("Running FeaturePipeline execution...")
        
        warmup_bars = self.max_warmup()
        print(f"-> Maximum warmup required: {warmup_bars} bars.")

        # Convert bar count to a timestamp offset dynamically
        offset = pd.tseries.frequencies.to_offset(freq)
        warmup_timedelta = offset * warmup_bars

        # Expand the fetch window backward safely
        fetch_start_date = start_date - warmup_timedelta
        print(f"-> Target Frequency: {freq}")
        print(f"-> Requested window: {start_date} to {end_date}")
        print(f"-> Expanded fetch window (with warmup): {fetch_start_date} to {end_date}")

        # Call the loader interface method
        print("-> Fetching data via loader object...")
        raw_df = loader.load(fetch_start_date, end_date, freq)
        
        if raw_df.empty:
            raise ValueError("The data loader returned an empty DataFrame!")

        # Execute each module sequentially on the expanded dataset
        df = raw_df.copy()
        for module in self.modules:
            df = module.compute(df)
            
        # Cleanly slice back to the user's requested timeframe
        final_df = df.loc[start_date:end_date].copy()
        
        print(f"-> Pipeline complete. Final dataset shape: {final_df.shape}")
        return final_df