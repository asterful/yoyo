import pandas as pd
from yoyo.features.base import BaseFeature
from yoyo.loaders.base import BaseLoader

class FeaturePipeline:
    def __init__(self, freq: str):
        self.modules = []
        self.freq = freq


    def register(self, feature_module: BaseFeature):
        """Registers a feature module into the feature set."""
        self.modules.append(feature_module)
        return self


    def max_warmup(self) -> int:
        """Calculates the maximum warmup bars required across all registered modules."""
        return max((m.warmup_bars for m in self.modules), default=0)


    def max_lookahead(self) -> int:
        """Calculates the maximum lookahead bars required across all registered modules."""
        return max((m.lookahead_bars for m in self.modules), default=0)


    def generate(self, start_date: pd.Timestamp, end_date: pd.Timestamp, loader: BaseLoader) -> pd.DataFrame:
        """
        Executes the feature pipeline, returning a DataFrame with all computed features 
        and transformations for the specified date range and frequency, supporting 
        both backward warmup and forward lookahead requirements.
        """
        print("Running FeaturePipeline execution...")
        
        warmup_bars = self.max_warmup()
        lookahead_bars = self.max_lookahead()
        
        print(f"-> Maximum warmup required: {warmup_bars} bars.")
        print(f"-> Maximum lookahead required: {lookahead_bars} bars.")

        # Convert bar counts to timestamp offsets dynamically
        offset = pd.tseries.frequencies.to_offset(self.freq)
        warmup_timedelta = offset * warmup_bars
        lookahead_timedelta = offset * lookahead_bars

        # Expand the fetch window symmetrically (backward for warmup, forward for lookahead)
        fetch_start_date = start_date - warmup_timedelta
        fetch_end_date = end_date + lookahead_timedelta
        
        print(f"-> Target Frequency: {self.freq}")
        print(f"-> Requested window: {start_date} to {end_date}")
        print(f"-> Expanded fetch window: {fetch_start_date} to {fetch_end_date}")

        # Call the loader interface method with the fully expanded bounds
        print("-> Fetching data via loader object...")
        raw_df = loader.load(fetch_start_date, fetch_end_date, self.freq)
        
        if raw_df.empty:
            raise ValueError("The data loader returned an empty DataFrame!")

        # Execute each module sequentially on the expanded dataset
        df = raw_df.copy()
        for module in self.modules:
            df = module.compute(df)
            
        # Cleanly slice back to the user's requested timeframe, 
        final_df = df.loc[start_date:end_date].copy()
        
        print(f"-> Pipeline complete. Final dataset shape: {final_df.shape}")
        return final_df