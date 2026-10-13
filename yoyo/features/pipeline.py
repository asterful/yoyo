import pandas as pd
from typing import Callable, Union
from yoyo.features import BaseFeature

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

    def transform(self, start_date: Union[str, pd.Timestamp], 
                  end_date: Union[str, pd.Timestamp], 
                  loader: Callable[[str, str], pd.DataFrame]) -> pd.DataFrame:
        """
        1. Calculates max warmup.
        2. Expands the start date backwards to fetch necessary warmup bars.
        3. Calls the injected loader function.
        4. Computes all features on the expanded dataset.
        5. Trims the final dataset strictly to [start_date, end_date].
        """
        print("Running Pipeline transformation...")
        
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        
        warmup_bars = self.max_warmup()
        print(f"-> Maximum warmup required: {warmup_bars} bars.")

        # Expand the fetch window backward by the number of warmup bars 
        # (Assuming hourly ETH/USDT candles: 1 bar = 1 hour)
        fetch_start_ts = start_ts - pd.Timedelta(hours=warmup_bars)
        print(f"-> Requested window: {start_ts} to {end_ts}")
        print(f"-> Expanded fetch window (with warmup): {fetch_start_ts} to {end_ts}")

        # Inject the loader to fetch the raw data including the warmup buffer
        print("-> Fetching data via loader...")
        raw_df = loader(fetch_start_ts, end_ts)
        
        if raw_df.empty:
            raise ValueError("The data loader returned an empty DataFrame!")

        # Execute each module sequentially on the expanded dataset
        df = raw_df.copy()
        for module in self.modules:
            df = module.compute(df)
            
        # Cleanly slice back to the user's requested timeframe using timestamp indexing.
        # This automatically drops all the warmup rows while keeping them in memory 
        # long enough to prevent NaN corruption during rolling calculations!
        final_df = df.loc[start_ts:end_ts].copy()
        
        print(f"-> Pipeline complete. Final dataset shape: {final_df.shape}")
        return final_df