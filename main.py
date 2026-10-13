import pandas as pd

from yoyo.loaders import CcxtCachedLoader
from yoyo.features import FeaturePipeline

from features.returns import ReturnFeature
from features.volatility import VolatilityFeature

if __name__ == "__main__":
    # 1. Instantiate the CCXT cached loader for Binance ETH/USDT futures
    loader = CcxtCachedLoader(symbol='ETH/USDT', exchange_id='binance', cache_dir='./data_cache')

    # 2. Build your feature pipeline
    pipeline = (
        FeaturePipeline()
        .register(ReturnFeature(name='return_1h'))
        .register(VolatilityFeature(name='vol_24h', window=24, return_column='return_1h'))
        .register(VolatilityFeature(name='vol_168h', window=168, return_column='return_1h'))
    )

    # 3. Run it! (First time fetches from Binance and saves to ./data_cache. Next time it loads instantly from disk.)
    processed_data = pipeline.generate(
        start_date=pd.Timestamp("2026-01-01 00:00:00"),
        end_date=pd.Timestamp("2026-03-31 23:00:00"),
        freq='1h',
        loader=loader
    )

    print(processed_data)