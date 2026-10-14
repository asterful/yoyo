import pandas as pd

from yoyo.loaders import CcxtCachedLoader
from yoyo.features import FeaturePipeline

from features.returns import ReturnFeature
from features.volatility import VolatilityFeature
from features.drop_columns import DropColumnsFeature
from features.ffd import FFDFeature
from features.log import LogFeature

from labels.triple_barrier import TripleBarrierLabel

from yoyo.core import BacktestSplitter

if __name__ == "__main__":
    # 1. Instantiate the CCXT cached loader for Binance ETH/USDT futures
    loader = CcxtCachedLoader(symbol='ETH/USDT', exchange_id='binance', cache_dir='./data/cache')

    # 2. Build your feature pipeline
    pipeline = (
        FeaturePipeline(freq='1h')
        .register(ReturnFeature(name='return_1h', close_column='close'))
        .register(LogFeature(name='log_close', target_column='close'))
        .register(FFDFeature(name='ffd_log_1h', price_column='log_close', d=0.4, thresh=1e-4))
        .register(VolatilityFeature(name='vol_24h', window=24, return_column='return_1h'))
        .register(VolatilityFeature(name='vol_168h', window=168, return_column='return_1h'))
        .register(TripleBarrierLabel(
            name='target_tb_24h', 
            horizon=24,
            pt_mult=2.0, 
            sl_mult=1.0, 
            volatility_col='vol_24h'
        ))
        .register(DropColumnsFeature(['open', 'high', 'low', 'close', 'volume']))
    )

    splitter = BacktestSplitter(
        pipeline, 
        loader, 
        ends_at=pd.Timestamp("2026-01-01 00:00:00"),
        in_sample_window=pd.Timedelta(days=365),
        out_of_sample_window=pd.Timedelta(days=90),
        n_splits=3
    )

    # to iterate all the splits and print the train and test data shapes
    for i, (train_data, test_data) in enumerate(splitter.split()):
        print(f"Split {i+1}:")
        print(f"Train data shape: {train_data}")
        print(f"Test data shape: {test_data}")
        print()

    # to just get the last split and print the train and test data shapes
    train_data, test_data = splitter.get_last_split()
    print(f"Last Split:")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    """
    # 3. Run it! (First time fetches from Binance and saves to ./data_cache. Next time it loads instantly from disk.)
    processed_data = pipeline.generate(
        start_date=pd.Timestamp("2026-01-01 00:00:00"),
        end_date=pd.Timestamp("2026-03-31 23:00:00"),
        loader=loader
    )
    processed_data.to_csv('./data/processed_data.csv', index=True)
    print((processed_data['vol_24h'] < 0).any())
    print(processed_data[['return_1h', 'vol_24h']].describe())
    print(processed_data)
    """