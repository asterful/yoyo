import pandas as pd
import numpy as np
import optuna

from yoyo.loaders import CcxtCachedLoader
from yoyo.features import FeaturePipeline
from yoyo.core import BacktestSplitter
from yoyo.core import WalkForwardCrossValidator
from yoyo.core import OptunaOptimizer
from yoyo.models import XGBoostQuantModel

from features.returns import ReturnFeature
from features.volatility import VolatilityFeature
from features.drop_columns import DropColumnsFeature
from features.ffd import FFDFeature
from features.log import LogFeature

from labels.triple_barrier import TripleBarrierLabel
from labels.print_label_distribution import PrintLabelDistribution
from labels.shift_triple_barrier import ShiftTribleBarrierLabel

from sklearn.metrics import accuracy_score, precision_score

from yoyo.loaders.backwards_dollars_loader import BackwardsDollarBarsLoader

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
            scaler=3, 
            ratio=1.5, 
            volatility_col='vol_24h'
        ))
        .register(PrintLabelDistribution(label_col='target_tb_24h'))
        .register(ShiftTribleBarrierLabel(target_col='target_tb_24h'))
        .register(DropColumnsFeature(['open', 'high', 'low', 'close', 'volume']))
    )

    splitter = BacktestSplitter(
        pipeline, 
        loader, 
        ends_at=pd.Timestamp("2026-01-01 00:00:00"),
        in_sample_window=pd.Timedelta(days=365),
        out_of_sample_window=pd.Timedelta(days=90),
        n_splits=1
    )

    train_data, test_data = splitter.get_last_split()

    """
    tuner = OptunaOptimizer(
        model_class=XGBoostQuantModel,
        validator_class=WalkForwardCrossValidator,
        train_data=train_data,
        target_col='target_tb_24h',
        n_splits=5
    )

    best_params = tuner.optimize(n_trials=30)
    tuner.evaluate_cv_folds(best_params)
    """

    start = pd.Timestamp('2026-01-01 00:00:00')
    end = pd.Timestamp('2026-09-30 23:59:59')

    # Initialize loader with a $5M dollar bar threshold
    dollar_loader = BackwardsDollarBarsLoader(
        symbol='ETH/USDT', 
        exchange_id='binance', 
        dollar_threshold=5_000_000,
        cache_dir='./data/cache'
    )

    # Load data (fetches/caches raw trades on first run, builds bars instantly on subsequent runs)
    dollar_bars_df = dollar_loader.load_backward(end, 10_000)  # Fetches 10,000 dollar bars backward from the end date
    dollar_bars_df.to_csv('./data/dollar_bars_eth_usdt.csv', index=True)
    print(dollar_bars_df.head())