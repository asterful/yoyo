import pandas as pd
import numpy as np
import optuna

from yoyo.loaders import CcxtCachedLoader
from yoyo.features import FeaturePipeline
from yoyo.core import BacktestSplitter
from yoyo.core import BarBacktestSplitter
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
    backwards_loader = BackwardsDollarBarsLoader(symbol='ETH/USDT', exchange_id='binance', dollar_threshold=5_000_000, cache_dir='./data/cache')
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

    splitter = BarBacktestSplitter(
        pipeline, 
        backwards_loader, 
        ends_at=pd.Timestamp("2026-09-30 23:59:59"),
        in_sample_bars=3650,
        out_of_sample_bars=900,
        n_splits=1
    )

    train_data, test_data = splitter.get_last_split()

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    train_data.to_csv("data/dollar_bars_train_data.csv", index=True)
    test_data.to_csv("data/dollar_bars_test_data.csv", index=True)

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