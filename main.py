import pandas as pd
import numpy as np
import optuna

from yoyo.loaders import CcxtCachedLoader
from yoyo.features import FeaturePipeline
from yoyo.core import BacktestSplitter
from yoyo.core import BarBacktestSplitter
from yoyo.core import OptunaOptimizer
from yoyo.cross_validation import WalkForwardCrossValidator
from yoyo.models import XGBoostQuantModel

from features.returns import ReturnFeature
from features.volatility import VolatilityFeature
from features.drop_columns import DropColumnsFeature
from features.ffd import FFDFeature
from features.log import LogFeature
from features.rogers_satchell import RogersSatchellFeature
from features.vwap_distance import VWAPDistanceFeature
from features.range import RangePositionFeature
from features.bar_duration import BarDurationFeature
from features.bar_speed import BarSpeedFeature
from features.ewma_volatility import EWMAVolatilityFeature
from features.ewma_range_position import EWMARangePositionFeature
from features.momentum import TrendScanningTValueFeature
from features.order_flow_imbalance import OrderFlowImbalanceFeature
from features.cusum_deviation import CUSUMDeviationFeature
from features.z_score_distance import ZScoreDistanceFeature
from features.rolling_hurst import RollingHurstFeature

from labels.triple_barrier import TripleBarrierLabel
from labels.print_label_distribution import PrintLabelDistribution
from labels.shift_triple_barrier import ShiftTribleBarrierLabel

from sklearn.metrics import accuracy_score, precision_score

from yoyo.loaders.backwards_dollars_loader import BackwardsDollarBarsLoader

if __name__ == "__main__":
    
    dollar_bars_loader = BackwardsDollarBarsLoader(
        symbol='ETH/USDT', 
        exchange_id='binance', 
        dollar_threshold=5_000_000, 
        cache_dir='./data/cache'
    )
    
    pipeline = (
        FeaturePipeline()
        .register(ReturnFeature(name='return', close_column='close'))
        .register(LogFeature(name='log_close', target_column='close'))
        .register(FFDFeature(name='ffd_log_0.18', source_column='log_close', d=0.18, thresh=1e-4))
        .register(ZScoreDistanceFeature(name='zscore_ffd_24b', span=24, price_col='ffd_log_0.18'))
        .register(RollingHurstFeature(name='hurst_100b', window=100, price_col='log_close'))
        .register(EWMAVolatilityFeature(name='ewma_vol_24b', span=24, return_column='return'))
        .register(EWMAVolatilityFeature(name='ewma_vol_168b', span=168, return_column='return'))
        .register(TripleBarrierLabel(
            name='target_tb_24b', 
            horizon=24,
            scaler=3.8, 
            ratio=1.0, 
            volatility_col='ewma_vol_24b'
        ))
        .register(PrintLabelDistribution(label_col='target_tb_24b'))
        .register(ShiftTribleBarrierLabel(target_col='target_tb_24b'))
        .register(DropColumnsFeature(['open', 'high', 'low', 'close', 'volume', 'return', 'dollar_volume', 'log_close']))
    )

    splitter = BarBacktestSplitter(
        pipeline, 
        dollar_bars_loader, 
        ends_at=pd.Timestamp("2026-09-30 23:59:59"),
        in_sample_bars=3650,
        out_of_sample_bars=900,
        n_splits=1
    )

    train_data, embargo_data, test_data = splitter.get_last_split()
    train_data.to_csv('data/train_data.csv', index=True)

    tuner = OptunaOptimizer(
        model_class=XGBoostQuantModel,
        validator_class=WalkForwardCrossValidator,
        train_data=train_data,
        target_col='target_tb_24b',
        n_splits=5,
        gap=pipeline.max_lookahead()
    )

    best_params = tuner.optimize(n_trials=30)
    tuner.evaluate_cv_folds(best_params)