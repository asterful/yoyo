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
from features.rogers_satchell import RogersSatchellFeature
from features.vwap_distance import VWAPDistanceFeature
from features.range import RangePositionFeature
from features.bar_duration import BarDurationFeature
from features.bar_speed import BarSpeedFeature

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
        .register(FFDFeature(name='ffd_log_0.4', price_column='log_close', d=0.4, thresh=1e-4))
        .register(RogersSatchellFeature(name='rogers_24b', window=24))
        .register(RogersSatchellFeature(name='rogers_168b', window=168))
        .register(VolatilityFeature(name='vol_24b', window=24, return_column='return'))
        .register(VWAPDistanceFeature(name='vwap'))
        .register(RangePositionFeature(name='range'))
        .register(BarDurationFeature(name='log_duration', time_unit='seconds', log_transform=True))
        .register(BarSpeedFeature(name='bar_speed_24b', window=24))
        .register(TripleBarrierLabel(
            name='target_tb_24b', 
            horizon=24,
            scaler=3, 
            ratio=1.5, 
            volatility_col='vol_24b'
        ))
        .register(PrintLabelDistribution(label_col='target_tb_24b'))
        .register(ShiftTribleBarrierLabel(target_col='target_tb_24b'))
        .register(DropColumnsFeature(['open', 'high', 'low', 'close', 'volume', 'return', 'dollar_volume', 'log_close', 'vol_24b']))
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