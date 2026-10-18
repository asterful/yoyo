import pandas as pd
import numpy as np
import optuna

from yoyo.features import FeaturePipeline
from yoyo.core import BarBacktestSplitter
from yoyo.core import OptunaOptimizer
from yoyo.cross_validation import WalkForwardCrossValidator
from yoyo.models import XGBoostQuantModel

from features import (
    ReturnFeature,
    LogFeature,
    FFDFeature,
    ZScoreDistanceFeature,
    RollingHurstFeature,
    EWMAVolatilityFeature,
    DropColumnsFeature,
)

from labels.triple_barrier import TripleBarrierLabel
from labels.print_label_distribution import PrintLabelDistribution
from labels.shift_triple_barrier import ShiftTribleBarrierLabel
from labels.sample_uniqueness_weights import UniquenessWeightFeature

from sklearn.metrics import accuracy_score, precision_score

from yoyo.loaders import ProcessedDollarBarsLoader

if __name__ == "__main__":
    
    dollar_bars_loader = ProcessedDollarBarsLoader()

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
        .register(UniquenessWeightFeature(name='uniqueness_weights', t1_col='target_tb_24b_t1'))
        .register(PrintLabelDistribution(label_col='target_tb_24b'))
        .register(ShiftTribleBarrierLabel(target_col='target_tb_24b'))
        .register(DropColumnsFeature(['open', 'high', 'low', 'close', 'volume', 'return', 'log_close']))
    )


    ends_at = pd.Timestamp("2026-09-30 23:59:59")
    train_data = pipeline.generate_backward(ends_at=ends_at, target_bars=1, loader=dollar_bars_loader)
    print(train_data)
    train_data.to_csv("data/train_data.csv", index=True)
