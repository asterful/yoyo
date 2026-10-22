import pandas as pd

from yoyo.features import FeaturePipeline
from yoyo.core import CrossValidationReport
from yoyo.cross_validation import WalkForwardCrossValidator
from yoyo.cross_validation import PurgedKFoldCrossValidator
from yoyo.models import XGBoostQuantModel
from yoyo.models import HeuristicsModel 

from features import (
    ReturnFeature,
    LogFeature,
    FFDFeature,
    EWMAMeanFeature,
    BollingerPercentBFeature,
    ZScoreDistanceFeature,
    RollingHurstFeature,
    EWMAVolatilityFeature,
    GarmanKlassVolatilityFeature,
    DivideFeature,
    MultiplyFeature,
    SelectColumnsFeature,
    AverageTradeNotionalFeature,
    DollarVelocityFeature,
    TradeCountImbalanceFeature,
    VolumeImbalanceRatioFeature,
)

from labels.triple_barrier import TripleBarrierLabel
from labels.print_label_distribution import PrintLabelDistribution
from labels.shift_triple_barrier import ShiftTribleBarrierLabel
from labels.sample_uniqueness_weights import UniquenessWeightFeature
from labels.class_balanced_weights import ClassBalancedWeightFeature

from yoyo.loaders import ProcessedDollarBarsLoader

if __name__ == "__main__":
    
    dollar_bars_loader = ProcessedDollarBarsLoader()

    pipeline = (
        FeaturePipeline()

        # Base
        .register(ReturnFeature(name='return', close_column='close'))
        .register(LogFeature(name='log_close', target_column='close'))
        .register(EWMAMeanFeature(name='ewma_mean_24b', span=24, price_col='close'))
        .register(EWMAMeanFeature(name='ewma_mean_100b', span=100, price_col='close'))
        .register(EWMAVolatilityFeature(name='ewma_vol_24b', span=24, return_column='return'))
        .register(FFDFeature(name='ffd_log_0.3', source_column='log_close', d=0.3, thresh=1e-4))

        # Statistical overextension
        .register(BollingerPercentBFeature(
            name='bollinger_percent_b_24b',
            mean_column='ewma_mean_24b',
            std_span=24,
            price_col='close',
        ))
        .register(BollingerPercentBFeature(
            name='bollinger_percent_b_100b',
            mean_column='ewma_mean_100b',
            std_span=100,
            price_col='close',
        ))

        # Microstructure
        .register(VolumeImbalanceRatioFeature(name='volume_imbalance'))
        .register(DollarVelocityFeature(name='dollar_velocity'))

        # Volatility and Regime
        .register(GarmanKlassVolatilityFeature(name='garman_klass_vol_24b', window=24))
        .register(GarmanKlassVolatilityFeature(name='garman_klass_vol_100b', window=100))
        .register(DivideFeature(
            name='garman_klass_vol_ratio_24b_100b',
            numerator_column='garman_klass_vol_24b',
            denominator_column='garman_klass_vol_100b',
        ))

        # Bounciness
        .register(RollingHurstFeature(name='hurst_100b', window=100, price_col='ffd_log_0.3'))

        # Labeling
        .register(TripleBarrierLabel(
            name='target_tb_24b',
            horizon=24,
            scaler=3.0,
            ratio=1.0,
            volatility_col='ewma_vol_24b'
        ))
        .register(PrintLabelDistribution(label_col='target_tb_24b'))
        .register(ShiftTribleBarrierLabel(target_col='target_tb_24b'))

        # Weights
        .register(UniquenessWeightFeature(name='sample_uniqueness_weights', t1_col='target_tb_24b_t1'))
        .register(ClassBalancedWeightFeature(name='class_balanced_weights', label_col='target_tb_24b'))
        .register(MultiplyFeature(
            name='weight',
            columns=['sample_uniqueness_weights', 'class_balanced_weights'],
        ))

        # Filtering
        .register(SelectColumnsFeature([
            # Base
            'ffd_log_0.3',
            'ewma_mean_24b',
            'ewma_mean_100b',
            # Over extensin
            'bollinger_percent_b_24b',
            'bollinger_percent_b_100b',
            # Micro structure
            'volume_imbalance',
            'dollar_velocity',
            # Volatility and regime
            'garman_klass_vol_24b',
            'garman_klass_vol_ratio_24b_100b',
            # Bounciness
            'hurst_100b',
            # Targets / Weights
            'target_tb_24b',
            'target_tb_24b_t1',
            'weight',
        ]))
    )

    ends_at = pd.Timestamp("2026-09-30 23:59:59")
    train_data = pipeline.generate_backward(ends_at=ends_at, target_bars=50000, loader=dollar_bars_loader)
    train_data.to_csv("data/train_data.csv")

    validator = PurgedKFoldCrossValidator(
        model=HeuristicsModel(lower_thresh=0.3, upper_thresh=0.7),
        data=train_data,
        t1_col="target_tb_24b_t1",
        n_splits=5,
        embargo=pipeline.max_warmup(),
    )

    results = validator.run(
        target_col="target_tb_24b",
        sample_weight_col="weight",
    )

    CrossValidationReport(
        results,
        title="TRAIN PERIOD (PURGED K-FOLD CV) REPORT",
    ).print_detailed_report()
