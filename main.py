import pandas as pd
import numpy as np

from yoyo.loaders import CcxtCachedLoader
from yoyo.features import FeaturePipeline
from yoyo.core import BacktestSplitter
from yoyo.core import WalkForwardCrossValidator
from yoyo.models import XGBoostQuantModel

from features.returns import ReturnFeature
from features.volatility import VolatilityFeature
from features.drop_columns import DropColumnsFeature
from features.ffd import FFDFeature
from features.log import LogFeature

from labels.triple_barrier import TripleBarrierLabel
from labels.shift_triple_barrier import ShiftTribleBarrierLabel

from sklearn.metrics import accuracy_score, precision_score

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

    model = XGBoostQuantModel(n_estimators=100, max_depth=5, learning_rate=0.1)
    validator = WalkForwardCrossValidator(model=model, data=train_data, n_splits=5)
    results = validator.run(target_col='target_tb_24h')

    fold_accuracies = []
    fold_precisions = []

    for res in results:
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        
        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        
        print(f"Fold {res['fold']} | Accuracy: {acc * 100:.2f}% | Precision: {prec * 100:.2f}%")

    # Print the final averages across all folds
    print("-" * 40)
    print(f"Average Accuracy:  {np.mean(fold_accuracies) * 100:.2f}%")
    print(f"Average Precision: {np.mean(fold_precisions) * 100:.2f}%")
