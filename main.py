import os
import pandas as pd
import ccxt

from yoyo.features import FeaturePipeline
from features.returns import ReturnFeature
from features.volatility import VolatilityFeature


def fetch_binance_data(filepath: str):
    exchange = ccxt.binance({'enableRateLimit': True})    
    symbol = 'ETH/USDT'
    timeframe = '1h'
    since = exchange.parse8601('2025-01-01T00:00:00Z')
    until = exchange.parse8601('2025-12-31T23:59:59Z')
    print(f"Fetching {symbol} from Binance...")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, params={'until': until, 'paginate': True})
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"Successfully saved {len(df)} candles to: {filepath}")
    return df


def main():
    filepath = "data/raw.csv"

    # 1. Load or fetch data
    if not os.path.exists(filepath):
        print(f"'{filepath}' not found.")
        df = fetch_binance_data(filepath)
    else:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')

    # 2. Build the FeatureSet pipeline using fluent chaining
    feature_set = (
        FeaturePipeline()
        .register(ReturnFeature("return_1h", close_column="close"))
        .register(VolatilityFeature("volatility_24h", 24, return_column="return_1h"))
    )

    # 3. Transform data (computes features and automatically trims max warmup)
    processed_df = feature_set.transform(df)

    # 4. Inspect results
    print("\n--- FeatureSet Output Preview ---")
    print(processed_df.head())
    print("\n...")
    print(processed_df.tail())
    print(f"\nFinal Dataset Shape: {processed_df.shape}")
    print(f"Columns present: {list(processed_df.columns)}")


if __name__ == "__main__":
    main()