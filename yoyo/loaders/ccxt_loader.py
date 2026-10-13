import os
import ccxt
import pandas as pd
from yoyo.loaders.base import BaseLoader

class CcxtCachedLoader(BaseLoader):
    def __init__(self, symbol: str = 'ETH/USDT', exchange_id: str = 'binance', cache_dir: str = './cache'):
        self.symbol = symbol
        # Initialize exchange with rate limiting enabled to prevent bans
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({'enableRateLimit': True})
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_filename(self, freq: str) -> str:
        """Generates a unique cache file name based on exchange, symbol, and frequency."""
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{self.exchange.id}_{safe_symbol}_{freq}.csv")

    def load(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str) -> pd.DataFrame:
        cache_file = self._get_cache_filename(freq)
        df = pd.DataFrame()

        # 1. Load from local cache if it exists
        if os.path.exists(cache_file):
            print(f"-> Loading cached data from: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
            df.index = pd.to_datetime(df.index)

        # 2. Check if cache covers the required range (including warmup buffer)
        cache_start = df.index.min() if not df.empty else None
        cache_end = df.index.max() if not df.empty else None

        # 3. If cache is missing or doesn't cover the required window, fetch from CCXT
        if df.empty or cache_start > start_date or cache_end < end_date:
            print(f"-> Cache miss or range out of bounds. Fetching from {self.exchange.id} via CCXT...")
            fetched_df = self._fetch_ccxt(start_date, end_date, freq)

            if not fetched_df.empty:
                if not df.empty:
                    # Merge existing cache with newly fetched data and deduplicate
                    df = pd.concat([df, fetched_df]).drop_duplicates().sort_index()
                else:
                    df = fetched_df
                
                # Update local cache file
                df.reset_index().to_csv(cache_file, index=False)
                print(f"-> Cache updated and saved to {cache_file}")

        if df.empty:
            raise ValueError(f"No data could be loaded for {self.symbol} at frequency {freq}.")

        # 4. Slice to the requested range and return
        return df.loc[start_date:end_date].copy()

    def _fetch_ccxt(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str) -> pd.DataFrame:
        """Paginates through CCXT historical OHLCV candles."""
        since = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        all_candles = []
        
        while since < end_ms:
            try:
                candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=freq, since=since, limit=1000)
                if not candles:
                    break
                
                since = candles[-1][0] + 1  # Move past the last fetched candle timestamp
                all_candles.extend(candles)
                
                # Stop if we've reached or passed the end date
                if candles[-1][0] >= end_ms:
                    break
            except Exception as e:
                print(f"-> Error fetching data from CCXT: {e}")
                break

        if not all_candles:
            return pd.DataFrame()

        # Format into a clean DataFrame
        fetched_df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        fetched_df['timestamp'] = pd.to_datetime(fetched_df['timestamp'], unit='ms')
        fetched_df = fetched_df.sort_values('timestamp').set_index('timestamp')
        
        return fetched_df