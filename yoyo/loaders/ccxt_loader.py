import os
import ccxt
import pandas as pd
from yoyo.loaders.base import BaseLoader

class CcxtCachedLoader(BaseLoader):
    def __init__(self, symbol: str = 'ETH/USDT', exchange_id: str = 'binance', cache_dir: str = './cache'):
        self.symbol = symbol
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({'enableRateLimit': True})
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)


    def _get_cache_filename(self, freq: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
        """Generates a unique cache file name using filesystem-safe hyphenated timestamps."""
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d_%H-%M-%S')
        end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(self.cache_dir, f"{self.exchange.id}_{safe_symbol}_{freq}_{start_str}_{end_str}.csv")


    def _fetch_ccxt(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str) -> pd.DataFrame:
        """Paginates through CCXT historical OHLCV candles using built-in pagination, until, and CCXT's parse8601."""
        since = self.exchange.parse8601(pd.to_datetime(start_date).isoformat())
        until = self.exchange.parse8601(pd.to_datetime(end_date).isoformat())
        
        # Panics by default if network issues
        candles = self.exchange.fetch_ohlcv(
            self.symbol, 
            timeframe=freq, 
            since=since, 
            params={
                'paginate': True,
                'paginationCalls': 50,
                'until': until
            }
        )

        if not candles:
            return pd.DataFrame()

        fetched_df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        fetched_df['timestamp'] = pd.to_datetime(fetched_df['timestamp'], unit='ms')
        fetched_df = fetched_df.sort_values('timestamp').set_index('timestamp')

        return fetched_df.sort_index()


    def load(self, start_date: pd.Timestamp, end_date: pd.Timestamp, freq: str) -> pd.DataFrame:
        """Loads data from a precise timestamp-specific cache file or fetches it from CCXT if missing."""
        cache_file = self._get_cache_filename(freq, start_date, end_date)

        # 1. Load from local cache if the exact date/time range file exists
        if os.path.exists(cache_file):
            print(f"-> Loading cached data from: {cache_file}")
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            if not df.empty:
                df = df.sort_values('timestamp').set_index('timestamp')
                return df.sort_index()

        # 2. Fetch from CCXT if cache does not exist for this exact range
        print(f"-> Cache miss for range. Fetching from {self.exchange.id} via CCXT...")
        df = self._fetch_ccxt(start_date, end_date, freq)

        if df.empty:
            raise ValueError(f"No data could be loaded for {self.symbol} at frequency {freq} between {start_date} and {end_date}.")

        # 3. Save new cache file for this specific precise range
        df.reset_index().to_csv(cache_file, index=False)
        print(f"-> Cache saved to {cache_file}")

        # 4. Ensure data is strictly sorted at the end and return
        return df.sort_index()