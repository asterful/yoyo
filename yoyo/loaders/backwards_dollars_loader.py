import os
import ccxt
import pandas as pd
import numpy as np


class BackwardsDollarBarsLoader():
    def __init__(self, symbol: str = 'ETH/USDT', exchange_id: str = 'binance', cache_dir: str = './cache', dollar_threshold: float = 5_000_000):
        self.symbol = symbol
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({'enableRateLimit': True})
        self.cache_dir = cache_dir
        self.dollar_threshold = dollar_threshold
        os.makedirs(self.cache_dir, exist_ok=True)


    def _get_monthly_cache_path(self, year: int, month: int) -> str:
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        return os.path.join(self.cache_dir, f"{self.exchange.id}_{safe_symbol}_1m_{year:04d}-{month:02d}.csv")


    def _get_range_cache_path(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
        safe_symbol = self.symbol.replace('/', '_').replace(':', '_')
        start_str = pd.to_datetime(start_date).strftime('%Y-%m-%d_%H-%M-%S')
        end_str = pd.to_datetime(end_date).strftime('%Y-%m-%d_%H-%M-%S')
        return os.path.join(self.cache_dir, f"{self.exchange.id}_{safe_symbol}_1m_{start_str}_{end_str}.csv")


    def _fetch_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        since = self.exchange.parse8601(start_date.isoformat())
        until = self.exchange.parse8601(end_date.isoformat())
        
        candles = self.exchange.fetch_ohlcv(
            self.symbol, timeframe='1m', since=since, 
            params={'paginate': True, 'paginationCalls': 100, 'until': until}
        )
        if not candles:
            return pd.DataFrame()
            
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.sort_values('timestamp').set_index('timestamp')


    def _load_or_fetch_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp, cache_path: str) -> pd.DataFrame:
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=['timestamp'])
            if not df.empty:
                return df.sort_values('timestamp').set_index('timestamp')

        print(f"-> Cache miss. Fetching range from {start_date} to {end_date} from {self.exchange.id}...")
        df = self._fetch_range(start_date, end_date)
        if not df.empty:
            df.reset_index().to_csv(cache_path, index=False)
        return df


    @staticmethod
    def _get_previous_month(dt: pd.Timestamp) -> pd.Timestamp:
        return pd.Timestamp(year=dt.year, month=dt.month, day=1) - pd.Timedelta(days=1)


    def _build_dollar_bars_forward(self, df_1m, ends_at):
        df = df_1m.loc[:ends_at]
        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'dollar_volume'])

        dollar_values = df['close'].values * df['volume'].values
        threshold = self.dollar_threshold

        bars = []
        current_open = df['open'].iat[0]
        current_high = df['high'].iat[0]
        current_low = df['low'].iat[0]
        current_volume = 0.0
        current_dollar_vol = 0.0

        for i in range(len(df)):
            t = df.index[i]
            o = df['open'].iat[i]
            h = df['high'].iat[i]
            l = df['low'].iat[i]
            c = df['close'].iat[i]
            v = df['volume'].iat[i]
            dv = dollar_values[i]

            if current_volume == 0.0:
                current_open = o

            current_high = max(current_high, h)
            current_low = min(current_low, l)
            current_volume += v
            current_dollar_vol += dv

            if current_dollar_vol >= threshold:
                bars.append({
                    'timestamp': t,
                    'open': current_open,
                    'high': current_high,
                    'low': current_low,
                    'close': c,
                    'volume': current_volume,
                    'dollar_volume': current_dollar_vol
                })
                current_high = -np.inf
                current_low = np.inf
                current_volume = 0.0
                current_dollar_vol = 0.0

        return pd.DataFrame(bars).set_index('timestamp')


    def load_backward(self, ends_at: pd.Timestamp, target_bars: int) -> pd.DataFrame:
        print(f"-> Loading backward: targeting {target_bars} dollar bars ending at {ends_at}...")
        
        monthly_dfs = []
        curr = ends_at
        is_first_iteration = True

        while True:
            month_start = pd.Timestamp(year=curr.year, month=curr.month, day=1)
            month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)

            # Use range cache if it's a mid-month partial endpoint, otherwise use full monthly cache
            if is_first_iteration and ends_at < month_end:
                cache_path = self._get_range_cache_path(month_start, ends_at)
                df_month = self._load_or_fetch_range(month_start, ends_at, cache_path)
            else:
                cache_path = self._get_monthly_cache_path(curr.year, curr.month)
                df_month = self._load_or_fetch_range(month_start, month_end, cache_path)

            if df_month.empty:
                break
                
            monthly_dfs.insert(0, df_month)
            is_first_iteration = False

            full_df = pd.concat(monthly_dfs).loc[:ends_at].sort_index()
            bars_df = self._build_dollar_bars_forward(full_df, ends_at)

            if len(bars_df) >= target_bars:
                print(f"-> Successfully generated {target_bars} dollar bars.")
                return bars_df.tail(target_bars)

            curr = self._get_previous_month(curr)

        raise ValueError(f"Could not generate {target_bars} dollar bars up to {ends_at}. Ran out of historical data.")