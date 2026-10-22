import pandas as pd
import numpy as np


class DynamicDollarBarBuilder:
    """
    Uses raw agg trades, accepts names of the columns in the dataset.
    """
    def __init__(
            self, 
            price: str, 
            quantity: str, 
            first_trade_id: str, 
            last_trade_id: str,
            timestamp: str, 
            is_buyer_maker: str,
            initial_threshold: float,
            target_bars_per_day: float = 20.0,
            span: float = 30.0):
        self.price = price
        self.quantity = quantity
        self.first_trade_id = first_trade_id
        self.last_trade_id = last_trade_id
        self.timestamp = timestamp
        self.is_buyer_maker = is_buyer_maker

        # State tracking for daily volume and thresholds
        self.target_bars_per_day = target_bars_per_day
        self.span = span
        self.expected_daily_dollar_volume = initial_threshold * target_bars_per_day 
        self.current_threshold = initial_threshold
        self.current_day = None
        self.current_day_dollar_volume = 0.0

        self.reset_bar_state()


    def reset_bar_state(self):
        """Resets the accumulator state for a new bar."""
        self.bar_dollar_value = 0.0
        self.bar_prices = []
        self.bar_quantities = []
        self.bar_timestamps = []
        self.bar_first_ids = []
        self.bar_last_ids = []
        self.bar_buyer_makers = []


    def _build_bar(self) -> dict:
        """Constructs and returns metrics for the completed dollar bar."""
        q_arr = np.array(self.bar_quantities)
        bm_arr = np.array(self.bar_buyer_makers)
        f_arr = np.array(self.bar_first_ids)
        l_arr = np.array(self.bar_last_ids)

        volume = np.sum(q_arr)
        buy_volume = np.sum(q_arr[~bm_arr])
        sell_volume = np.sum(q_arr[bm_arr])

        trade_counts = (l_arr - f_arr) + 1
        true_trade_count = np.sum(trade_counts)
        buy_trade_count = np.sum(trade_counts[~bm_arr])
        sell_trade_count = np.sum(trade_counts[bm_arr])
        
        return {
            "timestamp": self.bar_timestamps[-1],
            "duration": self.bar_timestamps[-1] - self.bar_timestamps[0],
            "open": self.bar_prices[0],
            "high": max(self.bar_prices),
            "low": min(self.bar_prices),
            "close": self.bar_prices[-1],
            "dollar_volume": self.bar_dollar_value,
            "volume": volume,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "trade_count": true_trade_count,
            "buy_trade_count": int(buy_trade_count),
            "sell_trade_count": int(sell_trade_count)
        }


    def _get_trade_day(self, t):
        """Safely extracts the calendar date from integer timestamps or datetime objects."""
        if isinstance(t, (int, np.integer, float)):
            unit = 'ms' if t > 1e10 else 's'
            return pd.to_datetime(t, unit=unit).date()
        return pd.to_datetime(t).date()
    

    def _update_daily_threshold(self):
        """Updates the expected daily dollar volume via EWMA and recalculates the threshold."""
        alpha = 2.0 / (self.span + 1.0)
        self.expected_daily_dollar_volume = (
            alpha * self.current_day_dollar_volume + (1.0 - alpha) * self.expected_daily_dollar_volume
        )
        self.current_threshold = self.expected_daily_dollar_volume / self.target_bars_per_day


    def _handle_day_transition(self, trade_day, dollar_val):
        """Handles day rollovers, triggers threshold updates, and resets daily volume tracking."""
        if self.current_day is None:
            self.current_day = trade_day
            self.current_day_dollar_volume = dollar_val
        elif trade_day != self.current_day:
            self._update_daily_threshold()
            self.current_day = trade_day
            self.current_day_dollar_volume = dollar_val
        else:
            self.current_day_dollar_volume += dollar_val


    def update(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a single batch/chunk of data, updates the internal state,
        and returns any completed dollar bars found in this chunk.

        Treshold can be static or dynamic, it must be a series with the desired treshold for every single timestamp,
        bars will try to use the closest treshold found in the timestamp 
        """
        prices = df[self.price].to_numpy()
        quantities = df[self.quantity].to_numpy()
        timestamps = df[self.timestamp].to_numpy()
        first_ids = df[self.first_trade_id].to_numpy()
        last_ids = df[self.last_trade_id].to_numpy()
        buyer_makers = df[self.is_buyer_maker].to_numpy()
        
        bars = []
        
        for p, q, t, f_id, l_id, b_maker in zip(prices, quantities, timestamps, first_ids, last_ids, buyer_makers):
            dollar_val = p * q

            trade_day = self._get_trade_day(t)
            self._handle_day_transition(trade_day, dollar_val)
            
            self.bar_dollar_value += dollar_val
            self.bar_prices.append(p)
            self.bar_quantities.append(q)
            self.bar_timestamps.append(t)
            self.bar_first_ids.append(f_id)
            self.bar_last_ids.append(l_id)
            self.bar_buyer_makers.append(b_maker)
            
            if self.bar_dollar_value >= self.current_threshold:
                bars.append(self._build_bar())
                self.reset_bar_state()
                
        return pd.DataFrame(bars)