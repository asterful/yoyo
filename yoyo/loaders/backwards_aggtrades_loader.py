import hashlib
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from yoyo.loaders.base import BackwardsLoader


PROCESSED_AGGTRADES_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "processed_aggTrades"
)
BATCH_SIZE = 250_000
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "dollar_volume"]


class _DollarBarAccumulator:
    def __init__(self, dollar_threshold: float, max_bars: int):
        self.dollar_threshold = dollar_threshold
        self.bars = deque(maxlen=max_bars)
        self.last_timestamp_ms = None
        self._reset_current_bar()

    def _reset_current_bar(self):
        self.has_current_bar = False
        self.current_open = None
        self.current_high = -np.inf
        self.current_low = np.inf
        self.current_volume = 0.0
        self.current_dollar_volume = 0.0

    def add_batch(self, prices, quantities, timestamps_ms):
        prices = np.asarray(prices, dtype=np.float64)
        quantities = np.asarray(quantities, dtype=np.float64)
        timestamps_ms = np.asarray(timestamps_ms, dtype=np.int64)

        if not len(prices):
            return
        if len(prices) != len(quantities) or len(prices) != len(timestamps_ms):
            raise ValueError("aggTrades price, quantity, and timestamp lengths differ")
        if (
            not np.all(np.isfinite(prices))
            or not np.all(np.isfinite(quantities))
            or np.any(prices < 0)
            or np.any(quantities < 0)
        ):
            raise ValueError("aggTrades contain invalid prices or quantities")
        if len(timestamps_ms) > 1 and np.any(timestamps_ms[1:] < timestamps_ms[:-1]):
            raise ValueError("aggTrades are not ordered by transact_time")
        if (
            self.last_timestamp_ms is not None
            and timestamps_ms[0] < self.last_timestamp_ms
        ):
            raise ValueError("aggTrades are not ordered by transact_time")
        self.last_timestamp_ms = int(timestamps_ms[-1])

        dollar_values = prices * quantities
        if not np.all(np.isfinite(dollar_values)):
            raise ValueError("aggTrades contain a non-finite dollar value")

        cumulative_dollars = np.cumsum(dollar_values)
        cumulative_quantities = np.cumsum(quantities)
        bar_ends = []
        next_threshold = self.dollar_threshold - self.current_dollar_volume

        while True:
            end_index = int(np.searchsorted(cumulative_dollars, next_threshold, side="left"))
            if end_index >= len(prices):
                break
            bar_ends.append(end_index)
            next_threshold = cumulative_dollars[end_index] + self.dollar_threshold

        segment_starts = [0]
        segment_starts.extend(end + 1 for end in bar_ends[:-1])
        tail_start = bar_ends[-1] + 1 if bar_ends else 0
        if tail_start < len(prices) and bar_ends:
            segment_starts.append(tail_start)
        segment_starts = np.asarray(segment_starts, dtype=np.int64)

        segment_highs = np.maximum.reduceat(prices, segment_starts)
        segment_lows = np.minimum.reduceat(prices, segment_starts)

        for segment_index, end_index in enumerate(bar_ends):
            start_index = int(segment_starts[segment_index])
            prior_quantity = cumulative_quantities[start_index - 1] if start_index else 0.0
            prior_dollars = cumulative_dollars[start_index - 1] if start_index else 0.0
            segment_volume = cumulative_quantities[end_index] - prior_quantity
            segment_dollar_volume = cumulative_dollars[end_index] - prior_dollars

            if segment_index == 0 and self.has_current_bar:
                bar_open = self.current_open
                bar_high = max(self.current_high, segment_highs[segment_index])
                bar_low = min(self.current_low, segment_lows[segment_index])
                bar_volume = self.current_volume + segment_volume
                bar_dollar_volume = self.current_dollar_volume + segment_dollar_volume
            else:
                bar_open = prices[start_index]
                bar_high = segment_highs[segment_index]
                bar_low = segment_lows[segment_index]
                bar_volume = segment_volume
                bar_dollar_volume = segment_dollar_volume

            self.bars.append(
                {
                    "timestamp": pd.Timestamp(int(timestamps_ms[end_index]), unit="ms"),
                    "open": float(bar_open),
                    "high": float(bar_high),
                    "low": float(bar_low),
                    "close": float(prices[end_index]),
                    "volume": float(bar_volume),
                    "dollar_volume": float(bar_dollar_volume),
                }
            )
            self._reset_current_bar()

        if tail_start < len(prices):
            tail_segment_index = len(bar_ends)
            prior_quantity = cumulative_quantities[tail_start - 1] if tail_start else 0.0
            prior_dollars = cumulative_dollars[tail_start - 1] if tail_start else 0.0
            tail_volume = cumulative_quantities[-1] - prior_quantity
            tail_dollar_volume = cumulative_dollars[-1] - prior_dollars

            if not self.has_current_bar:
                self.has_current_bar = True
                self.current_open = float(prices[tail_start])
                self.current_high = segment_highs[tail_segment_index]
                self.current_low = segment_lows[tail_segment_index]
            else:
                self.current_high = max(
                    self.current_high, segment_highs[tail_segment_index]
                )
                self.current_low = min(
                    self.current_low, segment_lows[tail_segment_index]
                )

            self.current_volume += tail_volume
            self.current_dollar_volume += tail_dollar_volume

    def to_frame(self):
        if not self.bars:
            return pd.DataFrame(
                columns=BAR_COLUMNS,
                index=pd.DatetimeIndex([], name="timestamp"),
            )
        return pd.DataFrame(self.bars).set_index("timestamp")[BAR_COLUMNS]


class BackwardsAggTradesDollarBarsLoader(BackwardsLoader):
    def __init__(
        self,
        symbol: str = "ETH/USDT",
        exchange_id: str = "binance",
        cache_dir: str = "./cache",
        dollar_threshold: float = 5_000_000,
    ):
        self.symbol = symbol
        self.exchange_id = exchange_id
        self.cache_dir = cache_dir
        self.dollar_threshold = float(dollar_threshold)
        self.data_dir = PROCESSED_AGGTRADES_DIR

        if self.dollar_threshold <= 0 or not np.isfinite(self.dollar_threshold):
            raise ValueError("dollar_threshold must be a finite positive number")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _market_symbol(self) -> str:
        return self.symbol.split(":", 1)[0].replace("/", "").upper()

    def _get_source_files(self, ends_at: pd.Timestamp):
        prefix = "{}-aggTrades-".format(self._market_symbol())
        files = []
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                "Processed aggTrades directory does not exist: {}".format(self.data_dir)
            )

        for path in self.data_dir.glob(prefix + "????-??.parquet"):
            try:
                year, month = map(int, path.stem[len(prefix) :].split("-"))
            except ValueError:
                continue
            if (year, month) <= (ends_at.year, ends_at.month):
                files.append(path)

        return sorted(files)

    def _get_query_cache_path(self, end_ms, target_bars, source_files):
        source_fingerprint = "\n".join(
            "{}:{}:{}".format(path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in source_files
        )
        fingerprint = hashlib.sha256(source_fingerprint.encode("utf-8")).hexdigest()[:16]
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        safe_exchange = self.exchange_id.replace("/", "_").replace(":", "_")
        threshold = format(self.dollar_threshold, ".12g").replace(".", "p")
        end_key = pd.Timestamp(end_ms, unit="ms").strftime("%Y%m%dT%H%M%S%f")
        filename = "{}_{}_aggTrades_{}_{}_{}_{}.parquet".format(
            safe_exchange,
            safe_symbol,
            threshold,
            end_key,
            target_bars,
            fingerprint,
        )
        return Path(self.cache_dir) / "aggtrades_dollar_bars" / filename

    @staticmethod
    def _normalize_timestamp(timestamp):
        timestamp = pd.Timestamp(timestamp)
        if pd.isna(timestamp):
            raise ValueError("ends_at must be a valid timestamp")
        if timestamp.tz is not None:
            timestamp = timestamp.tz_convert("UTC").tz_localize(None)
        return timestamp

    @staticmethod
    def _empty_bars():
        return pd.DataFrame(
            columns=BAR_COLUMNS,
            index=pd.DatetimeIndex([], name="timestamp"),
        )

    def _build_dollar_bars_forward(self, df_aggtrades, ends_at):
        if df_aggtrades.empty:
            return self._empty_bars()

        if "transact_time" in df_aggtrades.columns:
            df = df_aggtrades.sort_values("transact_time", kind="stable")
            timestamps_ms = df["transact_time"].to_numpy(dtype=np.int64)
        elif isinstance(df_aggtrades.index, pd.DatetimeIndex):
            df = df_aggtrades.sort_index()
            index = df.index
            if index.tz is not None:
                index = index.tz_convert("UTC").tz_localize(None)
            timestamps_ms = index.to_numpy(dtype="datetime64[ms]").astype(np.int64)
        else:
            raise ValueError(
                "aggTrades must contain transact_time or have a DatetimeIndex"
            )

        end_timestamp = self._normalize_timestamp(ends_at)
        end_ms = end_timestamp.value // 1_000_000
        included = timestamps_ms <= end_ms
        accumulator = _DollarBarAccumulator(
            self.dollar_threshold,
            max_bars=max(1, int(included.sum())),
        )
        accumulator.add_batch(
            df["price"].to_numpy(dtype=np.float64)[included],
            df["quantity"].to_numpy(dtype=np.float64)[included],
            timestamps_ms[included],
        )
        return accumulator.to_frame()

    def _load_cached_bars(self, cache_path, target_bars):
        if not cache_path.is_file():
            return None
        try:
            bars = pd.read_parquet(cache_path)
            if "timestamp" in bars.columns:
                bars = bars.set_index("timestamp")
            bars.index = pd.to_datetime(bars.index)
            bars.index.name = "timestamp"
            if len(bars) == target_bars and all(
                column in bars.columns for column in BAR_COLUMNS
            ):
                return bars[BAR_COLUMNS]
        except Exception as error:
            print("-> Could not read cached dollar bars: {}".format(error))
        return None

    def _save_cached_bars(self, bars, cache_path):
        temporary_path = cache_path.with_name(cache_path.name + ".tmp")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            bars.to_parquet(temporary_path, index=True)
            temporary_path.replace(cache_path)
        except Exception as error:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            print("-> Could not cache dollar bars: {}".format(error))

    def _build_bars_from_files(self, source_files, end_ms, target_bars):
        accumulator = _DollarBarAccumulator(self.dollar_threshold, target_bars)
        for file_index, path in enumerate(source_files, start=1):
            print(
                "-> Processing aggTrades file {}/{}: {}".format(
                    file_index, len(source_files), path.name
                )
            )
            parquet_file = pq.ParquetFile(path)
            try:
                schema = parquet_file.schema_arrow
                required_columns = {"price", "quantity", "transact_time"}
                missing_columns = required_columns.difference(schema.names)
                if missing_columns:
                    raise ValueError(
                        "{} is missing required columns: {}".format(
                            path.name, ", ".join(sorted(missing_columns))
                        )
                    )
                timestamp_column_index = schema.get_field_index("transact_time")

                for row_group_index in range(parquet_file.metadata.num_row_groups):
                    row_group = parquet_file.metadata.row_group(row_group_index)
                    timestamp_stats = row_group.column(
                        timestamp_column_index
                    ).statistics
                    if (
                        timestamp_stats is not None
                        and timestamp_stats.has_min_max
                        and timestamp_stats.min > end_ms
                    ):
                        continue

                    for batch in parquet_file.iter_batches(
                        batch_size=BATCH_SIZE,
                        row_groups=[row_group_index],
                        columns=["price", "quantity", "transact_time"],
                    ):
                        prices = batch.column(batch.schema.get_field_index("price")).to_numpy(
                            zero_copy_only=False
                        )
                        quantities = batch.column(
                            batch.schema.get_field_index("quantity")
                        ).to_numpy(zero_copy_only=False)
                        timestamps_ms = batch.column(
                            batch.schema.get_field_index("transact_time")
                        ).to_numpy(zero_copy_only=False)
                        if len(timestamps_ms) == 0:
                            continue

                        if timestamps_ms[-1] <= end_ms:
                            accumulator.add_batch(prices, quantities, timestamps_ms)
                        elif timestamps_ms[0] > end_ms:
                            continue
                        else:
                            included = timestamps_ms <= end_ms
                            accumulator.add_batch(
                                prices[included],
                                quantities[included],
                                timestamps_ms[included],
                            )
            finally:
                parquet_file.close()

        return accumulator.to_frame()

    def load_backward(self, ends_at: pd.Timestamp, target_bars: int) -> pd.DataFrame:
        if target_bars < 1:
            raise ValueError("target_bars must be at least 1")

        end_timestamp = self._normalize_timestamp(ends_at)
        end_ms = end_timestamp.value // 1_000_000
        print(
            "-> Loading backward from aggTrades: targeting {} dollar bars ending at {}...".format(
                target_bars, end_timestamp
            )
        )

        source_files = self._get_source_files(end_timestamp)
        if not source_files:
            raise ValueError(
                "No processed aggTrades files found for {} up to {} in {}.".format(
                    self.symbol, end_timestamp, self.data_dir
                )
            )

        cache_path = self._get_query_cache_path(end_ms, target_bars, source_files)
        cached_bars = self._load_cached_bars(cache_path, target_bars)
        if cached_bars is not None:
            print("-> Loaded {} cached dollar bars.".format(target_bars))
            return cached_bars

        months_to_load = 1
        while True:
            selected_files = source_files[-months_to_load:]
            bars = self._build_bars_from_files(selected_files, end_ms, target_bars)
            if len(bars) >= target_bars:
                break
            if months_to_load == len(source_files):
                raise ValueError(
                    "Could not generate {} dollar bars up to {}. Ran out of historical aggTrades data.".format(
                        target_bars, end_timestamp
                    )
                )
            print(
                "-> Generated {} bars from the most recent {} month(s); extending history...".format(
                    len(bars), months_to_load
                )
            )
            months_to_load += 1

        result = bars.tail(target_bars)
        self._save_cached_bars(result, cache_path)
        print("-> Successfully generated {} dollar bars.".format(target_bars))
        return result
