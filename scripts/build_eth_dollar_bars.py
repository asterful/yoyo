"""Build ETHUSDT dollar bars from the monthly raw aggTrades."""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from yoyo.core.dollar_bars import DollarBarBuilder
from yoyo.core.dynamic_dollar_bars import DynamicDollarBarBuilder

INPUT_DIR = ROOT_DIR / "data" / "processed_aggTrades"
OUTPUT_PATH = ROOT_DIR / "data" / "processed_dollarBars" / "ETHUSDT-dyn-dollarBars-2020-01_2026-09.parquet"
INITIAL_THRESHOLD = 100_000_000
TARGET_BARS_PER_DAY = 20.0
SPAN = 30.0
BATCH_SIZE = 1_000_000
COLUMNS = [
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


def main() -> None:
    files = [
        INPUT_DIR / f"ETHUSDT-aggTrades-{month:%Y-%m}.parquet"
        for month in pd.date_range("2020-01-01", "2026-09-01", freq="MS")
    ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    builder = DynamicDollarBarBuilder(
        price="price",
        quantity="quantity",
        first_trade_id="first_trade_id",
        last_trade_id="last_trade_id",
        timestamp="transact_time",
        is_buyer_maker="is_buyer_maker",
        initial_threshold=INITIAL_THRESHOLD,
        target_bars_per_day=TARGET_BARS_PER_DAY,
        span=SPAN,
    )
    writer = None
    trades_processed = 0
    bars_written = 0

    try:
        for path in tqdm(files, desc="Months", unit="month"):
            parquet_file = pq.ParquetFile(path)
            try:
                with tqdm(
                    total=parquet_file.metadata.num_rows,
                    desc=path.name,
                    unit="trades",
                    unit_scale=True,
                    leave=False,
                ) as progress:
                    for batch in parquet_file.iter_batches(
                        batch_size=BATCH_SIZE,
                        columns=COLUMNS,
                    ):
                        trades = batch.to_pandas()
                        bars = builder.update(trades)
                        if not bars.empty:
                            table = pa.Table.from_pandas(bars, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    OUTPUT_PATH,
                                    table.schema,
                                    compression="snappy",
                                )
                            writer.write_table(table)
                            bars_written += len(bars)
                            del table

                        trades_processed += batch.num_rows
                        progress.update(batch.num_rows)
                        del batch, trades, bars
            finally:
                parquet_file.close()
                del parquet_file
    finally:
        if writer is not None:
            writer.close()

    print(
        f"Processed {trades_processed:,} aggTrades; wrote "
        f"{bars_written:,} dollar bars to {OUTPUT_PATH}."
    )
    if builder.bar_prices:
        print(
            f"Final incomplete bar not written: "
            f"${builder.bar_dollar_value:,.2f} across {len(builder.bar_prices):,} aggTrades."
        )


if __name__ == "__main__":
    main()
