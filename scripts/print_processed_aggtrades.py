"""Print a small, sequential preview of each processed aggTrades Parquet file."""

from pathlib import Path
import argparse

try:
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit(
        "This script requires pyarrow. Install it with: python -m pip install pyarrow"
    ) from error


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed_aggTrades"
MIB = 1024 * 1024
DEFAULT_PREVIEW_ROWS = 5


def main():
    parser = argparse.ArgumentParser(
        description="Print each monthly aggTrades Parquet schema and a small row preview."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_PREVIEW_ROWS,
        help="number of initial rows to print per file (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")

    if not DATA_DIR.is_dir():
        print("Processed Parquet folder does not exist: {}".format(DATA_DIR))
        return 1

    files = sorted(DATA_DIR.glob("ETHUSDT-aggTrades-*.parquet"))
    if not files:
        print("No processed aggTrades Parquet files found in {}".format(DATA_DIR))
        return 1

    failed = 0
    for index, path in enumerate(files, start=1):
        print("\n[{}/{}] {}".format(index, len(files), path.name))
        parquet_file = None
        try:
            parquet_file = pq.ParquetFile(path)
            metadata = parquet_file.metadata
            print(
                "Size: {:.1f} MiB | Rows: {:,} | Row groups: {}".format(
                    path.stat().st_size / MIB,
                    metadata.num_rows,
                    metadata.num_row_groups,
                )
            )
            print("Schema:")
            for field in parquet_file.schema_arrow:
                print("  {}: {}".format(field.name, field.type))

            batch = next(parquet_file.iter_batches(batch_size=args.rows), None)
            if batch is None:
                print("Preview: (no rows)")
            else:
                print("First {} row(s):".format(min(args.rows, metadata.num_rows)))
                print(batch.to_pandas().to_string(index=False))
        except Exception as error:
            print("Could not read {}: {}".format(path.name, error))
            failed += 1
        finally:
            if parquet_file is not None:
                parquet_file.close()

    print("\nInspected {} file(s); {} read failure(s).".format(len(files), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
