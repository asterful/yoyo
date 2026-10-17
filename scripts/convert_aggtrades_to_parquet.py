from pathlib import Path
from zipfile import ZipFile
import csv
import sys

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as error:
    raise SystemExit(
        "This script requires pandas and pyarrow. Install them with: "
        "python -m pip install pandas pyarrow"
    ) from error


ROOT_DIR = Path(__file__).resolve().parent
INPUT_DIR = ROOT_DIR / "data" / "monthly_agg_trades"
OUTPUT_DIR = ROOT_DIR / "data" / "processed_aggTrades"
CHUNK_ROWS = 250_000
BAR_WIDTH = 30
MIB = 1024 * 1024

COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]
COLUMN_DTYPES = {
    "agg_trade_id": "int64",
    "price": "float64",
    "quantity": "float64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "transact_time": "int64",
    "is_buyer_maker": "bool",
}


def remove_partial(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_progress(filename, bytes_read, total_bytes, rows):
    fraction = min(bytes_read / total_bytes, 1.0) if total_bytes else 1.0
    filled = int(BAR_WIDTH * fraction)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    progress = "{:5.1f}% {:.1f}/{:.1f} MiB | {:,} rows".format(
        fraction * 100,
        bytes_read / MIB,
        total_bytes / MIB,
        rows,
    )
    sys.stdout.write("\r  {} [{}] {}".format(filename, bar, progress))
    sys.stdout.flush()


def convert_archive(zip_path, parquet_path):
    temporary_path = parquet_path.with_name(parquet_path.name + ".tmp")
    remove_partial(temporary_path)
    progress_started = False

    try:
        with ZipFile(zip_path) as archive:
            csv_members = [
                member for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".csv")
            ]
            if len(csv_members) != 1:
                raise ValueError(
                    "expected one CSV in {}, found {}".format(
                        zip_path.name, len(csv_members)
                    )
                )

            csv_member = csv_members[0]
            # Stream the ZIP member so the expanded CSV is never written to disk.
            with archive.open(csv_member) as csv_stream:
                first_row = next(
                    csv.reader([csv_stream.readline().decode("utf-8-sig")]),
                    [],
                )
                has_header = [value.strip() for value in first_row] == COLUMNS
                csv_stream.seek(0)
                with pd.read_csv(
                    csv_stream,
                    header=0 if has_header else None,
                    names=COLUMNS,
                    dtype=COLUMN_DTYPES,
                    chunksize=CHUNK_ROWS,
                ) as chunks:
                    first_chunk = next(chunks, None)
                    if first_chunk is None:
                        raise ValueError("{} contains an empty CSV".format(zip_path.name))

                    first_table = pa.Table.from_pandas(first_chunk, preserve_index=False)
                    rows_written = len(first_chunk)
                    del first_chunk
                    with pq.ParquetWriter(
                        temporary_path,
                        first_table.schema,
                        compression="snappy",
                    ) as writer:
                        writer.write_table(first_table, row_group_size=CHUNK_ROWS)
                        del first_table
                        write_progress(
                            csv_member.filename,
                            csv_stream.tell(),
                            csv_member.file_size,
                            rows_written,
                        )
                        progress_started = True

                        for chunk in chunks:
                            table = pa.Table.from_pandas(chunk, preserve_index=False)
                            writer.write_table(table, row_group_size=CHUNK_ROWS)
                            rows_written += len(chunk)
                            write_progress(
                                csv_member.filename,
                                csv_stream.tell(),
                                csv_member.file_size,
                                rows_written,
                            )

                        write_progress(
                            csv_member.filename,
                            csv_member.file_size,
                            csv_member.file_size,
                            rows_written,
                        )

        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise ValueError("Parquet output was not created")

        temporary_path.replace(parquet_path)
        print()
        return rows_written
    except Exception:
        if progress_started:
            print()
        remove_partial(temporary_path)
        raise


def main():
    if not INPUT_DIR.is_dir():
        print("Input folder does not exist: {}".format(INPUT_DIR))
        return 1

    archives = sorted(INPUT_DIR.glob("ETHUSDT-aggTrades-*.zip"))
    if not archives:
        print("No monthly aggTrades ZIP files found in {}".format(INPUT_DIR))
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    failed = 0

    for index, zip_path in enumerate(archives, start=1):
        parquet_path = OUTPUT_DIR / "{}.parquet".format(zip_path.stem)
        if parquet_path.is_file() and parquet_path.stat().st_size > 0:
            print("[{}/{}] Skipping {} (already converted)".format(
                index, len(archives), parquet_path.name
            ))
            skipped += 1
            continue

        print("[{}/{}] Converting {}".format(index, len(archives), zip_path.name))
        try:
            rows_written = convert_archive(zip_path, parquet_path)
        except Exception as error:
            print("Failed {}: {}".format(zip_path.name, error))
            failed += 1
        else:
            print("Saved {} ({:,} rows)".format(parquet_path.name, rows_written))
            converted += 1

    print(
        "Summary: {converted} converted, {skipped} already present, {failed} failed.".format(
            converted=converted,
            skipped=skipped,
            failed=failed,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
