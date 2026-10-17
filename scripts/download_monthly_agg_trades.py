from datetime import date, timedelta
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import argparse
import sys
import zipfile


BASE_URL = "https://data.binance.vision/data/futures/um/monthly/aggTrades/ETHUSDT"
DATA_DIR = Path(__file__).resolve().parent / "data" / "monthly_agg_trades"
EARLIEST_MONTH = date(2020, 1, 1)
CHUNK_SIZE = 1024 * 1024
BAR_WIDTH = 30


def previous_month(month):
    return (month.replace(day=1) - timedelta(days=1)).replace(day=1)


def parse_archive_month(path):
    prefix = "ETHUSDT-aggTrades-"
    if not path.name.startswith(prefix) or not path.name.endswith(".zip"):
        return None

    month_text = path.name[len(prefix):-len(".zip")]
    try:
        year, month = (int(part) for part in month_text.split("-"))
        return date(year, month, 1)
    except (TypeError, ValueError):
        return None


def latest_month_to_check():
    latest_completed_month = previous_month(date.today().replace(day=1))
    local_months = []

    if DATA_DIR.exists():
        for path in DATA_DIR.glob("ETHUSDT-aggTrades-*.zip"):
            month = parse_archive_month(path)
            if month is not None:
                local_months.append(month)

    return max([latest_completed_month] + local_months)


def is_valid_archive(path):
    return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def remove_partial(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def write_progress(filename, downloaded, total_bytes):
    downloaded_mb = downloaded / (1024 * 1024)
    if total_bytes:
        fraction = min(downloaded / total_bytes, 1.0)
        filled = int(BAR_WIDTH * fraction)
        bar = "#" * filled + "-" * (BAR_WIDTH - filled)
        progress = "{:5.1f}% {:.1f}/{:.1f} MB".format(
            fraction * 100,
            downloaded_mb,
            total_bytes / (1024 * 1024),
        )
    else:
        cycle = 2 * (BAR_WIDTH - 1)
        position = (downloaded // CHUNK_SIZE) % cycle
        if position >= BAR_WIDTH:
            position = cycle - position
        bar = "-" * position + ">" + "-" * (BAR_WIDTH - position - 1)
        progress = "{:.1f} MB".format(downloaded_mb)

    sys.stdout.write("\r{} [{}] {}".format(filename, bar, progress))
    sys.stdout.flush()


def download_month(month, dry_run=False):
    filename = "ETHUSDT-aggTrades-{:%Y-%m}.zip".format(month)
    destination = DATA_DIR / filename

    if is_valid_archive(destination):
        print("Skipping {} (already downloaded)".format(filename))
        return "skipped"

    url = "{}/{}".format(BASE_URL, filename)
    if dry_run:
        print("Would download {}".format(url))
        return "planned"

    print("Downloading {}".format(filename), flush=True)
    temporary_file = destination.with_name(destination.name + ".part")
    progress_started = False
    try:
        with urlopen(url, timeout=60) as response:
            try:
                total_bytes = int(response.headers.get("Content-Length"))
            except (TypeError, ValueError):
                total_bytes = None

            downloaded = 0
            with temporary_file.open("wb") as output:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)
                    write_progress(filename, downloaded, total_bytes)
                    progress_started = True

        if progress_started:
            write_progress(filename, downloaded, total_bytes or downloaded)
            print()

        if not zipfile.is_zipfile(temporary_file):
            raise ValueError("download did not produce a valid ZIP archive")

        temporary_file.replace(destination)
        print("Downloaded {} ({:.1f} MB)".format(
            filename, destination.stat().st_size / (1024 * 1024)
        ))
        return "downloaded"
    except HTTPError as error:
        if progress_started:
            print()
        remove_partial(temporary_file)
        if error.code == 404:
            print("Not found: {} (HTTP 404)".format(filename))
            return "unavailable"
        print("Failed {} (HTTP {}): {}".format(filename, error.code, error.reason))
        return "failed"
    except (URLError, HTTPException, OSError, ValueError) as error:
        if progress_started:
            print()
        remove_partial(temporary_file)
        print("Failed {}: {}".format(filename, error))
        return "failed"


def main():
    parser = argparse.ArgumentParser(
        description="Download missing ETHUSDT monthly aggTrades ZIP files through 2020-01."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list missing months without downloading them",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    latest_month = latest_month_to_check()
    if latest_month < EARLIEST_MONTH:
        print("No months to download; the latest month is before 2020-01.")
        return 0

    counts = {"downloaded": 0, "skipped": 0, "planned": 0, "unavailable": 0, "failed": 0}
    month = latest_month
    while month >= EARLIEST_MONTH:
        result = download_month(month, dry_run=args.dry_run)
        counts[result] += 1
        month = previous_month(month)

    print(
        "Summary: {downloaded} downloaded, {skipped} already present, "
        "{planned} planned, {unavailable} unavailable, {failed} failed.".format(**counts)
    )
    return 1 if counts["unavailable"] or counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
