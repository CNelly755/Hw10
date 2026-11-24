"""Download GHCN-Daily station records for long-running New England snow stations.

Steps:
1. Download NOAA's station list and inventory (cached in data/raw/).
2. Pick candidate stations in CT, MA, ME, NH, RI and VT whose snowfall (SNOW)
   record starts by 1945 and runs to at least 2020.
3. Download each candidate's full daily record (.dly file).

Writes data/candidates.csv. Files already downloaded are not fetched again;
pass --refresh to re-download everything.

Usage:  python scripts/download_data.py [--refresh]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.request
from pathlib import Path

from common import (
    CANDIDATE_FIRST_YEAR,
    CANDIDATE_LAST_YEAR,
    DATA_DIR,
    DLY_URL,
    INVENTORY_URL,
    RAW_DIR,
    REGION,
    STATIONS_URL,
    read_inventory,
    read_stations,
)

USER_AGENT = "uncle-snow-analysis/1.0 (educational project)"


def fetch(url: str, dest: Path, refresh: bool = False, tries: int = 4) -> Path:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # network hiccups: retry with backoff
            if attempt == tries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            wait = 5 * attempt
            print(f"  retry {attempt}/{tries - 1} for {url} in {wait}s ({exc})")
            time.sleep(wait)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download cached files")
    args = parser.parse_args()

    print("Downloading station list and inventory...")
    stations_path = fetch(STATIONS_URL, RAW_DIR / "ghcnd-stations.txt", args.refresh)
    inventory_path = fetch(INVENTORY_URL, RAW_DIR / "ghcnd-inventory.txt", args.refresh)

    stations = read_stations(stations_path)
    inventory = read_inventory(inventory_path)

    snow = inventory[inventory["element"] == "SNOW"]
    long_snow = snow[
        (snow["first_year"] <= CANDIDATE_FIRST_YEAR)
        & (snow["last_year"] >= CANDIDATE_LAST_YEAR)
    ]
    candidates = stations[
        stations["id"].str.startswith("US") & stations["state"].isin(REGION)
    ].merge(long_snow[["id", "first_year", "last_year"]], on="id")
    candidates = candidates.sort_values(["state", "name"]).reset_index(drop=True)

    if candidates.empty:
        print("No candidate stations found - check the inventory download.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(DATA_DIR / "candidates.csv", index=False)
    print(
        f"Found {len(candidates)} New England stations with snowfall records "
        f"from <= {CANDIDATE_FIRST_YEAR} to >= {CANDIDATE_LAST_YEAR}."
    )

    failures = []
    for i, row in candidates.iterrows():
        sid = row["id"]
        print(f"  [{i + 1}/{len(candidates)}] {sid} {row['name']}, {row['state']}")
        try:
            fetch(DLY_URL.format(station_id=sid), RAW_DIR / f"{sid}.dly", args.refresh)
        except RuntimeError as exc:
            print(f"    skipped: {exc}")
            failures.append(sid)

    print(f"Done. {len(candidates) - len(failures)} station files in {RAW_DIR}.")
    if failures:
        print(f"Could not download: {', '.join(failures)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
