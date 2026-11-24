"""Shared settings, paths and GHCN-Daily file parsers.

Everything that defines the question lives here, so the choices are easy to
find and easy to change:

* REGION      - the six New England states
* BASELINE    - winters 1941-2000 (the last six decades of the 20th century)
* RECENT      - winters 2001-2020 (the first two decades of the 21st century)

A "winter" (season) runs July 1 to June 30 and is labeled by the year its
January falls in, so the 1999-2000 winter is season 2000 and the 2000-01
winter is season 2001.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths (all of data/, output/ and site/ are git-ignored)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
SITE_DIR = ROOT / "site"
TEMPLATE = ROOT / "report" / "template.html"

# ----------------------------------------------------------------------------
# Data source
# ----------------------------------------------------------------------------
NOAA_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
STATIONS_URL = f"{NOAA_BASE}/ghcnd-stations.txt"
INVENTORY_URL = f"{NOAA_BASE}/ghcnd-inventory.txt"
DLY_URL = NOAA_BASE + "/all/{station_id}.dly"

# ----------------------------------------------------------------------------
# The question
# ----------------------------------------------------------------------------
REGION = ["CT", "MA", "ME", "NH", "RI", "VT"]
BASELINE = (1941, 2000)  # seasons, inclusive
RECENT = (2001, 2020)  # seasons, inclusive

# Candidate stations must have SNOW records starting by this year and running
# to at least this year (per the NOAA inventory). The completeness rules below
# then decide which stations are actually used.
CANDIDATE_FIRST_YEAR = 1945
CANDIDATE_LAST_YEAR = 2020

# ----------------------------------------------------------------------------
# Data-quality rules
# ----------------------------------------------------------------------------
# A season counts only if every core snow month has at least this many days
# with a valid (non-missing, not quality-flagged) observation.
CORE_MONTHS = [11, 12, 1, 2, 3]  # Nov-Mar
MIN_VALID_DAYS_PER_MONTH = 25
# A station is used for a metric only if at least this fraction of seasons in
# EACH period are complete.
MIN_SEASON_FRACTION = 0.75

# 1 inch = 25.4 mm. GHCN stores snowfall and snow depth in whole mm, and US
# observers measure in inches, so a 1.0-inch report is stored as 25 mm.
ONE_INCH_MM = 25
MM_PER_INCH = 25.4

# Headline rule, fixed before looking at results: answer "Yes" only if the
# regional average seasonal snowfall for 2001-2020 is lower than for 1941-2000
# AND a one-sided permutation test gives p < ALPHA.
ALPHA = 0.05
N_PERMUTATIONS = 20000
N_BOOTSTRAP = 20000
RANDOM_SEED = 20240101

# Metrics computed for every station-season.
METRICS = {
    "snowfall_in": {
        "element": "SNOW",
        "label": "Seasonal snowfall",
        "unit": "inches",
        "short": "snowfall",
    },
    "big_snow_days": {
        "element": "SNOW",
        "label": "Days with 1 inch or more of new snow",
        "unit": "days",
        "short": "snowy days",
    },
    "snow_cover_days": {
        "element": "SNWD",
        "label": "Days with 1 inch or more of snow on the ground (Nov-Mar)",
        "unit": "days",
        "short": "snow-cover days",
    },
}
PRIMARY_METRIC = "snowfall_in"


def season_of(year: pd.Series | np.ndarray, month: pd.Series | np.ndarray):
    """Season label: the year of the January in that July-June winter."""
    return np.asarray(year) + (np.asarray(month) >= 7)


def in_period(season, period):
    lo, hi = period
    return (season >= lo) & (season <= hi)


# ----------------------------------------------------------------------------
# Parsers for NOAA's fixed-width formats (documented in ghcnd readme.txt)
# ----------------------------------------------------------------------------
def read_stations(path: Path) -> pd.DataFrame:
    """ghcnd-stations.txt -> id, lat, lon, elev_m, state, name, hcn."""
    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (76, 79)]
    names = ["id", "lat", "lon", "elev_m", "state", "name", "hcn"]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str)
    for col in ["lat", "lon", "elev_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["state"] = df["state"].fillna("").str.strip()
    df["name"] = df["name"].fillna("").str.strip()
    df["hcn"] = df["hcn"].fillna("").str.strip()
    return df


def read_inventory(path: Path) -> pd.DataFrame:
    """ghcnd-inventory.txt -> id, element, first_year, last_year."""
    # The full inventory is ~800k lines; only snow elements are needed.
    rows = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            element = line[31:35]
            if element in ("SNOW", "SNWD"):
                rows.append((line[0:11], element, line[36:40], line[41:45]))
    df = pd.DataFrame(rows, columns=["id", "element", "first_year", "last_year"])
    df["first_year"] = pd.to_numeric(df["first_year"], errors="coerce")
    df["last_year"] = pd.to_numeric(df["last_year"], errors="coerce")
    return df


def read_dly(path: Path, elements=("SNOW", "SNWD")) -> pd.DataFrame:
    """Parse a GHCN-Daily .dly file into tidy daily rows.

    Returns columns: date, element, value (mm). Missing values (-9999) and any
    value with a quality flag (failed one of NOAA's QC checks) are dropped, so
    every returned row is a usable observation.
    """
    wanted = set(elements)
    years, months, days, elems, values = [], [], [], [], []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        for line in fh:
            element = line[17:21]
            if element not in wanted:
                continue
            year = int(line[11:15])
            month = int(line[15:17])
            for d in range(31):
                start = 21 + d * 8
                raw = line[start : start + 5]
                if len(raw) < 5:
                    break
                value = int(raw)
                if value == -9999:
                    continue
                qflag = line[start + 6 : start + 7]
                if qflag.strip():
                    continue
                years.append(year)
                months.append(month)
                days.append(d + 1)
                elems.append(element)
                values.append(value)
    df = pd.DataFrame(
        {"year": years, "month": months, "day": days, "element": elems, "value": values}
    )
    if df.empty:
        return pd.DataFrame(columns=["date", "element", "value"])
    df["date"] = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["value"] >= 0]
    return df[["date", "element", "value"]].reset_index(drop=True)
