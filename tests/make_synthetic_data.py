"""Write FAKE GHCN-Daily files into data/raw/ to test the pipeline offline.

The files use NOAA's exact fixed-width formats, so download_data.py finds them
already "cached" and the rest of the pipeline runs unchanged. Never use this
for the real report - run_all.py --refresh replaces them with real data.

Usage:  python tests/make_synthetic_data.py --trend -0.25   (25% less snow by 2020)
        python tests/make_synthetic_data.py --trend 0       (no change)
"""

import argparse
import calendar
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import RAW_DIR  # noqa: E402

STATES = ["CT", "MA", "ME", "NH", "RI", "VT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trend", type=float, default=-0.25, help="fractional change in snowfall 1941->2020")
    ap.add_argument("--stations", type=int, default=18)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    st_lines, inv_lines = [], []
    # a decoy station outside the region and one with a short record
    specs = [(f"USC00{190000 + i:06d}"[:11], STATES[i % 6], 1931, 2025) for i in range(args.stations)]
    specs.append(("USC00369999", "PA", 1931, 2025))
    specs.append(("USC00279999", "NH", 1975, 2025))

    for sid, state, y0, y1 in specs:
        lat, lon = 41 + rng.uniform(0, 4), -73 + rng.uniform(0, 4)
        elev = rng.uniform(5, 500)
        name = f"TESTVILLE {sid[-3:]}"
        st_lines.append(f"{sid:11} {lat:8.4f} {lon:9.4f} {elev:6.1f} {state:2} {name:30} {'':3} {'HCN':3} {'':5}")
        for el in ["PRCP", "SNOW", "SNWD"]:
            inv_lines.append(f"{sid:11} {lat:8.4f} {lon:9.4f} {el:4} {y0:4d} {y1:4d}")

        base_rate = rng.uniform(0.6, 1.6)  # station snowiness
        lines = []
        depth = 0.0
        for year in range(y0, y1 + 1):
            # linear change centered so 1941 = 1, 2020 = 1 + trend
            factor = 1 + args.trend * (year - 1941) / (2020 - 1941)
            year_noise = rng.lognormal(0, 0.35)
            for month in range(1, 13):
                ndays = calendar.monthrange(year, month)[1]
                snow_vals, depth_vals = [], []
                winter = {11: 0.15, 12: 0.25, 1: 0.3, 2: 0.28, 3: 0.2, 4: 0.06, 10: 0.03}.get(month, 0)
                for d in range(31):
                    if d >= ndays:
                        snow_vals.append("-9999   ")
                        depth_vals.append("-9999   ")
                        continue
                    fall = 0
                    if rng.random() < winter * base_rate:
                        fall = int(rng.exponential(60) * factor * year_noise)
                    depth = max(0.0, depth * (0.93 if month in (12, 1, 2) else 0.75) + fall * 0.8 - (6 if month in (3, 4) else 1))
                    if month in range(5, 10):
                        depth = 0.0
                    missing = rng.random() < 0.02
                    qflag = "I" if rng.random() < 0.002 else " "
                    snow_vals.append("-9999   " if missing else f"{fall:5d} {qflag}7")
                    depth_vals.append("-9999   " if missing else f"{int(depth):5d}  7")
                lines.append(f"{sid}{year:4d}{month:02d}SNOW" + "".join(snow_vals))
                lines.append(f"{sid}{year:4d}{month:02d}SNWD" + "".join(depth_vals))
        (RAW_DIR / f"{sid}.dly").write_text("\n".join(lines) + "\n")

    (RAW_DIR / "ghcnd-stations.txt").write_text("\n".join(st_lines) + "\n")
    (RAW_DIR / "ghcnd-inventory.txt").write_text("\n".join(inv_lines) + "\n")
    print(f"Wrote synthetic data for {len(specs)} stations to {RAW_DIR}")


if __name__ == "__main__":
    main()
