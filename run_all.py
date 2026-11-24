"""Run the whole pipeline: download -> analyze -> build the report.

Usage:  python run_all.py            (uses cached downloads in data/raw/)
        python run_all.py --refresh  (re-download from NOAA)
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} ===", flush=True)
    result = subprocess.run([sys.executable, str(SCRIPTS / script), *args])
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    refresh = ["--refresh"] if "--refresh" in sys.argv else []
    run("download_data.py", *refresh)
    run("analyze.py")
    run("build_report.py")
    print("\nDone. Open site/index.html to view the report.")
