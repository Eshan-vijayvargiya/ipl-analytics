"""Download the raw IPL 2008-2024 files into data/ (only needed to re-run the notebook from scratch).

    python scripts/get_data.py

The app does NOT need this: it ships with the cleaned tables in app/data/.
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/avinashyadav16/ipl-analytics/main/"
FILES = ["matches_2008-2024.csv", "deliveries_2008-2024.csv"]
DATA = Path(__file__).resolve().parent.parent / "data"

DATA.mkdir(exist_ok=True)
for name in FILES:
    target = DATA / name
    if target.exists():
        print(f"{name}: already there")
        continue
    print(f"Downloading {name} ...")
    urlretrieve(BASE + name, target)
    print(f"  saved {target.stat().st_size / 1e6:.1f} MB")
print("Done. Now open notebooks/ipl_analysis.ipynb")
