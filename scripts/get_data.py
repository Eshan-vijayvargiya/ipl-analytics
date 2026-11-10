"""Download every raw dataset into data/raw/ (only needed to rebuild the tables and retrain the models).

    python scripts/get_data.py
"""
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
GH = "https://raw.githubusercontent.com/"


def get(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"  {target.name}: already there")
        return
    print(f"  downloading {target.name} ...")
    urlretrieve(url, target)


def clone(repo, folder, sparse=None):
    if folder.exists():
        print(f"  {folder.name}: already there")
        return
    args = ["git", "clone", "--depth", "1", f"https://github.com/{repo}", str(folder)]
    if sparse:
        args[3:3] = ["--filter=blob:none", "--sparse"]
    subprocess.run(args, check=True)
    if sparse:
        subprocess.run(["git", "-C", str(folder), "sparse-checkout", "set", *sparse], check=True)


print("Cricket (IPL)")
for f in ("matches_2008-2024.csv", "deliveries_2008-2024.csv"):
    get(GH + "avinashyadav16/ipl-analytics/main/" + f, RAW.parent / f)
print("Football")
for f in ("results.csv", "goalscorers.csv", "shootouts.csv"):
    get(GH + "martj42/international_results/master/" + f, RAW / "football" / f)
print("Basketball (NBA)")
for kind in ("cdnnba", "nbastats"):
    for year in (2021, 2022, 2023):
        tar = RAW / "nba" / f"{kind}_{year}.tar.xz"
        get(GH + f"shufinskiy/nba_data/main/datasets/{kind}_{year}.tar.xz", tar)
        if kind == "cdnnba" and not (RAW / "nba" / f"cdnnba_{year}.csv").exists():
            with tarfile.open(tar) as t:
                t.extractall(RAW / "nba", filter="data")
print("Badminton (ShuttleSet)")
clone("wywyWang/CoachAI-Projects", RAW / "CoachAI-Projects", sparse=["ShuttleSet", "CoachAI-Challenge-IJCAI2023/ShuttleSet22"])
print("Volleyball (volleystat)")
clone("bozhinvi/volleystat", RAW / "volleystat")
print("Done. Now run the build scripts listed in data/README.md")
