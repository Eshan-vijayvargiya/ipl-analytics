"""Build the NBA tables and win-probability model from github.com/shufinskiy/nba_data (Apache-2.0).

    python scripts/build_basketball.py <folder with cdnnba_2021.csv ... and nbastats_*.tar.xz>
"""
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import basketball as bb

raw = Path(sys.argv[1])
YEARS = [2021, 2022, 2023]
COLS = ["gameId", "actionNumber", "orderNumber", "clock", "timeActual", "period", "actionType", "subType", "personId",
        "scoreHome", "scoreAway", "teamTricode", "playerNameI", "shotResult", "xLegacy", "yLegacy", "shotDistance",
        "assistPersonId"]

# full player names from the stats.nba.com files
names = {}
for y in YEARS:
    tar = raw / f"nbastats_{y}.tar.xz"
    if tar.exists():
        subprocess.run(["tar", "-xJf", str(tar), "-C", str(raw)], check=True)
        ns = pd.read_csv(raw / f"nbastats_{y}.csv", usecols=["PLAYER1_ID", "PLAYER1_NAME"], low_memory=False).dropna()
        names.update(dict(zip(ns.PLAYER1_ID.astype(int), ns.PLAYER1_NAME)))
        (raw / f"nbastats_{y}.csv").unlink()
print(f"{len(names):,} player names")

G, S, B, SH = [], [], [], []
for y in YEARS:
    pbp = pd.read_csv(raw / f"cdnnba_{y}.csv", usecols=COLS, low_memory=False)
    g, s, b, sh = bb.clean_season(pbp, y, names)
    print(bb.season_label(y), f"{len(g):,} games, {len(s):,} scoring plays, {len(b):,} player-games, {len(sh):,} shots")
    G.append(g); S.append(s); B.append(b); SH.append(sh)
games = bb.add_elo(pd.concat(G, ignore_index=True))
states = pd.concat(S, ignore_index=True)
box = pd.concat(B, ignore_index=True)
shots = pd.concat(SH, ignore_index=True)
out = ROOT / "app" / "data" / "basketball"
out.mkdir(parents=True, exist_ok=True)
games.to_parquet(out / "games.parquet", compression="zstd", index=False)
states.astype({"period": "int8", "home_score": "int16", "away_score": "int16", "t": "float32"}).to_parquet(
    out / "states.parquet", compression="zstd", index=False)
box.to_parquet(out / "box.parquet", compression="zstd", index=False)
shots = shots[shots.season == "2023-24"].dropna(subset=["x", "y"])
shots.astype({"x": "int16", "y": "int16", "distance": "float32", "period": "int8"}).to_parquet(
    out / "shots.parquet", compression="zstd", index=False)

# ------------------------------------------------------------------ model: train 2021-23, test 2023-24
ms = bb.model_states(games, states)
train, test = ms[ms.season != "2023-24"], ms[ms.season == "2023-24"]
params = bb.fit(train)
print("fitted (points per 100 Elo, sigma, home points):", [round(p, 2) for p in params])


def scores(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    from sklearn.metrics import roc_auc_score
    return {"Log loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "Brier": float(np.mean((p - y) ** 2)),
            "AUC": float(roc_auc_score(y, p))}


y = test.home_won.to_numpy(float)
base_params = (0.0, params[1], params[2])                 # same model without team strength
from scipy.stats import norm
naive = norm.cdf(test.margin / (params[1] * np.sqrt(np.maximum(test.f, 1e-4))))
res = {"Score and clock only (no home edge, no team strength)": scores(naive, y),
       "+ home advantage": scores(bb.win_prob(base_params, test.margin, test.f, 0 * test.elo_gap), y),
       "+ Elo team strength (final)": scores(bb.win_prob(params, test.margin, test.f, test.elo_gap), y)}
for k, v in res.items():
    print(f"{k:55s}", {a: round(b, 3) for a, b in v.items()})
# accuracy by stage of the game
stage = {}
for label, lo, hi in (("Tip-off", 0.999, 1.01), ("Half-time", 0.49, 0.51), ("Start of 4th", 0.24, 0.26), ("Last 2 minutes", 0.0, 0.042)):
    part = test[(test.f >= lo) & (test.f <= hi)]
    p = bb.win_prob(params, part.margin, part.f, part.elo_gap)
    stage[label] = {"states": int(len(part)), "Accuracy": float(np.mean((p > 0.5) == part.home_won)), **scores(p, part.home_won.to_numpy(float))}
    print(label, {a: round(b, 3) for a, b in stage[label].items()})

final = bb.fit(ms)
models = ROOT / "models"
joblib.dump({"params": final}, models / "basketball_model.joblib")
(models / "basketball_info.json").write_text(json.dumps({
    "model": "Normal random-walk model: final margin ~ Normal(margin + mu*f, sigma*sqrt(f))",
    "train": "2021-22, 2022-23", "test": "2023-24", "test_states": int(len(test)), "test_games": int(test.game_id.nunique()),
    "params_train": params, "params_final": final, "results": res, "by_stage": stage}, indent=1))
