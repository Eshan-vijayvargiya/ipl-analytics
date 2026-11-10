"""Build volleyball tables and model from the volleystat R package data (CC0, github.com/bozhinvi/volleystat).

    pip install pyreadr
    python scripts/build_volleyball.py <volleystat/data folder>
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import rally
import volleyball as vb

raw = Path(sys.argv[1])
L = lambda f: list(pyreadr.read_r(str(raw / f"{f}.rda")).values())[0]
games, sets, stats = vb.clean(L("matches"), L("sets"), L("matchstats"), L("players"))
out = ROOT / "app" / "data" / "volleyball"
out.mkdir(parents=True, exist_ok=True)
games.to_parquet(out / "matches.parquet", compression="zstd", index=False)
sets.to_parquet(out / "sets.parquet", compression="zstd", index=False)
stats.to_parquet(out / "player_stats.parquet", compression="zstd", index=False)
print(f"{len(games):,} matches ({games.complete.sum():,} with every set score), {len(sets):,} sets, {len(stats):,} player-matches")

train, test = games[games.season < "2018/19"], games[games.season == "2018/19"]
coef = vb.fit_rally(train)
print("rally link (a, b per 100 Elo):", [round(c, 4) for c in coef])


def evaluate(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    from sklearn.metrics import roc_auc_score
    return {"Log loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "Brier": float(np.mean((p - y) ** 2)),
            "Accuracy": float(np.mean((p > 0.5) == y)), "AUC": float(roc_auc_score(y, p))}


y = test.home_won.to_numpy(float)
base = np.full(len(test), train.home_won.mean())
elo = 1 / (1 + 10 ** (-(test.elo_home + 40 - test.elo_away) / 400))
mk = np.array([vb.match_prob(coef, a, b) for a, b in zip(test.elo_home, test.elo_away)])
res = {"Home team wins ({:.0%})".format(train.home_won.mean()): evaluate(base, y),
       "Elo expected score": evaluate(elo, y), "Rally Markov model (final)": evaluate(mk, y)}
for k, v in res.items():
    print(f"{k:28s}", {a: round(b, 3) for a, b in v.items()})

# final score distribution: predicted vs real on the test season
pred = {}
for a, b in zip(test.elo_home, test.elo_away):
    for k, v in rally.set_score_chances(vb.rally_p(coef, a, b), "volleyball").items():
        pred[k] = pred.get(k, 0) + v
scores = {k: {"predicted": pred[k] / len(test), "real": float((test.result == k).mean())} for k in ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]}
print({k: (round(v["predicted"], 3), round(v["real"], 3)) for k, v in scores.items()})

# how strongly to trust the pre-match estimate once sets are played (fit on training seasons)
tr = train[train.complete]
best, best_ll = None, 1e9
for strength in (20, 60, 150, 400, 800, 1500, 3000):
    ll = []
    for gm in tr.sample(min(1500, len(tr)), random_state=0).itertuples():
        r = vb.replay(coef, gm, sets, strength)
        p = np.clip(r.home_win.iloc[1:-1].to_numpy(), 1e-9, 1 - 1e-9)        # after set 1 .. before the last set
        yy = float(gm.home_won)
        ll.extend(-(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
    if np.mean(ll) < best_ll:
        best, best_ll = strength, float(np.mean(ll))
print("pre-match estimate worth", best, "rallies")
after = {}
te = test[test.complete]
for k in (1, 2, 3):
    P, Y = [], []
    for gm in te.itertuples():
        r = vb.replay(coef, gm, sets, best)
        if len(r) > k + 1:
            P.append(r.home_win.iloc[k]); Y.append(float(gm.home_won))
    after[f"after set {k}"] = {"matches": len(P), **evaluate(np.array(P), np.array(Y))}
    print(f"after set {k}", {a: round(b, 3) for a, b in after[f'after set {k}'].items()})

final = vb.fit_rally(games)
joblib.dump({"coef": final, "strength": best}, ROOT / "models" / "volleyball_model.joblib")
(ROOT / "models" / "volleyball_info.json").write_text(json.dumps({
    "model": "Rally win chance from Elo (logistic) + Markov chain for sets and match", "train": "2013/14-2017/18",
    "test": "2018/19", "test_matches": int(len(test)), "coef_train": coef, "coef_final": final, "results": res,
    "final_scores": scores, "strength": best, "after_sets": after}, indent=1))
