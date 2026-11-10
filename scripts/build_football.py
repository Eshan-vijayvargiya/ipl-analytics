"""Build the football tables and model from the raw files of github.com/martj42/international_results.

    python scripts/build_football.py <folder with results.csv, goalscorers.csv, shootouts.csv>
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import football as fb

raw = Path(sys.argv[1])
m, g = fb.clean(pd.read_csv(raw / "results.csv"), pd.read_csv(raw / "goalscorers.csv"), pd.read_csv(raw / "shootouts.csv"))
out = ROOT / "app" / "data" / "football"
out.mkdir(parents=True, exist_ok=True)
m.to_parquet(out / "matches.parquet", compression="zstd", index=False)
g.to_parquet(out / "goals.parquet", compression="zstd", index=False)
print(f"{len(m):,} matches, {len(g):,} goals with minutes ({m.has_timeline.sum():,} full timelines)")


def logloss3(p, y):
    idx = {"H": 0, "D": 1, "A": 2}
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), [idx[v] for v in y]], 1e-12, 1))))


def rps(p, y):
    idx = {"H": 0, "D": 1, "A": 2}
    o = np.zeros_like(p); o[np.arange(len(y)), [idx[v] for v in y]] = 1
    return float(np.mean(((np.cumsum(p, 1) - np.cumsum(o, 1))[:, :2] ** 2).sum(1) / 2))


train = m[(m.year >= 1990) & (m.year <= 2021)]
test = m[m.year >= 2022]
model = fb.fit_goal_model(train)


def probs(model, d, rho):
    out = []
    for r in d.itertuples():
        lh, la = fb.expected_goals(model, r.elo_home, r.elo_away, r.neutral, r.kind)
        out.append(fb.outcome(fb.score_grid(lh, la, rho=rho)))
    return np.array(out)


# choose the draw correction on the TRAINING seasons only
sample = train.sample(6000, random_state=1)
best_rho = min(np.arange(0, 0.21, 0.02), key=lambda r: logloss3(probs(model, sample, r), sample.result.to_numpy()))
p_test = probs(model, test, best_rho)
y = test.result.to_numpy()
base = train.result.value_counts(normalize=True).reindex(["H", "D", "A"]).to_numpy()
p_base = np.tile(base, (len(test), 1))
# Elo alone: the classic expected score, split into H/D/A with the training draw rate
dr = (test.elo_home - test.elo_away + np.where(test.neutral, 0, 100)).to_numpy()
we = 1 / (10 ** (-dr / 400) + 1)
draw = base[1]
p_elo = np.column_stack([we * (1 - draw), np.full(len(we), draw), (1 - we) * (1 - draw)])
pick = p_test.argmax(1)
res = {
    "Base rates (home win 48%)": {"Log loss": logloss3(p_base, y), "RPS": rps(p_base, y), "Accuracy": float(np.mean(np.array(["H", "D", "A"])[p_base.argmax(1)] == y))},
    "Elo expected score": {"Log loss": logloss3(p_elo, y), "RPS": rps(p_elo, y), "Accuracy": float(np.mean(np.array(["H", "D", "A"])[p_elo.argmax(1)] == y))},
    "Poisson goal model (final)": {"Log loss": logloss3(p_test, y), "RPS": rps(p_test, y), "Accuracy": float(np.mean(np.array(["H", "D", "A"])[pick] == y))},
}
for k, v in res.items():
    print(f"{k:32s}", {a: round(b, 3) for a, b in v.items()})
print("draw correction rho =", round(float(best_rho), 2))

# in-play check on test matches with a full goal timeline
share, extra = fb.goal_time_share(g[g.match_id.isin(train.match_id)])
tl = test[test.has_timeline & test.shootout_winner.isna()]
gt = g[g.match_id.isin(tl.match_id)]
inplay = {}
for minute in (0, 30, 60, 75, 85):
    P = []
    for r in tl.itertuples():
        lh, la = fb.expected_goals(model, r.elo_home, r.elo_away, r.neutral, r.kind)
        so = gt[(gt.match_id == r.match_id) & (gt.minute <= minute)]
        h = int((so.team == r.home_team).sum())
        P.append(fb.in_play(lh, la, minute, h, len(so) - h, share, rho=best_rho))
    P = np.array(P); yy = tl.result.to_numpy()
    inplay[minute] = {"Log loss": logloss3(P, yy), "Accuracy": float(np.mean(np.array(["H", "D", "A"])[P.argmax(1)] == yy))}
    print("minute", minute, {a: round(b, 3) for a, b in inplay[minute].items()}, "matches", len(tl))

# final model: all matches since 1990
final = fb.fit_goal_model(m[m.year >= 1990])
share_all, extra_all = fb.goal_time_share(g)
models = ROOT / "models"
models.mkdir(exist_ok=True)
joblib.dump({"model": final, "rho": float(best_rho), "share": share_all, "extra": float(extra_all)}, models / "football_model.joblib")
info = {"model": "Poisson goal model on Elo gap, home advantage and match type", "train": "1990-2021", "test": "2022-2026",
        "test_matches": int(len(test)), "results": res, "rho": float(best_rho),
        "in_play": {str(k): v for k, v in inplay.items()}, "in_play_matches": int(len(tl)),
        "coef": dict(zip(fb.design(fb.team_rows(m.head(2))).columns, map(float, final.coef_)))}
(models / "football_info.json").write_text(json.dumps(info, indent=1))
