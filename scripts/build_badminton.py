"""Build badminton tables and model from ShuttleSet + ShuttleSet22 (MIT, github.com/wywyWang/CoachAI-Projects).

    python scripts/build_badminton.py <CoachAI-Projects folder>
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import badminton as bd
import rally

raw = Path(sys.argv[1])
m, rl, st = bd.clean([raw / "CoachAI-Challenge-IJCAI2023" / "ShuttleSet22" / "set", raw / "ShuttleSet" / "set"])
out = ROOT / "app" / "data" / "badminton"
out.mkdir(parents=True, exist_ok=True)
m.to_parquet(out / "matches.parquet", index=False)
rl.to_parquet(out / "rallies.parquet", compression="zstd", index=False)
st.to_parquet(out / "strokes.parquet", compression="zstd", index=False)
print(f"{len(m)} matches, {len(rl):,} rallies, {len(st):,} shots, {m.winner.nunique() + 0} winners; "
      f"{m.year.value_counts().sort_index().to_dict()}; India matches {m.india.sum()}")

train_m, test_m = m[m.year <= 2021], m[m.year == 2022]
print(f"train {len(train_m)} matches, test {len(test_m)}")
ratings, serve, _ = bd.fit_ratings(train_m, rl[rl.match_id.isin(train_m.match_id)])
print("serve effect on rally (logit):", round(serve, 3), " top ratings:", ratings.head(5).round(2).to_dict())


def ll(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return float(-np.mean(np.log(p)))


# pre-match: chance given to the player who actually won
pre = np.array([rally.match(0, 0, bd.rally_p(ratings, r.winner, r.loser), "badminton") for r in test_m.itertuples()])
known = np.array([(r.winner in ratings.index) and (r.loser in ratings.index) for r in test_m.itertuples()])
pre_res = {"matches": int(len(test_m)), "both players seen before": int(known.sum()),
           "Accuracy": float(np.mean(np.where(pre == 0.5, 0.5, pre > 0.5))), "Log loss": ll(pre),
           "Accuracy (both seen)": float(np.mean(np.where(pre[known] == 0.5, 0.5, pre[known] > 0.5))),
           "Coin flip log loss": float(np.log(2))}
print("pre-match", {k: round(v, 3) for k, v in pre_res.items()})

# how much weight the even 50% start gets against the rallies played so far (fitted on training matches)
best, best_ll = None, 1e9
for strength in (10, 25, 50, 100, 200, 400, 1000, 10**9):
    P = []
    for r in train_m.itertuples():
        P.extend(bd.replay(0.5, rl[rl.match_id == r.match_id], strength).a_win.iloc[:-1])
    if ll(P) < best_ll:
        best, best_ll = strength, ll(P)
print("start weight", best, "rallies")

# rally-by-rally on the 2022 matches
res = {}
for label, p0_fn, strength in (("Score only (every rally 50/50)", lambda r: 0.5, 10**9),
                               ("Score + rallies won so far (final)", lambda r: 0.5, best),
                               ("+ player ratings from 2018-21", lambda r: bd.rally_p(ratings, r.winner, r.loser), best)):
    P = []
    for r in test_m.itertuples():
        P.extend(bd.replay(p0_fn(r), rl[rl.match_id == r.match_id], strength).a_win.iloc[:-1])
    P = np.array(P)
    res[label] = {"Log loss": ll(P), "Accuracy": float(np.mean(np.where(P == 0.5, 0.5, P > 0.5))), "rallies": int(len(P))}
    print(f"{label:38s}", {k: round(v, 3) for k, v in res[label].items()})
# by stage of the match: after game 1
g1 = []
for r in test_m.itertuples():
    rp = bd.replay(0.5, rl[rl.match_id == r.match_id], best)
    nxt = rp[rp.game == 2]
    if len(nxt):
        g1.append(nxt.a_win.iloc[0])
g1 = np.array(g1)
res["after game 1"] = {"matches": int(len(g1)), "Accuracy": float(np.mean(g1 > 0.5)), "Log loss": ll(g1)}
print("after game 1", res["after game 1"])

final, serve_final, rallies_n = bd.fit_ratings(m, rl)
joblib.dump({"ratings": final, "serve": serve_final, "strength": best, "rallies": rallies_n}, ROOT / "models" / "badminton_model.joblib")
(ROOT / "models" / "badminton_info.json").write_text(json.dumps({
    "model": "Player ratings from rallies (regularised logistic regression) + Markov chain", "train": "2018-2021",
    "test": "2022", "pre_match": pre_res, "in_match": res, "strength": best, "serve_effect": serve}, indent=1))
