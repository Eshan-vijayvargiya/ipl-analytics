"""Football (international matches 1872-2026): cleaning, Elo ratings and a Poisson goal model.

How the models work
-------------------
* Elo ratings (World Football Elo rules): every team starts at 1500. After each match the winner takes
  points from the loser. Bigger tournaments, bigger wins and away wins move ratings more.
* Goal model: each team's goals in a match follow a Poisson distribution whose average depends on the
  Elo gap, home advantage and the type of match (fitted with scikit-learn's PoissonRegressor).
* In-play: at minute t with the score (h, a), the goals still to come are Poisson with the same averages
  scaled by the share of goals that are usually scored after minute t. Adding them to the current score gives
  the chances of a home win, a draw and an away win.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "football"

# How much a match moves Elo ratings (World Football Elo weights)
K_BY_TYPE = {"World Cup": 60, "Continental finals": 50, "Qualifier": 40, "Nations League": 40,
             "Other tournament": 30, "Friendly": 20}
CONTINENTAL = ["UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup", "Gold Cup",
               "Oceania Nations Cup", "Confederations Cup", "CONCACAF Championship"]
TYPES = ["Friendly", "Other tournament", "Qualifier", "Nations League", "Continental finals", "World Cup"]


def match_type(t):
    t = str(t)
    if t == "Friendly":
        return "Friendly"
    if "qualification" in t:
        return "Qualifier"
    if t == "FIFA World Cup":
        return "World Cup"
    if "Nations League" in t:
        return "Nations League"
    if t in CONTINENTAL:
        return "Continental finals"
    return "Other tournament"


# ------------------------------------------------------------------ cleaning
def clean(results, goals, shootouts):
    m = results.copy()
    m["date"] = pd.to_datetime(m.date)
    m = m.dropna(subset=["home_score", "away_score"]).sort_values(["date", "home_team"]).reset_index(drop=True)
    m["home_score"] = m.home_score.astype(int)
    m["away_score"] = m.away_score.astype(int)
    m["neutral"] = m.neutral.astype(str).str.upper().eq("TRUE") | (m.neutral is True)
    m["year"] = m.date.dt.year
    m["kind"] = m.tournament.map(match_type)
    m["result"] = np.select([m.home_score > m.away_score, m.home_score < m.away_score], ["H", "A"], "D")
    so = shootouts.assign(date=pd.to_datetime(shootouts.date))[["date", "home_team", "away_team", "winner"]]
    m = m.merge(so.rename(columns={"winner": "shootout_winner"}), on=["date", "home_team", "away_team"], how="left")
    m["match_id"] = np.arange(len(m))
    g = goals.copy()
    g["date"] = pd.to_datetime(g.date)
    g = g.merge(m[["match_id", "date", "home_team", "away_team"]], on=["date", "home_team", "away_team"])
    g = g.dropna(subset=["minute"])
    g["minute"] = g.minute.astype(int)
    g["own_goal"] = g.own_goal.astype(str).str.upper().eq("TRUE")
    g["penalty"] = g.penalty.astype(str).str.upper().eq("TRUE")
    # keep the goal timeline only for matches where every goal has a minute
    n = g.groupby("match_id").size()
    tot = m.set_index("match_id").eval("home_score + away_score")
    full = n.index[n.values == tot.reindex(n.index).values]
    m["has_timeline"] = m.match_id.isin(full)
    g = g[g.match_id.isin(full)].sort_values(["match_id", "minute"]).reset_index(drop=True)
    return add_elo(m), g[["match_id", "team", "scorer", "minute", "own_goal", "penalty", "assist"]]


# ------------------------------------------------------------------ Elo ratings
def add_elo(m, start=1500.0, home_adv=100.0):
    """Adds each team's rating BEFORE the match (elo_home, elo_away): what a model may use to predict it."""
    rating = {}
    eh, ea = np.empty(len(m)), np.empty(len(m))
    k_all = m.kind.map(K_BY_TYPE).to_numpy(float)
    for i, (h, a, hs, as_, neu) in enumerate(zip(m.home_team, m.away_team, m.home_score, m.away_score, m.neutral)):
        rh, ra = rating.get(h, start), rating.get(a, start)
        eh[i], ea[i] = rh, ra
        dr = rh - ra + (0 if neu else home_adv)
        expected = 1 / (10 ** (-dr / 400) + 1)
        actual = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        gd = abs(hs - as_)
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        change = k_all[i] * mult * (actual - expected)
        rating[h], rating[a] = rh + change, ra - change
    m = m.copy()
    m["elo_home"], m["elo_away"] = eh, ea
    return m


def ratings_now(m, start=1500.0, home_adv=100.0):
    """Every team's rating after its most recent match, highest first."""
    rating, history = {}, []
    k_all = m.kind.map(K_BY_TYPE).to_numpy(float)
    for i, (d, h, a, hs, as_, neu) in enumerate(zip(m.date, m.home_team, m.away_team, m.home_score, m.away_score, m.neutral)):
        rh, ra = rating.get(h, start), rating.get(a, start)
        dr = rh - ra + (0 if neu else home_adv)
        expected = 1 / (10 ** (-dr / 400) + 1)
        actual = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        gd = abs(hs - as_)
        mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        change = k_all[i] * mult * (actual - expected)
        rating[h], rating[a] = rh + change, ra - change
    return pd.Series(rating, name="elo").sort_values(ascending=False)


def elo_history(m, team):
    """A team's rating after each of its matches."""
    rows = []
    for side, opp, me, them in (("home", "away", "home_score", "away_score"), ("away", "home", "away_score", "home_score")):
        part = m[m[f"{side}_team"] == team]
        rows.append(pd.DataFrame({"date": part.date, "elo_before": part[f"elo_{side}"], "opponent": part[f"{opp}_team"],
                                  "scored": part[me], "conceded": part[them], "tournament": part.tournament,
                                  "match_id": part.match_id}))
    return pd.concat(rows).sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ the goal model
def team_rows(m):
    """Two rows per match, one per team: the features of that team's attack and the goals it scored."""
    home = pd.DataFrame({"elo_diff": m.elo_home - m.elo_away, "at_home": (~m.neutral).astype(int), "away_side": 0,
                         "kind": m.kind, "goals": m.home_score, "match_id": m.match_id, "year": m.year})
    away = pd.DataFrame({"elo_diff": m.elo_away - m.elo_home, "at_home": 0, "away_side": (~m.neutral).astype(int),
                         "kind": m.kind, "goals": m.away_score, "match_id": m.match_id, "year": m.year})
    return pd.concat([home, away], ignore_index=True)


def design(rows):
    X = pd.DataFrame({"elo_diff": rows.elo_diff / 100, "elo_diff_sq": (rows.elo_diff / 100).clip(-8, 8) ** 2 *
                      np.sign(rows.elo_diff), "at_home": rows.at_home, "away_side": rows.away_side})
    for k in TYPES[1:]:
        X[f"kind_{k}"] = (rows.kind == k).astype(int)
    return X


def fit_goal_model(m, alpha=1e-4):
    from sklearn.linear_model import PoissonRegressor
    rows = team_rows(m)
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(design(rows), rows.goals.clip(upper=10))
    return model


def expected_goals(model, elo_home, elo_away, neutral=True, kind="Friendly"):
    """Average goals for (home, away) in one match."""
    one = pd.DataFrame({"elo_diff": [elo_home - elo_away, elo_away - elo_home], "at_home": [0 if neutral else 1, 0],
                        "away_side": [0, 0 if neutral else 1], "kind": [kind, kind]})
    lam = model.predict(design(one))
    return float(lam[0]), float(lam[1])


def score_grid(lh, la, max_goals=10, rho=0.0):
    """Chance of every scoreline. rho > 0 adds a few draws (Dixon-Coles correction for 0-0 and 1-1)."""
    g = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    if rho:
        grid[0, 0] *= 1 - lh * la * rho
        grid[0, 1] *= 1 + lh * rho
        grid[1, 0] *= 1 + la * rho
        grid[1, 1] *= 1 - rho
    return grid / grid.sum()


def outcome(grid):
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())


# ------------------------------------------------------------------ in-play
def goal_time_share(goals):
    """share_after[t] = share of a match's goals usually scored after minute t (t = 0..120)."""
    mins = goals.minute.clip(1, 90).to_numpy()            # 90+ stoppage-time goals are recorded as 90
    counts = np.bincount(mins, minlength=91)[: 91].astype(float)
    after = 1 - np.cumsum(counts) / counts.sum()
    share = np.zeros(121)
    share[:91] = np.concatenate([[1.0], after[1:91]])
    share[0] = 1.0
    extra = 30 / 90 * np.mean(counts[46:91]) / np.mean(counts[1:91])     # extra time ~ the 2nd-half rate
    share[91:] = 0
    return share, extra


def in_play(lh, la, minute, home_goals, away_goals, share, extra_time=False, extra_share=0.33, rho=0.0):
    """Chances of (home win, draw, away win) at the end, given the score at `minute`."""
    if minute <= 90:
        left = share[int(minute)] + (extra_share if extra_time else 0)
    else:
        left = extra_share * max(0, 120 - minute) / 30
    grid = score_grid(lh * left, la * left, rho=rho * left)
    n = grid.shape[0]
    diff = np.subtract.outer(np.arange(n), np.arange(n)) + (home_goals - away_goals)
    return float(grid[diff > 0].sum()), float(grid[diff == 0].sum()), float(grid[diff < 0].sum())


def worm(model, match, goals, share, extra_share, rho=0.0):
    """The three chances after every minute of a match (and right after each goal)."""
    lh, la = expected_goals(model, match.elo_home, match.elo_away, bool(match.neutral), match.kind)
    gl = goals[goals.match_id == match.match_id]
    et = bool(gl.minute.max() > 90) if len(gl) else False
    et = et or isinstance(match.shootout_winner, str)
    end = 120 if et else 90
    rows = []
    for t in range(0, end + 1):
        so_far = gl[gl.minute <= t]
        h = int((so_far.team == match.home_team).sum())
        a = len(so_far) - h
        ph, pd_, pa = in_play(lh, la, t, h, a, share, extra_time=et and t <= 90, extra_share=extra_share, rho=rho)
        rows.append((t, h, a, ph, pd_, pa))
    w = pd.DataFrame(rows, columns=["minute", "home_goals", "away_goals", "home_win", "draw", "away_win"])
    return (lh, la), w, gl
