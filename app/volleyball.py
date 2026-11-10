"""Volleyball (German Bundesliga, men and women, 2013/14 to 2018/19): cleaning, Elo and a rally model.

The model in one line: a team's chance of winning a single rally comes from the Elo gap (a logistic link
fitted on every rally of the training seasons), and the Markov chain in rally.py turns that rally chance into
the chance of winning a set and the match, and of each final score (3-0, 3-1, 3-2 ...).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import rally

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "volleyball"
KEY = ["gender", "season", "match_id"]


def season_label(sid):
    sid = int(sid)
    return f"20{sid // 100:02d}/{sid % 100:02d}"


# ------------------------------------------------------------------ cleaning
def clean(matches, sets, matchstats, players):
    m = matches.rename(columns={"league_gender": "gender"}).copy()
    m["season"] = m.season_id.map(season_label)
    home = m[m.match == "home"].set_index(KEY)
    away = m[m.match == "away"].set_index(KEY)
    both = home.index.intersection(away.index)
    g = pd.DataFrame({
        "date": pd.to_datetime(home.loc[both, "date_time"]),
        "stage": home.loc[both, "competition_stage"].replace({"Finale": "Final"}),
        "home": home.loc[both, "team_name"], "away": away.loc[both, "team_name"],
        "home_id": home.loc[both, "team_id"], "away_id": away.loc[both, "team_id"],
        "home_sets": home.loc[both, "set_won"].astype(int), "away_sets": away.loc[both, "set_won"].astype(int),
        "spectators": home.loc[both, "spectators"], "minutes": home.loc[both, "match_duration"]}).reset_index()
    g = g[(g[["home_sets", "away_sets"]].max(axis=1) == 3) & (g[["home_sets", "away_sets"]].min(axis=1) < 3)]

    s = sets.rename(columns={"league_gender": "gender"}).copy()
    s["season"] = s.season_id.map(season_label)
    sp = s.pivot_table(index=KEY + ["set"], columns="match", values="pt_set", aggfunc="first").reset_index()
    sp = sp.rename(columns={"home": "home_pts", "away": "away_pts"}).dropna(subset=["home_pts", "away_pts"])
    # a few sets hold averages (22.8) instead of points: those can't be real scores, so they are dropped
    sp = sp[(sp.home_pts % 1 == 0) & (sp.away_pts % 1 == 0)]
    sp[["home_pts", "away_pts"]] = sp[["home_pts", "away_pts"]].astype(int)
    target = np.where(sp.set == 5, 15, 25)
    hi, lo = sp[["home_pts", "away_pts"]].max(axis=1), sp[["home_pts", "away_pts"]].min(axis=1)
    sp = sp[(hi >= target) & ((hi - lo == 2) | ((hi == target) & (hi - lo >= 2)))]     # only scores that can end a set
    sp = sp.merge(g[KEY], on=KEY)
    complete = sp.groupby(KEY).size().rename("n_sets").reset_index()
    g = g.merge(complete, on=KEY, how="left")
    g["complete"] = g.n_sets == g.home_sets + g.away_sets
    pts = sp.groupby(KEY)[["home_pts", "away_pts"]].sum()
    g = g.merge(pts, on=KEY, how="left")
    g.loc[~g.complete, ["home_pts", "away_pts"]] = np.nan
    lines = sp.sort_values("set").groupby(KEY).apply(
        lambda d: ", ".join(f"{a}-{b}" for a, b in zip(d.home_pts, d.away_pts)), include_groups=False).rename("score_line")
    g = g.merge(lines.reset_index(), on=KEY, how="left")
    g.loc[~g.complete, "score_line"] = None
    g["home_won"] = g.home_sets > g.away_sets
    g["result"] = g.home_sets.astype(str) + "-" + g.away_sets.astype(str)
    g = g.sort_values(["date", "gender", "match_id"]).reset_index(drop=True)
    g["uid"] = np.arange(len(g))

    p = players.drop(columns=["gender"], errors="ignore").rename(columns={"league_gender": "gender"})
    p["season"] = p.season_id.map(season_label)
    p["name"] = (p.firstname.fillna("") + " " + p.lastname.fillna("")).str.strip()
    ps = matchstats.rename(columns={"league_gender": "gender"}).copy()
    ps["season"] = ps.season_id.map(season_label)
    ps = ps.merge(p[["gender", "season", "team_id", "player_id", "name", "position", "height", "nationality"]],
                  on=["gender", "season", "team_id", "player_id"], how="left")
    teams = pd.concat([m[["gender", "season", "team_id", "team_name"]]]).drop_duplicates(["gender", "season", "team_id"])
    ps = ps.merge(teams, on=["gender", "season", "team_id"], how="left").merge(g[KEY + ["uid"]], on=KEY)
    ps = ps.rename(columns={"team_name": "team", "pt_tot": "points", "serv_pt": "aces", "att_pt": "kills",
                            "att_tot": "attacks", "att_err": "attack_errors", "att_blo": "attacks_blocked",
                            "blo_pt": "blocks", "serv_tot": "serves", "serv_err": "serve_errors",
                            "rec_tot": "receptions", "rec_err": "reception_errors"})
    keep = ["uid", "gender", "season", "team", "player_id", "name", "position", "height", "nationality", "points",
            "kills", "attacks", "attack_errors", "attacks_blocked", "aces", "serves", "serve_errors", "blocks",
            "receptions", "reception_errors"]
    ps = ps[keep].dropna(subset=["name"])
    sp = sp.merge(g[KEY + ["uid"]], on=KEY)[["uid", "set", "home_pts", "away_pts"]]
    return add_elo(g), sp, ps


# ------------------------------------------------------------------ Elo (men's and women's leagues kept apart)
def add_elo(g, k=30, home_adv=40, carry=0.8, mean=1500):
    g = g.copy()
    eh, ea = np.empty(len(g)), np.empty(len(g))
    rating, last_season = {}, {}
    for i, r in enumerate(g.itertuples()):
        out = []
        for team in (r.home, r.away):
            key = (r.gender, team)
            if key in last_season and last_season[key] != r.season:
                rating[key] = carry * rating[key] + (1 - carry) * mean
            last_season[key] = r.season
            out.append(rating.get(key, mean))
        rh, ra = out
        eh[i], ea[i] = rh, ra
        expected = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        margin = abs(r.home_sets - r.away_sets)              # 3-0 counts more than 3-2
        mult = {3: 1.25, 2: 1.0, 1: 0.75}.get(margin, 1.0)
        change = k * mult * ((1.0 if r.home_won else 0.0) - expected)
        rating[(r.gender, r.home)], rating[(r.gender, r.away)] = rh + change, ra - change
    g["elo_home"], g["elo_away"] = eh, ea
    return g


# ------------------------------------------------------------------ rally model
def fit_rally(g):
    """Logistic link: logit(p_home_rally) = a + b * (Elo gap / 100). Fitted on rally counts (binomial)."""
    from scipy.optimize import minimize
    d = g.dropna(subset=["home_pts", "away_pts"])
    w, l, gap = d.home_pts.to_numpy(float), d.away_pts.to_numpy(float), ((d.elo_home - d.elo_away) / 100).to_numpy()

    def nll(c):
        p = rally.expit(c[0] + c[1] * gap)
        return -np.sum(w * np.log(p) + l * np.log(1 - p)) / np.sum(w + l)
    res = minimize(nll, [0.0, 0.05], method="Nelder-Mead")
    return tuple(float(v) for v in res.x)


def rally_p(coef, elo_home, elo_away):
    return float(rally.expit(coef[0] + coef[1] * (elo_home - elo_away) / 100))


def match_prob(coef, elo_home, elo_away):
    return rally.match(0, 0, rally_p(coef, elo_home, elo_away), "volleyball")


def replay(coef, game, sets, strength=60):
    """Home team's chance of winning the match before the first serve and after every set.
    After each set the rally estimate is updated with the points actually won (a Bayesian blend)."""
    p0 = rally_p(coef, game.elo_home, game.elo_away)
    s = sets[sets.uid == game.uid].sort_values("set")
    rows = [{"after": "Start", "home_sets": 0, "away_sets": 0, "p_rally": p0, "home_win": rally.match(0, 0, p0, "volleyball")}]
    hs = as_ = won = played = 0
    for r in s.itertuples():
        hs += r.home_pts > r.away_pts
        as_ += r.home_pts < r.away_pts
        won += r.home_pts
        played += r.home_pts + r.away_pts
        p = rally.update_p(p0, won, played, strength)
        rows.append({"after": f"Set {r.set} ({r.home_pts}-{r.away_pts})", "home_sets": hs, "away_sets": as_, "p_rally": p,
                     "home_win": rally.match(hs, as_, p, "volleyball")})
    return pd.DataFrame(rows)
