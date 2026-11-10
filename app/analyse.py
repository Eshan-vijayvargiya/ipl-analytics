"""Analysis of an uploaded table once the engine knows its sport, kind and columns.

Results tables (any sport): standings, Elo ratings learned from the file itself, a walk-forward test of how well
those ratings predict later matches, head-to-head and a predictor that uses each sport's own model.
Play-by-play tables: the sport's win-probability replay (cricket chase model, NBA random walk, football Poisson,
badminton and volleyball Markov chains). Player tables: leaderboards with sport-aware rates.
"""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import basketball as bb
import engine as en
import football as fb
import rally
import volleyball as vb

ROOT = Path(__file__).resolve().parent.parent
K = {"Football": 30, "Basketball": 20, "Volleyball": 30, "Badminton": 32, "Cricket": 32, None: 30}
HOME_ELO = {"Football": 100, "Basketball": 100, "Volleyball": 40, "Badminton": 0, "Cricket": 30, None: 50}


@lru_cache(maxsize=None)
def models():
    out = {}
    for name in ("football", "basketball", "volleyball", "badminton"):
        try:
            out[name] = joblib.load(ROOT / "models" / f"{name}_model.joblib")
        except Exception:
            out[name] = None
    try:
        out["cricket"] = joblib.load(ROOT / "models" / "win_model.joblib")
    except Exception:
        out["cricket"] = None
    return out


# ================================================================== results tables
def _pairs(text):
    return [(int(a), int(b)) for a, b in en.SCORE_PAIR.findall(str(text))]


def standard_results(d, sport, roles):
    """Matches as: a, b, sa, sb (goals/points/sets), outcome 'A' | 'B' | 'D', plus date/competition/neutral if known."""
    m = pd.DataFrame({"a": d["side_a"], "b": d["side_b"]}, index=d.index)
    m["date"] = d.get("date")
    m["competition"] = d.get("competition")
    m["season"] = d.get("season")
    if "score_a" in d and "score_b" in d:
        m["sa"], m["sb"] = d["score_a"], d["score_b"]
    elif "score_text" in d:
        t = d["score_text"].astype(str)
        if sport == "Cricket" and t.str.contains("/").mean() > 0.3:
            runs = t.str.findall(r"(\d{1,3})\s*/\s*\d{1,2}")
            m["sa"] = pd.to_numeric(runs.str[0], errors="coerce")
            m["sb"] = pd.to_numeric(runs.str[1], errors="coerce")
        else:
            pr = t.map(_pairs)
            many = pr.str.len() >= 2
            if sport in ("Badminton", "Volleyball") and many.mean() > 0.5:
                m["sa"] = pr.map(lambda p: sum(x > y for x, y in p) if p else np.nan)
                m["sb"] = pr.map(lambda p: sum(y > x for x, y in p) if p else np.nan)
                m["detail"] = t
            else:
                m["sa"] = pr.map(lambda p: p[0][0] if p else np.nan)
                m["sb"] = pr.map(lambda p: p[0][1] if p else np.nan)
    else:
        m["sa"] = m["sb"] = np.nan
    m["outcome"] = np.select([m.sa > m.sb, m.sa < m.sb, (m.sa == m.sb) & m.sa.notna()], ["A", "B", "D"], None)
    if "winner" in d and "side_a" in roles and roles.get("winner") not in (roles.get("side_a"), roles.get("side_b")):
        w = d["winner"].astype(str).str.casefold().str.strip()
        aa, bb_ = m.a.astype(str).str.casefold(), m.b.astype(str).str.casefold()
        from_w = np.select([w == aa, w == bb_, w.isin(["draw", "tie", "tied", "no result", "nr", "drawn", "d"])], ["A", "B", "D"], None)
        m["outcome"] = np.where(pd.Series(from_w, index=m.index).notna(), from_w, m.outcome)
    if sport != "Football" and sport is not None:
        m.loc[m.outcome == "D", "outcome"] = "D" if sport == "Cricket" else None
    m["neutral"] = d["neutral"].astype(str).str.lower().isin(["true", "1", "yes", "y", "neutral"]) if "neutral" in d else False
    m = m.dropna(subset=["a", "b"])
    m = m[m.a != m.b]
    if m.date.notna().mean() > 0.8:
        m = m.sort_values("date", kind="stable")
    return m.reset_index(drop=True)


def has_home(roles, df_columns=None):
    col = str(roles.get("side_a", ""))
    return bool(en.tokens(col) & {"home", "host", "hosts", "hometeam"})


def win_chances(sport, ea, eb, home, neutral=False):
    """(P a wins, P draw, P b wins) from two Elo ratings, using each sport's own model."""
    mdl = models()
    at_home = home and not neutral
    if sport == "Football" and mdl["football"] is not None:
        lh, la = fb.expected_goals(mdl["football"]["model"], ea, eb, neutral=not at_home, kind="Other tournament")
        return fb.outcome(fb.score_grid(lh, la, rho=mdl["football"]["rho"]))
    if sport == "Basketball" and mdl["basketball"] is not None:
        per100, sigma, home_pts = mdl["basketball"]["params"]
        p = float(bb.win_prob((per100, sigma, home_pts if at_home else 0.0), 0, 1.0, ea - eb))
        return p, 0.0, 1 - p
    if sport == "Volleyball" and mdl["volleyball"] is not None:
        a0, b1 = mdl["volleyball"]["coef"]
        p_r = float(rally.expit((a0 if at_home else 0.0) + b1 * (ea - eb) / 100))
        p = rally.match(0, 0, p_r, "volleyball")
        return p, 0.0, 1 - p
    dr = ea - eb + (HOME_ELO.get(sport, 50) if at_home else 0)
    p = 1 / (1 + 10 ** (-dr / 400))
    return p, 0.0, 1 - p


def elo_walk(m, sport, home):
    """Elo ratings learned from the file, match by match. Also the chances the model gave BEFORE each match
    (a walk-forward test: every prediction only uses earlier matches)."""
    k, adv = K.get(sport, 30), HOME_ELO.get(sport, 50)
    rating, played = {}, {}
    pre_a, pre_b, P = [], [], []
    for r in m.itertuples():
        ea, eb = rating.get(r.a, 1500.0), rating.get(r.b, 1500.0)
        pre_a.append(ea)
        pre_b.append(eb)
        P.append(win_chances(sport, ea, eb, home, bool(r.neutral)))
        if r.outcome not in ("A", "B", "D"):
            continue
        dr = ea - eb + (adv if home and not r.neutral else 0)
        exp = 1 / (1 + 10 ** (-dr / 400))
        act = {"A": 1.0, "B": 0.0, "D": 0.5}[r.outcome]
        mult = 1.0
        if pd.notna(r.sa) and pd.notna(r.sb):
            gd = abs(r.sa - r.sb)
            if sport == "Football":
                mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
            elif sport == "Basketball":
                mult = (gd + 3) ** 0.8 / (7.5 + 0.006 * abs(dr))
        ch = k * mult * (act - exp)
        rating[r.a], rating[r.b] = ea + ch, eb - ch
        played[r.a] = played.get(r.a, 0) + 1
        played[r.b] = played.get(r.b, 0) + 1
    m = m.copy()
    m["elo_a"], m["elo_b"] = pre_a, pre_b
    P = np.array(P) if P else np.zeros((0, 3))
    m["p_a"], m["p_d"], m["p_b"] = P[:, 0], P[:, 1], P[:, 2]
    return m, pd.Series(rating).sort_values(ascending=False), pd.Series(played)


def evaluate(m, sport, burn=0.3):
    """How well did ratings learned from earlier matches predict the later ones?"""
    d = m[m.outcome.isin(["A", "B", "D"])]
    if len(d) < 30:
        return None
    start = int(len(d) * burn)
    test = d.iloc[start:]
    first = d.iloc[:start]
    pick = np.array(["A", "D", "B"])[np.argmax(test[["p_a", "p_d", "p_b"]].to_numpy(), axis=1)]
    acc = float(np.mean(pick == test.outcome.to_numpy()))
    idx = test.outcome.map({"A": 0, "D": 1, "B": 2}).to_numpy()
    probs = test[["p_a", "p_d", "p_b"]].to_numpy()
    if sport != "Football":                      # two-way sports: a draw/tie counts half each way
        two = test.outcome != "D"
        probs, idx, pick_ok = probs[two.to_numpy()], idx[two.to_numpy()], (pick == test.outcome.to_numpy())[two.to_numpy()]
        acc = float(np.mean(pick_ok)) if len(pick_ok) else acc
    ll = float(-np.mean(np.log(np.clip(probs[np.arange(len(idx)), idx], 1e-9, 1))))
    base = first.outcome.value_counts(normalize=True)
    base_p = np.array([base.get("A", 0), base.get("D", 0), base.get("B", 0)]) + 1e-3
    base_p = base_p / base_p.sum()
    base_ll = float(-np.mean(np.log(base_p[idx])))
    base_acc = float(np.mean(np.array(["A", "D", "B"])[np.argmax(base_p)] == test.outcome.to_numpy()))
    return {"tested": int(len(test)), "learned_from": int(start), "accuracy": acc, "log_loss": ll,
            "baseline_accuracy": base_acc, "baseline_log_loss": base_ll, "baseline": "always picking " +
            {"A": "the first-named side", "B": "the second-named side", "D": "a draw"}[["A", "D", "B"][int(np.argmax(base_p))]]}


def standings(m, sport):
    rows = []
    for side, other, s_me, s_them, win in (("a", "b", "sa", "sb", "A"), ("b", "a", "sb", "sa", "B")):
        rows.append(pd.DataFrame({"team": m[side], "won": m.outcome == win, "drawn": m.outcome == "D",
                                  "lost": m.outcome.isin(["A", "B"]) & (m.outcome != win), "for": m[s_me], "against": m[s_them]}))
    t = pd.concat(rows)
    s = t.groupby("team").agg(P=("won", "size"), W=("won", "sum"), D=("drawn", "sum"), L=("lost", "sum"),
                              For=("for", "sum"), Against=("against", "sum"))
    s["Win %"] = s.W / s.P * 100
    s["Diff"] = s.For - s.Against
    if sport == "Football":
        s["Pts"] = s.W * 3 + s.D
        s = s.sort_values(["Pts", "Diff", "For"], ascending=False)
    else:
        s = s.sort_values(["W", "Win %", "Diff"], ascending=False)
    return s.reset_index()


# ================================================================== player / team stats tables
RATES = {
    "Cricket": [("Strike rate", {"runs"}, {"balls", "bf", "ballsfaced"}, 100), ("Average", {"runs"}, {"outs", "dismissals", "no"}, 1),
                ("Economy", {"conceded", "runsconceded"}, {"overs", "o"}, 1)],
    "Basketball": [("Points a game", {"pts", "points"}, {"gp", "games", "g", "matches"}, 1),
                   ("Rebounds a game", {"reb", "rebounds", "trb"}, {"gp", "games", "g"}, 1),
                   ("Assists a game", {"ast", "assists"}, {"gp", "games", "g"}, 1)],
    "Football": [("Goals a game", {"goals", "gls", "g"}, {"apps", "appearances", "matches", "mp", "games", "gp"}, 1),
                 ("Assists a game", {"assists", "ast", "a"}, {"apps", "appearances", "matches", "mp", "games"}, 1)],
    "Volleyball": [("Points a match", {"points", "pts"}, {"matches", "games", "mp", "m"}, 1),
                   ("Attack success %", {"kills", "attackpoints", "attpt"}, {"attacks", "attempts", "atttot"}, 100)],
    "Badminton": [("Win %", {"wins", "won", "w"}, {"matches", "played", "mp", "p"}, 100)],
}


def player_table(df, roles, sport):
    """Entity column + every numeric column (cleaned), plus sport-aware rates where the parts exist."""
    ent = roles.get("player") or roles.get("team")
    out = pd.DataFrame({"name": en.tidy_names(df[ent])[0]})
    if roles.get("team") and roles.get("player") and roles["team"] != ent:
        out["team"] = en.tidy_names(df[roles["team"]])[0]
    nums = {}
    for c in df.columns:
        if c in (ent, roles.get("team")):
            continue
        if en.num_share(df[c]) >= 0.7:
            v, _ = en.to_number(df[c])
            nums[c] = v
    for c, v in nums.items():
        out[c] = v
    by_tok = {c: en.tokens(c) | {en.key(c)} for c in nums}
    for label, top, bottom, scale in RATES.get(sport, []):
        tc = next((c for c, t in by_tok.items() if t & top), None)
        bc = next((c for c, t in by_tok.items() if t & bottom and c != tc), None)
        if tc and bc and label not in out:
            out[label] = out[tc] / out[bc].replace(0, np.nan) * scale
    out = out.dropna(subset=["name"])
    if out.name.duplicated().any():          # one row per game -> totals per player
        num_cols = [c for c in out.columns if c not in ("name", "team") and pd.api.types.is_numeric_dtype(out[c])]
        agg = {c: "sum" for c in num_cols if c not in [r[0] for r in RATES.get(sport, [])]}
        if "team" in out:
            agg["team"] = "first"
        games = out.groupby("name").size().rename("Rows (games)")
        out = out.groupby("name").agg(agg).join(games).reset_index()
    return out


# ================================================================== play-by-play tables
def cricket_events(d):
    """Ball-by-ball rows -> per match: innings totals, and the chase's win-probability worm (IPL chase model)."""
    b = pd.DataFrame(index=d.index)
    b["match_id"] = d["match_id"] if "match_id" in d else (d["date"].astype(str) if "date" in d else 1)
    inn = d.get("period")
    b["inning"] = inn.fillna(1).astype(int) if inn is not None else 1
    over = d.get("over")
    ball = d.get("ball")
    if over is None and ball is not None:                      # Cricsheet style: 0.1, 0.2 ... 19.6
        over, ball = np.floor(ball), np.round((ball - np.floor(ball)) * 10)
    elif over is not None and ball is None and (over % 1 != 0).mean() > 0.3:
        over, ball = np.floor(over), np.round((over - np.floor(over)) * 10)
    b["over"] = (over if over is not None else 0)
    b["ball"] = (ball if ball is not None else 0)
    if "over" in d and b.over.min() >= 1 and b.over.max() <= 20 and (b.over == 0).sum() == 0:
        b["over"] = b.over - 1                                   # 1-based overs -> 0-based
    runs = d["runs"].fillna(0)
    extras = d["extras"].fillna(0) if "extras" in d else 0
    b["batter_runs"] = runs
    b["total_runs"] = runs + extras
    b["batter"] = d.get("batter", pd.Series("", index=d.index)).fillna("")
    b["bowler"] = d.get("bowler", pd.Series("", index=d.index)).fillna("")
    b["batting_team"] = d.get("batting_team", pd.Series("", index=d.index))
    w = d.get("wicket")
    if w is None:
        b["wicket"] = False
        b["bowler_wicket"] = False
    else:
        ws = w.astype(str).str.strip().str.lower()
        b["wicket"] = w.notna() & ~ws.isin(["", "0", "false", "no", "nan", "none", "not out", "n", "<na>"])
        not_bowler = ws.str.contains("run out|runout|retired|obstruct|timed out|handled", regex=True)
        b["bowler_wicket"] = b.wicket & ~not_bowler
    b["legal"] = True
    b = b.sort_values(["match_id", "inning", "over", "ball"], kind="stable")
    out = []
    mdl = models()["cricket"]
    for mid, mb in b.groupby("match_id", sort=False):
        i1, i2 = mb[mb.inning == 1], mb[mb.inning == 2]
        info = {"match": mid, "teams": " vs ".join(str(x) for x in mb.batting_team.dropna().unique()[:2]),
                "first": int(i1.total_runs.sum()), "first_wkts": int(i1.wicket.sum()),
                "second": int(i2.total_runs.sum()) if len(i2) else None, "second_wkts": int(i2.wicket.sum()) if len(i2) else None}
        if mdl is not None and len(i1) and len(i2) >= 6:
            target = info["first"] + 1
            start, w = __import__("winprob").worm(mdl, mb.assign(legal=True), target, False)
            info["worm"] = (start, w)
        out.append(info)
    return out, b


def basketball_events(d):
    """Play-by-play rows -> per game: the home team's win chance after each change of score (NBA model, equal teams)."""
    mdl = models()["basketball"]
    per100, sigma, home_pts = mdl["params"] if mdl else (6.3, 20.5, 3.5)
    g = pd.DataFrame(index=d.index)
    g["game"] = d["match_id"] if "match_id" in d else (d["date"].astype(str) if "date" in d else 1)
    g["period"] = d["period"].fillna(1).astype(int)
    if "clock" in d:
        c = d["clock"].astype(str)
        mm = c.str.extract(r"(\d+):(\d+)").astype(float)
        pt = c.str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
        left = (mm[0] * 60 + mm[1]).fillna(pt[0] * 60 + pt[1]).fillna(pd.to_numeric(c, errors="coerce"))
    else:
        left = (12 - d["minute"].fillna(0)).clip(lower=0) * 60
    g["t"] = bb.elapsed(g.period.to_numpy(), left.fillna(0).to_numpy())
    if "score_a" in d and "score_b" in d:
        g["home"], g["away"] = d["score_a"], d["score_b"]
    else:
        pr = d["score_text"].astype(str).map(_pairs)
        g["away"] = pr.map(lambda p: p[0][0] if p else np.nan)      # NBA style 'away - home'
        g["home"] = pr.map(lambda p: p[0][1] if p else np.nan)
    g[["home", "away"]] = g.groupby("game")[["home", "away"]].ffill().fillna(0)
    out = []
    for gid, x in g.sort_values(["game", "t"], kind="stable").groupby("game", sort=False):
        x = x[(x.home.diff().fillna(1) != 0) | (x.away.diff().fillna(1) != 0)]
        f = bb.share_left(x.t, x.period)
        p = bb.win_prob((per100, sigma, home_pts), x.home - x.away, f, 0)
        p = np.asarray(p, float)
        if len(p):
            p[-1] = 1.0 if x.home.iloc[-1] > x.away.iloc[-1] else 0.0 if x.home.iloc[-1] < x.away.iloc[-1] else 0.5
        out.append({"game": gid, "final": f"{int(x.away.iloc[-1])}-{int(x.home.iloc[-1])} (away-home)",
                    "worm": pd.DataFrame({"minute": x.t / 60, "home_win": p, "home": x.home, "away": x.away})})
    return out


def football_events(d, roles):
    """Goal rows (minute + team) -> per match: the home win / draw / away win chances, minute by minute."""
    mdl = models()["football"]
    key_cols = [r for r in ("match_id",) if r in d] or [r for r in ("date", "side_a", "side_b") if r in d]
    gl = d.copy()
    gl["mkey"] = gl[key_cols].astype(str).agg(" · ".join, axis=1) if key_cols else "match"
    out = []
    share = mdl["share"] if mdl else np.linspace(1, 0, 121)
    for mk, x in gl.groupby("mkey", sort=False):
        if "side_a" in x and "side_b" in x:
            home, away = x.side_a.dropna().iloc[0] if x.side_a.notna().any() else "Home", x.side_b.dropna().iloc[0] if x.side_b.notna().any() else "Away"
        else:
            teams = x["team"].dropna().unique().tolist() if "team" in x else []
            home, away = (teams + ["Team 1", "Team 2"])[:2]
        tcol = "team" if "team" in x else None
        goals = x.dropna(subset=["minute"]).sort_values("minute")
        rows = []
        end = 120 if goals.minute.max() > 90 else 90
        for t in range(0, int(end) + 1):
            so = goals[goals.minute <= t]
            h = int((so[tcol] == home).sum()) if tcol else 0
            a = len(so) - h
            ph, pdr, pa = fb.in_play(1.45, 1.15, t, h, a, share, extra_time=end == 120 and t <= 90)
            rows.append((t, h, a, ph, pdr, pa))
        w = pd.DataFrame(rows, columns=["minute", "home_goals", "away_goals", "home_win", "draw", "away_win"])
        out.append({"match": mk, "home": home, "away": away, "worm": w, "goals": goals})
    return out


def badminton_events(d, roles):
    """Point-by-point rows -> per match: the chance the first player wins, after every rally (Markov chain)."""
    strength = (models()["badminton"] or {}).get("strength", 100)
    x = d.copy()
    x["mkey"] = x["match_id"].astype(str) if "match_id" in x else "match"
    out = []
    for mk, m in x.groupby("mkey", sort=False):
        m = m.sort_index()
        if "score_a" in m and "score_b" in m:
            sa, sb = m.score_a.fillna(0).astype(int).to_numpy(), m.score_b.fillna(0).astype(int).to_numpy()
            games = m.period.fillna(1).astype(int).to_numpy()
            a_won = []
            prev = {}
            for g_, a_, b_ in zip(games, sa, sb):
                pa, pb = prev.get(g_, (0, 0))
                a_won.append(a_ - pa > b_ - pb)
                prev[g_] = (a_, b_)
        else:
            names = m.point_winner.dropna().unique().tolist()
            first = names[0] if names else None
            a_won = (m.point_winner == first).to_list()
            games = m.period.fillna(1).astype(int).to_numpy()
        rows, ga, gb, won, played = [], 0, 0, 0, 0
        for g_ in pd.unique(games):
            mask = games == g_
            aw = [w for w, k in zip(a_won, mask) if k]
            a = b = 0
            for w in aw:
                p = rally.update_p(0.5, won, played, strength)
                rows.append({"n": len(rows), "game": int(g_), "score": f"{a}-{b}", "a_win": rally.match(ga, gb, p, "badminton", a, b)})
                won += w
                played += 1
                a, b = (a + 1, b) if w else (a, b + 1)
            ga += a > b
            gb += b > a
        rows.append({"n": len(rows), "game": int(g_) if len(games) else 1, "score": f"{a}-{b}", "a_win": 1.0 if ga > gb else 0.0 if gb > ga else 0.5})
        out.append({"match": mk, "worm": pd.DataFrame(rows), "games": f"{ga}-{gb}"})
    return out


def volleyball_events(d, roles):
    """Set-score rows -> per match: the first team's chance of winning after every set (Markov chain)."""
    mdl = models()["volleyball"]
    coef = mdl["coef"] if mdl else (0.037, 0.082)
    strength = mdl["strength"] if mdl else 800
    x = d.copy()
    x["mkey"] = x["match_id"].astype(str) if "match_id" in x else "match"
    out = []
    for mk, m in x.groupby("mkey", sort=False):
        m = m.sort_values("period")
        p0 = float(rally.expit(coef[0]))
        rows = [{"after": "Start", "a_win": rally.match(0, 0, p0, "volleyball")}]
        sa = sb = won = played = 0
        for r in m.itertuples():
            if pd.isna(r.score_a) or pd.isna(r.score_b):
                continue
            sa += r.score_a > r.score_b
            sb += r.score_b > r.score_a
            won += r.score_a
            played += r.score_a + r.score_b
            p = rally.update_p(p0, won, played, strength)
            rows.append({"after": f"Set {int(r.period)} ({int(r.score_a)}-{int(r.score_b)})", "a_win": rally.match(int(sa), int(sb), p, "volleyball")})
        out.append({"match": mk, "worm": pd.DataFrame(rows), "sets": f"{int(sa)}-{int(sb)}"})
    return out
