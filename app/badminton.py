"""Badminton (top BWF singles matches 2018-2022, every shot): cleaning, player ratings and a rally model.

Data: ShuttleSet and ShuttleSet22 (MIT licence), hand-labelled from BWF broadcast videos: every stroke's shot
type, where it landed, who won the rally and why.

The model:
* Player ratings come from every rally (a regularised logistic regression, like Elo fitted in one go):
  logit(P(A wins a rally)) = rating_A - rating_B + serve effect.
* The Markov chain in rally.py turns that rally chance into the chance of winning the game and the match
  from any score, and during a match the rally chance is updated with the points actually played.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import rally

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "badminton"

SHOTS = {"放小球": "Net shot", "擋小球": "Return net", "殺球": "Smash", "點扣": "Wrist smash", "挑球": "Lob",
         "防守回挑": "Defensive lob", "長球": "Clear", "平球": "Drive", "小平球": "Driven flight",
         "後場抽平球": "Back-court drive", "切球": "Drop", "過度切球": "Passive drop", "推球": "Push", "撲球": "Rush",
         "防守回抽": "Defensive drive", "勾球": "Cross-court net shot", "發短球": "Short serve", "發長球": "Long serve",
         "未知球種": "Unknown"}
ENDINGS = {"對手落地致勝": "Winner (landed in)", "出界": "Error: hit out", "掛網": "Error: hit the net",
           "未過網": "Error: didn't clear the net", "落點判斷失誤": "Error: misjudged (left it, it landed in)",
           "犯規": "Fault", "對手落地判斷失誤": "Winner (opponent misjudged)"}
SHOT_GROUP = {"Smash": "Attack", "Wrist smash": "Attack", "Rush": "Attack", "Drive": "Flat", "Driven flight": "Flat",
              "Back-court drive": "Flat", "Push": "Flat", "Defensive drive": "Defence", "Defensive lob": "Defence",
              "Lob": "Defence", "Clear": "Rear court", "Drop": "Rear court", "Passive drop": "Rear court",
              "Net shot": "Net", "Return net": "Net", "Cross-court net shot": "Net", "Short serve": "Serve",
              "Long serve": "Serve", "Unknown": "Other"}
# landing_area 1-9 is a 3x3 grid of the court: depth worked out from where each shot type lands
ZONES = {1: ("Front", "Side"), 7: ("Front", "Centre"), 2: ("Front", "Side"), 5: ("Mid", "Side"), 8: ("Mid", "Centre"),
         6: ("Mid", "Side"), 3: ("Back", "Side"), 9: ("Back", "Centre"), 4: ("Back", "Side")}
WOMEN = {"An Se Young", "Busanan ONGBAMRUNGPHAN", "Carolina MARIN", "Evgeniya KOSETSKAYA", "Mia BLICHFELDT",
         "Michelle LI", "Neslihan YIGIT", "PUSARLA V. Sindhu", "Pornpawee CHOCHUWONG", "Ratchanok INTANON",
         "Supanida KATETHONG", "Aakarshi KASHYAP", "Akane YAMAGUCHI", "CHEN Yufei", "Gregoria Mariska TUNJUNG",
         "HE Bingjiao", "WANG Zhi Yi", "TAI Tzu Ying", "Sayaka TAKAHASHI"}
INDIA = {"KIDAMBI Srikanth", "Sameer VERMA", "PUSARLA V. Sindhu", "Lakshya SEN", "PRANNOY H. S.", "Aakarshi KASHYAP"}
NAME_FIX = {"Anders Antonsen": "Anders ANTONSEN", "Viktor Axelsen": "Viktor AXELSEN"}


def tidy_name(n):
    n = " ".join(str(n).split())
    return NAME_FIX.get(n, n)


def short(n):
    """'Viktor AXELSEN' -> 'Axelsen'; 'PUSARLA V. Sindhu' -> 'Sindhu'; 'An Se Young' -> 'An Se Young'."""
    caps = [w for w in n.split() if w.isupper() and len(w) > 1]
    if n.startswith("PUSARLA"):
        return "Sindhu"
    if n == "PRANNOY H. S.":
        return "Prannoy"
    return caps[-1].title() if caps else n


# ------------------------------------------------------------------ cleaning
def clean(folders):
    """folders: list of ShuttleSet 'set' folders (each with match.csv and one sub-folder per match)."""
    ms, rallies, strokes, seen = [], [], [], set()
    for folder in folders:
        folder = Path(folder)
        meta = pd.read_csv(folder / "match.csv")
        for r in meta.itertuples():
            key = (tidy_name(r.winner), tidy_name(r.loser), str(r.tournament).upper(), str(r.round).lower().replace("-", ""))
            if key in seen:                       # the same match appears in both releases
                continue
            seen.add(key)
            mdir = folder / r.video
            if not mdir.exists():
                continue
            mid = len(ms)
            a, b = tidy_name(r.winner), tidy_name(r.loser)
            games = 0
            for g in (1, 2, 3):
                f = mdir / f"set{g}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f)
                if d.empty:
                    continue
                games += 1
                d["player_name"] = np.where(d.player == "A", a, b)
                d["shot"] = d.type.map(SHOTS).fillna("Unknown")
                d["zone"] = d.landing_area.map(lambda z: ZONES.get(z, ("Out", "Out"))[0] if pd.notna(z) else None)
                d["lane"] = d.landing_area.map(lambda z: ZONES.get(z, ("Out", "Out"))[1] if pd.notna(z) else None)
                d["ball_round"] = pd.to_numeric(d.ball_round, errors="coerce")
                d = d.sort_values(["rally", "ball_round"])
                strokes.append(pd.DataFrame({"match_id": mid, "game": g, "rally": d.rally, "stroke": d.ball_round,
                                             "player": d.player_name, "shot": d.shot, "zone": d.zone, "lane": d.lane,
                                             "backhand": d.backhand.fillna(0).astype(float) == 1}))
                first = d.groupby("rally").first()
                last = d.groupby("rally").last()
                n = d.groupby("rally").size()
                # who won each rally: read it off the running score (some rallies have no 'getpoint' label)
                sa, sb = last.roundscore_A.astype(int), last.roundscore_B.astype(int)
                da, db = sa.diff().fillna(sa), sb.diff().fillna(sb)
                won_by_a = np.where(da > db, True, np.where(db > da, False, last.getpoint_player.eq("A")))
                ending = last.lose_reason.map(ENDINGS)
                ending = ending.where(last.getpoint_player.isin(["A", "B"]), "Not labelled").fillna("Not labelled")
                rl = pd.DataFrame({"match_id": mid, "game": g, "rally": last.index,
                                   "score_a": sa.to_numpy(), "score_b": sb.to_numpy(),
                                   "a_won": won_by_a, "server_a": first.player.eq("A").to_numpy(),
                                   "shots": n.to_numpy(), "last_shot": last.shot.to_numpy(),
                                   "last_by_a": last.player.eq("A").to_numpy(), "ending": ending.to_numpy()})
                rallies.append(rl)
            date = pd.Timestamp(int(r.year), int(r.month), int(r.day)) if pd.notna(r.year) and pd.notna(r.month) and pd.notna(r.day) else pd.NaT
            ms.append({"match_id": mid, "winner": a, "loser": b, "tournament": str(r.tournament).title().replace("Bwf", "BWF").replace("Hsbc", "HSBC").replace("Yonex", "YONEX").replace("Toyota", "TOYOTA"),
                       "round": str(r.round).replace("-", " ").replace("Finals", "Final").replace("Final s", "Finals").strip().capitalize(),
                       "date": date, "year": int(r.year) if pd.notna(r.year) else None, "games": games,
                       "minutes": r.duration, "event": "Women's singles" if a in WOMEN or b in WOMEN else "Men's singles"})
    m = pd.DataFrame(ms)
    rl = pd.concat(rallies, ignore_index=True)
    st = pd.concat(strokes, ignore_index=True)
    rl = rl.sort_values(["match_id", "game", "rally"]).reset_index(drop=True)
    games = rl.groupby(["match_id", "game"]).agg(a=("score_a", "max"), b=("score_b", "max")).reset_index()
    games["a_won_game"] = games.a > games.b
    line = games.groupby("match_id").apply(lambda d: ", ".join(f"{x}-{y}" for x, y in zip(d.a, d.b)), include_groups=False)
    m["score_line"] = m.match_id.map(line)
    m["india"] = m.winner.isin(INDIA) | m.loser.isin(INDIA)
    return m, rl, st


# ------------------------------------------------------------------ player ratings (fitted on rallies)
def rally_table(m, rl):
    d = rl.merge(m[["match_id", "winner", "loser", "year"]], on="match_id")
    return d


def fit_ratings(m, rl, c=0.5):
    """Regularised logistic regression on rallies: +1 for player A's column, -1 for player B's, plus serve."""
    from sklearn.linear_model import LogisticRegression
    d = rally_table(m, rl)
    players = sorted(set(d.winner) | set(d.loser))
    idx = {p: i for i, p in enumerate(players)}
    X = np.zeros((len(d), len(players) + 1))
    X[np.arange(len(d)), d.winner.map(idx)] = 1
    X[np.arange(len(d)), d.loser.map(idx)] = -1
    X[:, -1] = np.where(d.server_a, 1, -1)
    # mirror every rally so the model can't learn "A always wins" (A is the match winner in the files)
    Xm = np.vstack([X, -X])
    y = np.concatenate([d.a_won.to_numpy(), ~d.a_won.to_numpy()]).astype(int)
    lr = LogisticRegression(C=c, fit_intercept=False, max_iter=2000)
    lr.fit(Xm, y)
    ratings = pd.Series(lr.coef_[0][:-1], index=players).sort_values(ascending=False)
    rallies = pd.concat([d.winner, d.loser]).value_counts()
    return ratings, float(lr.coef_[0][-1]), rallies


def rally_p(ratings, a, b):
    """Chance that a wins a rally against b (players never seen get the average rating, 0)."""
    return float(rally.expit(ratings.get(a, 0.0) - ratings.get(b, 0.0)))


def replay(p0, rl_match, strength):
    """Player A's chance of winning the match before every rally."""
    rows, games_a, games_b, won, played = [], 0, 0, 0, 0
    for g, d in rl_match.groupby("game", sort=True):
        a = b = 0
        for r in d.itertuples():
            p = rally.update_p(p0, won, played, strength)
            rows.append({"game": g, "rally": r.rally, "score_a": a, "score_b": b, "games_a": games_a, "games_b": games_b,
                         "p_rally": p, "a_win": rally.match(games_a, games_b, p, "badminton", a, b),
                         "point_value": rally.point_value(p, 21, 30, a, b),
                         "a_won": r.a_won, "shots": r.shots, "last_shot": r.last_shot, "ending": r.ending,
                         "last_by_a": r.last_by_a})
            won += r.a_won
            played += 1
            a, b = (a + 1, b) if r.a_won else (a, b + 1)
        games_a += a > b
        games_b += b > a
    out = pd.DataFrame(rows)
    end = {"game": out.game.max(), "rally": out.rally.max() + 1, "score_a": a, "score_b": b, "games_a": games_a,
           "games_b": games_b, "p_rally": rally.update_p(p0, won, played, strength),
           "a_win": 1.0 if games_a > games_b else 0.0, "point_value": 0.0, "a_won": None, "shots": 0, "last_shot": "",
           "ending": "", "last_by_a": None}
    out = pd.concat([out, pd.DataFrame([end])], ignore_index=True)
    out["n"] = np.arange(len(out))
    return out
