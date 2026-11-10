"""Basketball (NBA 2021-22 to 2023-24, every play): cleaning, Elo ratings and in-game win probability.

How the win-probability model works
-----------------------------------
Basketball scoring behaves a lot like a random walk: hundreds of small scoring chances, each adding a few
points to one side. So the final margin is roughly normal around

    current margin + (expected margin for the rest of the game)

with a spread that shrinks as the clock runs down (it scales with the square root of the time left).
The home team's chance of winning is the area of that normal curve above zero:

    P(home wins) = Phi( (margin + mu * f) / (sigma * sqrt(f)) )      f = share of the game still to play

mu (how many points better the home team is over a full game) comes from the pre-game Elo ratings, and
mu's scale and sigma are fitted on two seasons, then tested on a third season the model never saw.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "basketball"
GAME_SECONDS = 2880                      # 4 quarters x 12 minutes
TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets", "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons", "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards"}


# ------------------------------------------------------------------ cleaning one season of play-by-play
def clock_seconds(clock):
    """'PT11M41.00S' -> 701.0 seconds left in the period."""
    parts = clock.astype(str).str.extract(r"PT(\d+)M([\d.]+)S").astype(float)
    return parts[0] * 60 + parts[1]


def elapsed(period, left):
    """Seconds played since tip-off (overtime periods are 5 minutes)."""
    period = np.asarray(period)
    reg = np.minimum(period, 4)
    return np.where(period <= 4, (reg - 1) * 720 + (720 - left), GAME_SECONDS + (period - 5) * 300 + (300 - left))


def season_label(year):
    return f"{year}-{str(year + 1)[-2:]}"


def clean_season(pbp, year, names):
    d = pbp.copy()
    d = d[d.gameId.notna()]
    d["left"] = clock_seconds(d.clock)
    d["t"] = elapsed(d.period.to_numpy(), d.left.to_numpy())
    d["scoreHome"] = pd.to_numeric(d.scoreHome, errors="coerce")
    d["scoreAway"] = pd.to_numeric(d.scoreAway, errors="coerce")
    d = d.sort_values(["gameId", "orderNumber"])
    d[["scoreHome", "scoreAway"]] = d.groupby("gameId")[["scoreHome", "scoreAway"]].ffill().fillna(0)

    # which tricode is home: the team whose baskets raise scoreHome
    step = d.groupby("gameId")[["scoreHome", "scoreAway"]].diff().fillna(0)
    scored_home = d[(step.scoreHome > 0) & d.teamTricode.notna()]
    scored_away = d[(step.scoreAway > 0) & d.teamTricode.notna()]
    home = scored_home.groupby("gameId").teamTricode.agg(lambda s: s.mode().iat[0])
    away = scored_away.groupby("gameId").teamTricode.agg(lambda s: s.mode().iat[0])
    last = d.groupby("gameId").agg(home_score=("scoreHome", "max"), away_score=("scoreAway", "max"),
                                   periods=("period", "max"), start=("timeActual", "min"))
    games = last.join(home.rename("home")).join(away.rename("away")).dropna(subset=["home", "away"]).reset_index()
    games["date"] = (pd.to_datetime(games.start, utc=True, format="mixed") - pd.Timedelta(hours=8)).dt.tz_localize(None).dt.normalize()
    games["season"] = season_label(year)
    games["overtime"] = games.periods > 4
    games["home_won"] = games.home_score > games.away_score
    games = games.drop(columns=["start"]).rename(columns={"gameId": "game_id"})

    # scoring states: every change of score
    sc = d[(step.scoreHome > 0) | (step.scoreAway > 0)][["gameId", "period", "t", "scoreHome", "scoreAway", "teamTricode",
                                                         "personId", "actionType"]]
    states = sc.rename(columns={"gameId": "game_id", "scoreHome": "home_score", "scoreAway": "away_score",
                                "teamTricode": "team", "personId": "player_id", "actionType": "kind"})

    # box score per player per game
    made = d.shotResult.eq("Made")
    fg = d.actionType.isin(["2pt", "3pt"])
    pts = np.select([fg & made & d.actionType.eq("3pt"), fg & made, d.actionType.eq("freethrow") & made], [3, 2, 1], 0)
    d["pts"], d["fgm"], d["fga"] = pts, (fg & made).astype(int), fg.astype(int)
    d["fg3m"] = (made & d.actionType.eq("3pt")).astype(int)
    d["fg3a"] = d.actionType.eq("3pt").astype(int)
    d["reb"] = (d.actionType.eq("rebound") & (d.personId > 0)).astype(int)
    d["stl"] = d.actionType.eq("steal").astype(int)
    d["blk"] = d.actionType.eq("block").astype(int)
    d["tov"] = (d.actionType.eq("turnover") & (d.personId > 0)).astype(int)
    stats = ["pts", "fgm", "fga", "fg3m", "fg3a", "reb", "stl", "blk", "tov"]
    box = d[d.personId > 0].groupby(["gameId", "personId", "teamTricode"])[stats].sum().reset_index()
    ast = d[made & fg & d.assistPersonId.notna() & (d.assistPersonId > 0)].groupby(["gameId", "assistPersonId"]).size()
    ast.index = ast.index.set_names(["gameId", "personId"])
    box = box.merge(ast.rename("ast").reset_index(), on=["gameId", "personId"], how="left").fillna({"ast": 0})
    box["ast"] = box.ast.astype(int)
    box = box.rename(columns={"gameId": "game_id", "personId": "player_id", "teamTricode": "team"})
    box["player"] = box.player_id.map(names)
    fallback = d.dropna(subset=["playerNameI"]).groupby("personId").playerNameI.first()
    box["player"] = box.player.fillna(box.player_id.map(fallback))
    box["season"] = season_label(year)

    shots = d[fg][["gameId", "personId", "teamTricode", "xLegacy", "yLegacy", "shotDistance", "actionType", "shotResult",
                   "period"]].rename(columns={"gameId": "game_id", "personId": "player_id", "teamTricode": "team",
                                              "xLegacy": "x", "yLegacy": "y", "shotDistance": "distance"})
    shots["made"] = shots.pop("shotResult").eq("Made")
    shots["three"] = shots.pop("actionType").eq("3pt")
    shots["season"] = season_label(year)
    return games, states, box, shots


# ------------------------------------------------------------------ Elo (FiveThirtyEight NBA rules)
def add_elo(games, k=20, home_adv=100, carry=0.75, mean=1505):
    g = games.sort_values(["date", "game_id"]).reset_index(drop=True).copy()
    rating, season = {}, None
    eh, ea = np.empty(len(g)), np.empty(len(g))
    for i, r in enumerate(g.itertuples()):
        if r.season != season:                       # new season: pull every team a quarter of the way back
            rating = {t: carry * v + (1 - carry) * mean for t, v in rating.items()}
            season = r.season
        rh, ra = rating.get(r.home, mean), rating.get(r.away, mean)
        eh[i], ea[i] = rh, ra
        dr = rh + home_adv - ra
        expected = 1 / (1 + 10 ** (-dr / 400))
        mov = abs(r.home_score - r.away_score)
        win_dr = dr if r.home_won else -dr
        mult = (mov + 3) ** 0.8 / (7.5 + 0.006 * win_dr)
        change = k * mult * ((1.0 if r.home_won else 0.0) - expected)
        rating[r.home], rating[r.away] = rh + change, ra - change
    g["elo_home"], g["elo_away"] = eh, ea
    return g


def ratings_now(games, k=20, home_adv=100):
    rating = {}
    for r in games.sort_values(["date", "game_id"]).itertuples():
        rating.setdefault(r.home, r.elo_home)
        rating.setdefault(r.away, r.elo_away)
        rh, ra = r.elo_home, r.elo_away
        dr = rh + home_adv - ra
        expected = 1 / (1 + 10 ** (-dr / 400))
        mov = abs(r.home_score - r.away_score)
        mult = (mov + 3) ** 0.8 / (7.5 + 0.006 * (dr if r.home_won else -dr))
        change = k * mult * ((1.0 if r.home_won else 0.0) - expected)
        rating[r.home], rating[r.away] = rh + change, ra - change
    return pd.Series(rating).sort_values(ascending=False)


# ------------------------------------------------------------------ win probability
def share_left(t, period):
    """Share of a regulation game still to play (in overtime: the share of an OT period, scaled the same way)."""
    t, period = np.asarray(t, float), np.asarray(period)
    left = np.where(period <= 4, GAME_SECONDS - t, (GAME_SECONDS + (period - 4) * 300) - t)
    return np.clip(left, 0, None) / GAME_SECONDS


def win_prob(params, margin, f, elo_gap):
    """params = (points per 100 Elo, sigma, home edge in points). elo_gap = home Elo - away Elo."""
    per100, sigma, home_pts = params
    mu = per100 * np.asarray(elo_gap) / 100 + home_pts           # expected full-game margin for the home team
    f = np.maximum(np.asarray(f, float), 1e-4)
    return norm.cdf((np.asarray(margin) + mu * f) / (sigma * np.sqrt(f)))


def fit(states):
    """Maximum-likelihood fit of the three parameters on game states."""
    from scipy.optimize import minimize
    y = states.home_won.to_numpy(float)
    m, f, gap = states.margin.to_numpy(float), states.f.to_numpy(float), states.elo_gap.to_numpy(float)

    def nll(p):
        q = np.clip(win_prob(p, m, f, gap), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    res = minimize(nll, x0=[3.5, 12.0, 2.5], method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-3})
    return tuple(float(v) for v in res.x)


def model_states(games, states):
    """Training rows: the tip-off plus every change of score, with the final result attached."""
    s = states.merge(games[["game_id", "home_won", "elo_home", "elo_away", "season"]], on="game_id")
    start = games[["game_id", "home_won", "elo_home", "elo_away", "season"]].assign(period=1, t=0.0, home_score=0, away_score=0)
    s = pd.concat([start, s], ignore_index=True)
    s["margin"] = s.home_score - s.away_score
    s["f"] = share_left(s.t, s.period)
    s["elo_gap"] = s.elo_home - s.elo_away
    return s


def worm(params, game, states):
    """Home team's chance of winning after every score of one game."""
    s = states[states.game_id == game.game_id].sort_values("t")
    s = pd.concat([pd.DataFrame({"t": [0.0], "period": [1], "home_score": [0], "away_score": [0], "team": [""],
                                 "player_id": [0], "kind": [""]}), s], ignore_index=True)
    s["margin"] = s.home_score - s.away_score
    p = win_prob(params, s.margin, share_left(s.t, s.period), game.elo_home - game.elo_away)
    last = s.index[-1]
    p = np.asarray(p, float)
    p[last] = 1.0 if game.home_won else 0.0
    s["home_win"] = p
    s["minute"] = s.t / 60
    return s
