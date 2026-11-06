"""Ball-by-ball win probability for the team chasing a target.

Each row is the state of a chase just before a legal ball: runs needed, balls left, wickets left, the target,
and whether it is the Impact Player era (2023+). The label is whether the chasing team went on to win.
"""
import numpy as np
import pandas as pd

FEATURES = ["runs_needed", "balls_left", "wickets_left", "target", "impact_era"]


def chase_states(matches, balls):
    """One row per legal ball of every normal chase (no D/L target, no super over, a result)."""
    ok = matches[~matches.dls & ~matches.super_over & ~matches.no_result & matches.winner.notna()]
    second = balls[(balls.inning == 2) & balls.match_id.isin(ok.match_id)].copy()
    first_total = balls[balls.inning == 1].groupby("match_id").total_runs.sum()
    second["target"] = second.match_id.map(first_total + 1)
    g = second.groupby("match_id", sort=False)
    # state BEFORE each ball: subtract the ball's own runs, wickets and legality from the running totals
    second["runs_before"] = g.total_runs.cumsum() - second.total_runs
    second["wkts_before"] = g.wicket.cumsum() - second.wicket.astype(int)
    second["legal_before"] = g.legal.cumsum() - second.legal.astype(int)
    states = second[second.legal].copy()
    states["runs_needed"] = states.target - states.runs_before
    states["balls_left"] = 120 - states.legal_before
    states["wickets_left"] = 10 - states.wkts_before
    states["impact_era"] = states.season >= 2023
    won = ok.set_index("match_id").chasing_won.astype(bool)
    states["chase_won"] = states.match_id.map(won).astype(int)
    keep = ["match_id", "season", "batting_team", "bowling_team", "over", "ball"] + FEATURES + ["chase_won"]
    return states[keep].reset_index(drop=True)


def add_rates(X):
    """Derived features the model also sees: required and current run rate."""
    X = X.copy()
    X["required_rate"] = X["runs_needed"] / np.maximum(X["balls_left"], 1) * 6
    played = 120 - X["balls_left"]
    X["current_rate"] = np.where(played > 0, (X["target"] - X["runs_needed"]) / np.maximum(played, 1) * 6, 0.0)
    X["impact_era"] = X["impact_era"].astype(int)
    return X


def probability(model, runs_needed, balls_left, wickets_left, target, impact_era=True):
    """Chance the chasing team wins from one game state (handles the finished cases exactly)."""
    if runs_needed <= 0:
        return 1.0
    if balls_left <= 0 or wickets_left <= 0:
        return 0.0
    row = pd.DataFrame([dict(runs_needed=runs_needed, balls_left=balls_left, wickets_left=wickets_left,
                             target=target, impact_era=impact_era)])
    return float(model.predict_proba(row)[:, 1][0])


def worm(model, match_balls, target, impact_era):
    """Win probability for the chasing team after every ball of the chase (for the replay chart)."""
    ch = match_balls[match_balls.inning == 2].copy()
    ch["runs"] = ch.total_runs.cumsum()
    ch["wkts"] = ch.wicket.cumsum()
    ch["legal_n"] = ch.legal.cumsum()
    X = pd.DataFrame({"runs_needed": target - ch.runs, "balls_left": 120 - ch.legal_n,
                      "wickets_left": 10 - ch.wkts, "target": target, "impact_era": impact_era})
    p = model.predict_proba(X)[:, 1]
    p = np.where(X.runs_needed <= 0, 1.0, np.where((X.balls_left <= 0) | (X.wickets_left <= 0), 0.0, p))
    start = probability(model, target, 120, 10, target, impact_era)
    out = pd.DataFrame({"ball_no": np.arange(1, len(ch) + 1), "over": ch.over.values, "ball": ch.ball.values,
                        "batter": ch.batter.astype(str).values, "bowler": ch.bowler.astype(str).values,
                        "runs": ch.total_runs.values, "wicket": ch.wicket.values, "score": ch.runs.values,
                        "wkts": ch.wkts.values, "win_prob": p})
    return start, out


def features(X):
    """Everything the model looks at, built from the five raw inputs."""
    X = add_rates(X)
    X["rr_x_wickets_lost"] = X["required_rate"] * (10 - X["wickets_left"])   # a high rate hurts more with few wickets
    X["need_per_wicket"] = X["runs_needed"] / X["wickets_left"].clip(lower=1)
    return X


def make_model():
    """Logistic regression on spline features: simple and well calibrated, but able to learn curves
    (a required rate of 14 is far worse than 12, not just a bit worse)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, SplineTransformer, StandardScaler
    curves = ["required_rate", "wickets_left", "balls_left", "need_per_wicket"]
    return make_pipeline(
        FunctionTransformer(features),
        ColumnTransformer([("curves", SplineTransformer(n_knots=6, degree=3), curves),
                           ("linear", StandardScaler(), ["rr_x_wickets_lost", "impact_era"])]),
        LogisticRegression(max_iter=5000, C=0.5))


# Real chase situations used to sanity-check the model against what actually happened
SITUATIONS = [("Need 10 off 6, 2 wickets left", (9, 11), 6, (1, 3), (10, 6, 2)),
              ("Need 10 off 6, 5 wickets left", (9, 11), 6, (4, 7), (10, 6, 5)),
              ("Need 30 off 18, 5 wickets left", (28, 32), 18, (4, 6), (30, 18, 5)),
              ("Need 60 off 36, 6 wickets left", (55, 65), 36, (5, 7), (60, 36, 6)),
              ("Need 16 off 12, 7 wickets left", (15, 17), 12, (6, 8), (16, 12, 7))]
