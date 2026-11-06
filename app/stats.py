"""Cricket statistics built from the clean ball-by-ball table. Pure pandas, so they're easy to test."""
import numpy as np
import pandas as pd

import ipl_data


def seasons(df, lo, hi):
    return df[df.season.between(lo, hi)]


def season_summary(matches, balls):
    """One row per season: scoring, 200+ totals, sixes, chase and toss numbers."""
    inn = balls.groupby(["match_id", "inning", "season"]).agg(runs=("total_runs", "sum")).reset_index()
    first = inn[inn.inning == 1].merge(matches[["match_id", "dls", "no_result"]], on="match_id")
    first = first[~first.dls & ~first.no_result]
    out = pd.DataFrame({
        "matches": matches.groupby("season").size(),
        "avg_first_innings": first.groupby("season").runs.mean(),
        "totals_200": inn[inn.runs >= 200].groupby("season").size(),
        "sixes_per_match": balls.groupby("season").six.sum() / matches.groupby("season").size(),
    }).fillna({"totals_200": 0})
    dec = matches[~matches.no_result & matches.winner.notna() & ~matches.super_over]
    out["chase_win_pct"] = dec.groupby("season").chasing_won.apply(lambda s: s.astype(bool).mean() * 100)
    out["field_first_pct"] = matches.groupby("season").toss_decision.apply(lambda s: (s == "field").mean() * 100)
    champs = matches[matches.match_type == "Final"].set_index("season").winner_short
    out["champion"] = champs
    return out.reset_index().rename(columns={"index": "season"})


def phase_rates(balls):
    g = balls.groupby(["season", "phase"], observed=True).agg(runs=("total_runs", "sum"), legal=("legal", "sum"))
    g["run_rate"] = g.runs / g.legal * 6
    return g.reset_index()


def venues(matches, min_matches=10):
    dec = matches[~matches.no_result & matches.winner.notna() & ~matches.super_over]
    v = dec.groupby("venue").agg(matches=("match_id", "size"),
                                 chase_win_pct=("chasing_won", lambda s: s.astype(bool).mean() * 100),
                                 field_first_pct=("toss_decision", lambda s: (s == "field").mean() * 100))
    v["city"] = v.index.map(ipl_data.VENUE_CITY)
    return v[v.matches >= min_matches].sort_values("chase_win_pct")


def team_games(matches):
    """Two rows per decided match: one for each team, with whether it won."""
    dec = matches[~matches.no_result & matches.winner.notna()]
    rows = []
    for side, opp in (("team1", "team2"), ("team2", "team1")):
        part = dec[["match_id", "season", "date", "venue", "match_type", side, opp, "winner", "result",
                    "result_margin"]].rename(columns={side: "team", opp: "opponent"})
        rows.append(part)
    g = pd.concat(rows, ignore_index=True)
    g["won"] = g.team == g.winner
    return g


def team_record(matches):
    g = team_games(matches)
    rec = g.groupby("team").agg(played=("won", "size"), won=("won", "sum"), first=("season", "min"),
                                last=("season", "max"))
    rec["win_pct"] = rec.won / rec.played * 100
    finals = matches[matches.match_type == "Final"]
    rec["titles"] = finals.winner.value_counts().reindex(rec.index).fillna(0).astype(int)
    rec["short"] = rec.index.map(ipl_data.SHORT)
    return rec.sort_values("win_pct", ascending=False)


def _plain(obj):
    """Player names as a plain string index. Categorical indexes from different subsets can carry different
    internal code sizes, and pandas then fails to line them up (a 'Buffer dtype mismatch' error)."""
    obj = obj.copy()
    obj.index = obj.index.astype(str)
    return obj


def batting(balls, min_balls=0):
    bf = balls[balls.faced]
    bat = _plain(bf.groupby("batter", observed=True).agg(runs=("batter_runs", "sum"), balls=("faced", "sum"),
                                                         fours=("four", "sum"), sixes=("six", "sum"),
                                                         matches=("match_id", "nunique")))
    outs = _plain(balls[balls.wicket].groupby("player_dismissed", observed=True).size())
    bat["outs"] = outs.reindex(bat.index).fillna(0).astype(int)
    per_inns = bf.groupby(["batter", "match_id"], observed=True).batter_runs.sum()
    per_inns.index = per_inns.index.set_levels(per_inns.index.levels[0].astype(str), level=0)
    by_player = per_inns.groupby(level=0)
    bat["fifties"] = per_inns[(per_inns >= 50) & (per_inns < 100)].groupby(level=0).size().reindex(bat.index).fillna(0).astype(int)
    bat["hundreds"] = per_inns[per_inns >= 100].groupby(level=0).size().reindex(bat.index).fillna(0).astype(int)
    bat["best"] = by_player.max().reindex(bat.index)
    bat["strike_rate"] = bat.runs / bat.balls * 100
    bat["average"] = bat.runs / bat.outs.replace(0, np.nan)
    bat.index.name = "batter"
    return bat[bat.balls >= min_balls].sort_values("runs", ascending=False)


def bowling(balls, min_balls=0):
    bw = _plain(balls.groupby("bowler", observed=True).agg(balls=("legal", "sum"), runs=("bowler_runs", "sum"),
                                                           wickets=("bowler_wicket", "sum"), dots=("dot", "sum"),
                                                           fours=("four", "sum"), sixes=("six", "sum"),
                                                           matches=("match_id", "nunique")))
    bw["economy"] = bw.runs / bw.balls.replace(0, np.nan) * 6
    bw["average"] = bw.runs / bw.wickets.replace(0, np.nan)
    bw["strike_rate"] = bw.balls / bw.wickets.replace(0, np.nan)
    bw["dot_pct"] = bw.dots / bw.balls.replace(0, np.nan) * 100
    bw["boundary_pct"] = (bw.fours + bw.sixes) / bw.balls.replace(0, np.nan) * 100
    figs = balls.groupby(["bowler", "match_id"], observed=True).agg(w=("bowler_wicket", "sum"), r=("bowler_runs", "sum"))
    best = figs.sort_values(["w", "r"], ascending=[False, True]).groupby(level=0, observed=True).head(1)
    best = _plain(best.reset_index().set_index("bowler"))
    bw["best"] = (best.w.astype(int).astype(str) + "/" + best.r.astype(int).astype(str)).reindex(bw.index)
    bw.index.name = "bowler"
    return bw[bw.balls >= min_balls].sort_values("wickets", ascending=False)


def by_phase(balls, player, role):
    if role == "bat":
        b = balls[(balls.batter == player) & balls.faced]
        g = b.groupby("phase", observed=True).agg(runs=("batter_runs", "sum"), balls=("faced", "sum"))
        g["strike_rate"] = g.runs / g.balls * 100
    else:
        b = balls[balls.bowler == player]
        g = b.groupby("phase", observed=True).agg(runs=("bowler_runs", "sum"), balls=("legal", "sum"),
                                                  wickets=("bowler_wicket", "sum"))
        g["economy"] = g.runs / g.balls * 6
    return g.reset_index()


def by_season(balls, player, role):
    if role == "bat":
        b = balls[(balls.batter == player) & balls.faced]
        g = b.groupby("season").agg(runs=("batter_runs", "sum"), balls=("faced", "sum"))
        g["strike_rate"] = g.runs / g.balls * 100
    else:
        b = balls[balls.bowler == player]
        g = b.groupby("season").agg(wickets=("bowler_wicket", "sum"), runs=("bowler_runs", "sum"),
                                    balls=("legal", "sum"))
        g["economy"] = g.runs / g.balls * 6
    return g.reset_index()


def matchup(balls, batter, bowler):
    b = balls[(balls.batter == batter) & (balls.bowler == bowler)]
    faced = b[b.faced]
    outs = int((b.bowler_wicket & (b.player_dismissed == batter)).sum())
    return {"balls": int(len(faced)), "runs": int(faced.batter_runs.sum()), "outs": outs,
            "strike_rate": faced.batter_runs.sum() / len(faced) * 100 if len(faced) else np.nan,
            "dots": int(faced["dot"].sum()), "boundaries": int((faced["four"] | faced["six"]).sum())}


def scorecard_totals(balls, match_id):
    b = balls[balls.match_id == match_id]
    t = b.groupby(["inning", "batting_team"], observed=True).agg(runs=("total_runs", "sum"), wkts=("wicket", "sum"),
                                                                 legal=("legal", "sum")).reset_index()
    t["overs"] = (t.legal // 6).astype(str) + "." + (t.legal % 6).astype(str)
    return t


def over_by_over(balls, match_id):
    b = balls[balls.match_id == match_id]
    g = b.groupby(["inning", "batting_team", "over"], observed=True).agg(runs=("total_runs", "sum"),
                                                                         wickets=("wicket", "sum")).reset_index()
    g["cumulative"] = g.groupby("inning").runs.cumsum()
    return g
