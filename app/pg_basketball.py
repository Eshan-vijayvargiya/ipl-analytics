"""Basketball section: NBA 2021-22 to 2023-24, every play."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import basketball as bb
import style
from ui import AQUA, BLUE, GOLD, MUTED, ORANGE, chart, pct, prob_bar, two_way_worm

ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner="Loading three NBA seasons…")
def load():
    g = pd.read_parquet(bb.DATA / "games.parquet")
    s = pd.read_parquet(bb.DATA / "states.parquet")
    b = pd.read_parquet(bb.DATA / "box.parquet")
    return g, s, b


@st.cache_data(show_spinner=False)
def shots():
    return pd.read_parquet(bb.DATA / "shots.parquet")


@st.cache_resource
def model():
    return joblib.load(ROOT / "models" / "basketball_model.joblib")["params"], json.loads((ROOT / "models" / "basketball_info.json").read_text())


@st.cache_data(show_spinner=False)
def names():
    _, _, b = load()
    return b.dropna(subset=["player"]).groupby("player_id").player.first()


def team_label(t):
    return f"{bb.TEAM_NAMES.get(t, t)}"


def game_label(r):
    return f"{r.date:%d %b %Y} · {r.away} {r.away_score} @ {r.home} {r.home_score}" + (" (OT)" if r.overtime else "")


@st.cache_data(show_spinner=False)
def standings(season):
    g, _, _ = load()
    d = g[g.season == season]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"team": d[side], "won": d.home_won if side == "home" else ~d.home_won,
                                  "pf": d[f"{side}_score"], "pa": d[f"{opp}_score"], "where": side}))
    t = pd.concat(rows)
    s = t.groupby("team").agg(W=("won", "sum"), G=("won", "size"), pf=("pf", "mean"), pa=("pa", "mean"))
    s["L"] = s.G - s.W
    s["Win %"] = s.W / s.G * 100
    s["Net"] = s.pf - s.pa
    s["Team"] = s.index.map(team_label)
    return s.sort_values("Win %", ascending=False).reset_index()


def court_layers():
    """Half-court lines in NBA shot-chart units (tenths of a foot, hoop at 0,0)."""
    def arc(r, a0, a1, cx=0, cy=0, n=60):
        a = np.linspace(a0, a1, n)
        return pd.DataFrame({"x": cx + r * np.cos(a), "y": cy + r * np.sin(a)})
    parts = {
        "outer": pd.DataFrame({"x": [-250, -250, 250, 250, -250], "y": [-47.5, 422.5, 422.5, -47.5, -47.5]}),
        "paint": pd.DataFrame({"x": [-80, -80, 80, 80], "y": [-47.5, 142.5, 142.5, -47.5]}),
        "ft": arc(60, 0, np.pi, 0, 142.5),
        "rim": arc(7.5, 0, 2 * np.pi),
        "board": pd.DataFrame({"x": [-30, 30], "y": [-7.5, -7.5]}),
        "ra": arc(40, 0, np.pi),
        "three": pd.concat([pd.DataFrame({"x": [-220, -220], "y": [-47.5, 237.5 * np.sin(np.arccos(220 / 237.5))]}),
                            arc(237.5, np.pi - np.arccos(220 / 237.5), np.arccos(220 / 237.5)),
                            pd.DataFrame({"x": [220, 220], "y": [237.5 * np.sin(np.arccos(220 / 237.5)), -47.5]})]),
        "centre": arc(60, np.pi, 2 * np.pi, 0, 422.5),
    }
    layers = None
    for k, d in parts.items():
        d = d.reset_index(drop=True).assign(order=lambda x: np.arange(len(x)))
        l = alt.Chart(d).mark_line(color="#3a4a6e", strokeWidth=1.4).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[-250, 250]), axis=None),
                                                                            y=alt.Y("y:Q", scale=alt.Scale(domain=[-52, 422.5]), axis=None),
                                                                            order="order:Q")
        layers = l if layers is None else layers + l
    return layers


# ------------------------------------------------------------------ overview
def overview():
    g, s, b = load()
    params, info = model()
    style.hero("What decides an NBA game?", "Three seasons, 3,690 games and every play: how much home court is worth, why "
               "everyone shoots threes, and how safe a lead really is.", eyebrow="🏀 Basketball · NBA 2021-22 to 2023-24")
    sh = shots()
    style.kpis([("Games", f"{len(g):,}", f"{len(s):,} scoring plays"), ("Points a game", f"{(g.home_score + g.away_score).mean():.0f}", "both teams"),
                ("Home team won", pct(g.home_won.mean() * 100), "home court is worth ~3.5 points", "gold"),
                ("Went to overtime", pct(g.overtime.mean() * 100, 1), f"{int(g.overtime.sum())} games"),
                ("Shots that were threes", pct(sh.three.mean() * 100), "2023-24")])
    t1, t2, t3, t4 = st.tabs(["Why everyone shoots threes", "How safe is a lead?", "Best teams", "Biggest nights"])
    with t1:
        d = sh.assign(ft=sh.distance.clip(0, 30).round().astype(int))
        d = d[d.distance <= 30].groupby("ft").agg(att=("made", "size"), made=("made", "sum"), three=("three", "mean"))
        d["pps"] = (d.made * np.where(d.three > 0.5, 3, 2)) / d.att
        d = d[d.att >= 300].reset_index()
        d["Shot"] = np.where(d.three > 0.5, "Three", "Two")
        c = alt.Chart(d).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=16).encode(
            x=alt.X("ft:O", title="Shot distance (feet)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pps:Q", title="Points per shot"), color=alt.Color("Shot:N", scale=alt.Scale(domain=["Two", "Three"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("ft:O", title="Feet"), alt.Tooltip("pps:Q", format=".2f", title="Points per shot"), alt.Tooltip("att:Q", format=",", title="Shots")])
        rule = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
        chart(c + rule, 300)
        two, three = sh[~sh.three], sh[sh.three]
        style.card("The maths behind the three-point boom", f"""<p>A two goes in {two.made.mean():.1%} of the time and a three
        {three.made.mean():.1%}, so each attempt is worth about the same: <b>{two.made.mean() * 2:.2f} vs {three.made.mean() * 3:.2f}
        points</b>. But that average two includes dunks and layups. The <b>mid-range jumper (10-22 feet) returns well under 1 point a
        shot</b>, which is why teams now take shots at the rim or behind the arc, and almost nothing in between.</p>""", insight=True)
    with t2:
        st2 = s.merge(g[["game_id", "home_won"]], on="game_id")
        rows = []
        for label, t_cut in (("Half-time", 1440), ("Start of 4th quarter", 2160), ("5 minutes left", 2580)):
            snap = st2[st2.t <= t_cut].groupby("game_id").last()
            snap["lead"] = (snap.home_score - snap.away_score).abs()
            snap["leader_won"] = np.where(snap.home_score > snap.away_score, snap.home_won, ~snap.home_won)
            snap = snap[snap.lead > 0]
            snap["bucket"] = pd.cut(snap.lead, [0, 3, 6, 10, 15, 20, 100], labels=["1-3", "4-6", "7-10", "11-15", "16-20", "21+"])
            r = snap.groupby("bucket", observed=True).leader_won.agg(["mean", "size"]).reset_index()
            r["when"] = label
            rows.append(r)
        d = pd.concat(rows)
        d["Won"] = d["mean"] * 100
        c = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.5).encode(
            x=alt.X("bucket:N", sort=None, title="Lead (points)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Won:Q", title="Team in front went on to win (%)", scale=alt.Scale(domain=[40, 100])),
            color=alt.Color("when:N", sort=["Half-time", "Start of 4th quarter", "5 minutes left"],
                            scale=alt.Scale(range=[BLUE, AQUA, ORANGE]), legend=alt.Legend(title=None)),
            tooltip=["when:N", "bucket:N", alt.Tooltip("Won:Q", format=".0f"), alt.Tooltip("size:Q", title="Games")])
        chart(c, 300)
        style.card("No lead is safe early on", "<p>A 10-point lead at half-time wins about <b>3 games in 4</b>; the same lead with "
                   "5 minutes left wins <b>97%</b>. Time left matters as much as the score: that is exactly what the "
                   "win-probability model learns.</p>", insight=True)
    with t3:
        season = st.segmented_control("Season", sorted(g.season.unique()), default="2023-24", key="bb_ov_season") or "2023-24"
        t = standings(season)
        st.dataframe(t[["Team", "W", "L", "Win %", "pf", "pa", "Net"]], hide_index=True, width="stretch", height=420,
                     column_config={"Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.0f%%"),
                                    "pf": st.column_config.NumberColumn("Points for", format="%.1f"),
                                    "pa": st.column_config.NumberColumn("Points against", format="%.1f"),
                                    "Net": st.column_config.NumberColumn("Net", format="%+.1f", help="Average winning margin")})
    with t4:
        top = b.sort_values("pts", ascending=False).head(15).merge(g[["game_id", "date", "home", "away"]], on="game_id")
        top["Opponent"] = np.where(top.team == top.home, top.away, "@ " + top.home)
        st.dataframe(top[["player", "team", "pts", "reb", "ast", "fg3m", "date", "Opponent"]], hide_index=True, width="stretch",
                     column_config={"player": "Player", "team": "Team", "pts": "Points", "reb": "Rebounds", "ast": "Assists",
                                    "fg3m": "Threes", "date": st.column_config.DateColumn("Date", format="D MMM YYYY")})


# ------------------------------------------------------------------ teams
def teams():
    g, s, b = load()
    a, c = st.columns([2, 1])
    team = a.selectbox("Team", sorted(bb.TEAM_NAMES), index=sorted(bb.TEAM_NAMES).index("BOS"), format_func=team_label, key="bb_team")
    season = c.selectbox("Season", sorted(g.season.unique(), reverse=True), key="bb_team_season")
    style.hero(team_label(team), f"Season {season}", eyebrow="🏀 Team profile")
    st_ = standings(season)
    row = st_[st_.team == team].iloc[0]
    rank = int(st_.index[st_.team == team][0]) + 1
    gt = g[(g.season == season) & ((g.home == team) | (g.away == team))].sort_values("date")
    home = gt[gt.home == team]
    away = gt[gt.away == team]
    style.kpis([("Record", f"{int(row.W)}-{int(row.L)}", f"No. {rank} of 30 in the league", "gold"),
                ("Home", f"{int(home.home_won.sum())}-{int((~home.home_won).sum())}", "won-lost"),
                ("Away", f"{int((~away.home_won).sum())}-{int(away.home_won.sum())}", "won-lost"),
                ("Points for / against", f"{row.pf:.1f} / {row.pa:.1f}", f"net {row.Net:+.1f} a game")])
    rows = []
    for r in g[(g.home == team) | (g.away == team)].sort_values("date").itertuples():
        rows.append({"date": r.date, "elo": r.elo_home if r.home == team else r.elo_away, "season": r.season,
                     "opp": r.away if r.home == team else r.home,
                     "result": ("W" if (r.home == team) == r.home_won else "L") + f" {max(r.home_score, r.away_score)}-{min(r.home_score, r.away_score)}"})
    e = pd.DataFrame(rows)
    st.markdown("##### Elo rating, game by game (three seasons)")
    chart(alt.Chart(e).mark_line(color=GOLD, strokeWidth=1.8).encode(
        x=alt.X("date:T", title=None), y=alt.Y("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), "opp:N", "result:N", alt.Tooltip("elo:Q", format=".0f")]), 240)
    bx = b[(b.team == team) & (b.season == season)]
    pl = bx.groupby("player").agg(G=("game_id", "nunique"), pts=("pts", "sum"), reb=("reb", "sum"), ast=("ast", "sum"), fg3m=("fg3m", "sum"))
    for k in ("pts", "reb", "ast"):
        pl[k] = pl[k] / pl.G
    pl = pl[pl.G >= 10].sort_values("pts", ascending=False).head(10).reset_index()
    st.markdown("##### Top scorers")
    st.dataframe(pl, hide_index=True, width="stretch",
                 column_config={"player": "Player", "G": "Games", "pts": st.column_config.NumberColumn("Points", format="%.1f"),
                                "reb": st.column_config.NumberColumn("Rebounds", format="%.1f"),
                                "ast": st.column_config.NumberColumn("Assists", format="%.1f"), "fg3m": "Threes made"})


# ------------------------------------------------------------------ players
def players():
    g, s, b = load()
    season = st.segmented_control("Season", sorted(g.season.unique()), default="2023-24", key="bb_pl_season") or "2023-24"
    bx = b[b.season == season]
    pl = bx.groupby("player").agg(team=("team", lambda x: x.mode().iat[0]), G=("game_id", "nunique"), pts=("pts", "sum"),
                                  reb=("reb", "sum"), ast=("ast", "sum"), stl=("stl", "sum"), blk=("blk", "sum"),
                                  fg3m=("fg3m", "sum"), fgm=("fgm", "sum"), fga=("fga", "sum"))
    for k in ("pts", "reb", "ast", "stl", "blk"):
        pl[f"{k}_g"] = pl[k] / pl.G
    pl["fg"] = pl.fgm / pl.fga.replace(0, np.nan) * 100
    qual = pl[pl.G >= 40].sort_values("pts_g", ascending=False)
    names_ = qual.index.tolist()
    who = st.selectbox("Player", names_, index=0, key=f"bb_player_{season}")
    r = pl.loc[who]
    style.hero(who, f"{team_label(r.team)} · {season} · {int(r.G)} games", eyebrow="🏀 Player profile")
    style.kpis([("Points", f"{r.pts_g:.1f}", f"No. {names_.index(who) + 1} in the league", "gold"), ("Rebounds", f"{r.reb_g:.1f}", "a game"),
                ("Assists", f"{r.ast_g:.1f}", "a game"), ("Field goals", pct(r.fg, 1), f"{int(r.fg3m)} threes made"),
                ("Steals + blocks", f"{r.stl_g + r.blk_g:.1f}", "a game")])
    a, c = st.columns([1, 1])
    with a:
        log = bx[bx.player == who].merge(g[["game_id", "date", "home", "away"]], on="game_id").sort_values("date")
        log["opp"] = np.where(log.team == log.home, log.away, "@" + log.home)
        st.markdown("##### Points, game by game")
        base = alt.Chart(log).encode(x=alt.X("date:T", title=None))
        chart(base.mark_bar(color=BLUE, size=4).encode(y=alt.Y("pts:Q", title="Points"),
                                                       tooltip=[alt.Tooltip("date:T", format="%d %b"), "opp:N", "pts:Q", "reb:Q", "ast:Q"])
              + alt.Chart(pd.DataFrame({"y": [r.pts_g]})).mark_rule(color=GOLD, strokeDash=[4, 4]).encode(y="y:Q"), 300)
    with c:
        if season == "2023-24":
            pid = bx[bx.player == who].player_id.iat[0]
            sp = shots()
            sp = sp[(sp.player_id == pid) & (sp.y <= 420)]
            sp = sp.assign(Result=np.where(sp.made, "Made", "Missed"))
            st.markdown(f"##### Shot chart ({len(sp):,} shots)")
            dots = alt.Chart(sp).mark_circle(size=14, opacity=0.55).encode(
                x=alt.X("x:Q", scale=alt.Scale(domain=[-250, 250]), axis=None), y=alt.Y("y:Q", scale=alt.Scale(domain=[-52, 422.5]), axis=None),
                color=alt.Color("Result:N", scale=alt.Scale(domain=["Made", "Missed"], range=[AQUA, ORANGE]), legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("distance:Q", format=".0f", title="Feet"), "Result:N"])
            chart((court_layers() + dots).properties(width=380), 360)
        else:
            st.info("Shot charts are included for the 2023-24 season.")
    st.markdown("#### League leaders")
    lead = qual.reset_index()[["player", "team", "G", "pts_g", "reb_g", "ast_g", "stl_g", "blk_g", "fg"]].head(40)
    st.dataframe(lead, hide_index=True, width="stretch", height=360,
                 column_config={"player": "Player", "team": "Team", "G": "Games", "pts_g": st.column_config.NumberColumn("Points", format="%.1f"),
                                "reb_g": st.column_config.NumberColumn("Rebounds", format="%.1f"), "ast_g": st.column_config.NumberColumn("Assists", format="%.1f"),
                                "stl_g": st.column_config.NumberColumn("Steals", format="%.1f"), "blk_g": st.column_config.NumberColumn("Blocks", format="%.1f"),
                                "fg": st.column_config.NumberColumn("FG %", format="%.1f")})
    st.caption("Players with 40+ games, per-game averages.")


# ------------------------------------------------------------------ match replay
def replay():
    g, s, b = load()
    params, info = model()
    style.hero("Game replay", "Pick any game and watch the home team's chance of winning move after every basket.",
               eyebrow="🏀 Play-by-play win probability")
    a, c, d = st.columns([0.8, 1.4, 2.6])
    seasons = sorted(g.season.unique(), reverse=True)
    season = a.selectbox("Season", seasons, key="bb_rp_season")
    team = c.selectbox("Team", ["Any team"] + sorted(bb.TEAM_NAMES), index=sorted(bb.TEAM_NAMES).index("DAL") + 1 if season == "2023-24" else 0,
                       format_func=lambda t: t if t == "Any team" else team_label(t), key=f"bb_rp_team_{season}")
    gs = g[g.season == season]
    if team != "Any team":
        gs = gs[(gs.home == team) | (gs.away == team)]
    gs = gs.sort_values("date", ascending=False)
    labels = [game_label(r) for r in gs.itertuples()]
    default = 0
    if season == "2023-24" and team == "DAL":
        hit = [i for i, l in enumerate(labels) if l.startswith("26 Jan 2024")]
        default = hit[0] if hit else 0
    pick = d.selectbox("Game", labels, index=default, key=f"bb_rp_game_{season}_{team}")
    game = gs.iloc[labels.index(pick)]
    w = bb.worm(params, game, s)
    nm = names()
    w["who"] = w.player_id.map(nm).fillna("")
    w["play"] = w.kind.map({"2pt": "2", "3pt": "3", "freethrow": "free throw"}).fillna("")
    w["pct"] = w.home_win * 100
    w["score"] = game.away + " " + w.away_score.astype(str) + " - " + w.home_score.astype(str) + " " + game.home
    w["clock"] = np.where(w.period <= 4, "Q" + w.period.astype(str), "OT" + (w.period - 4).astype(str)) + " " + \
        ((np.where(w.period <= 4, w.period * 720, 2880 + (w.period - 4) * 300) - w.t) // 60).astype(int).astype(str) + ":" + \
        ((np.where(w.period <= 4, w.period * 720, 2880 + (w.period - 4) * 300) - w.t) % 60).astype(int).map("{:02d}".format)
    top = b[b.game_id == game.game_id].sort_values("pts", ascending=False).head(3)
    stars = ", ".join(f"{r.player} {r.pts} pts" for r in top.itertuples())
    style.card(f"{team_label(game.away)} {game.away_score} @ {team_label(game.home)} {game.home_score}",
               f"<p>{game.date:%A %d %B %Y}{' · overtime' if game.overtime else ''} · top scorers: {stars}</p>"
               f"<p>Before tip-off: Elo {game.elo_home:.0f} (home) vs {game.elo_away:.0f} → {game.home} "
               f"<b>{w.home_win.iloc[0]:.0%}</b> to win</p>")
    tips = [alt.Tooltip("clock:N", title="Clock"), alt.Tooltip("score:N", title="Score"), alt.Tooltip("who:N", title="Scored by"),
            alt.Tooltip("play:N", title="Shot"), alt.Tooltip("pct:Q", format=".0f", title=f"{game.home} win %")]
    two_way_worm(w, "minute", "pct", "Minutes played", f"{game.home}'s chance of winning (%)", tips, height=340,
                 x_max=max(48.0, float(w.minute.max())))
    w["change"] = w.pct.diff().fillna(0)
    sw = w.loc[w.change.abs().sort_values(ascending=False).index].head(5)
    items = "".join(f"<li><b>{r.clock}</b>: {r.who or r.team} {r.play} → {r.score}, {game.home} {'up' if r.change > 0 else 'down'} "
                    f"<b>{abs(r.change):.0f} points</b> to {r.pct:.0f}%</li>" for r in sw.itertuples())
    style.card("Biggest swings", f"<ul style='margin:.2rem 0 0 1rem;padding:0'>{items}</ul>")


# ------------------------------------------------------------------ predictor
def predictor():
    g, s, b = load()
    params, info = model()
    style.hero("Win predictor", "Set up any game: pre-game chances from Elo ratings, then the live chance for any score and "
               "time left.", eyebrow="🏀 What are the chances?")
    elo = bb.ratings_now(g)
    teams_ = sorted(bb.TEAM_NAMES)
    a, c = st.columns(2)
    home = a.selectbox("Home team", teams_, index=teams_.index("BOS"), format_func=team_label, key="bb_ph")
    away = c.selectbox("Away team", teams_, index=teams_.index("DEN"), format_func=team_label, key="bb_pa")
    if home == away:
        st.info("Pick two different teams.")
        return
    gap = elo[home] - elo[away]
    p0 = float(bb.win_prob(params, 0, 1.0, gap))
    exp_margin = params[0] * gap / 100 + params[2]
    style.kpis([(f"{home} Elo", f"{elo[home]:.0f}", "end of 2023-24"), (f"{away} Elo", f"{elo[away]:.0f}", "end of 2023-24"),
                ("Expected margin", f"{exp_margin:+.1f}", f"for {home} (incl. home court)", "gold")])
    prob_bar([(home, p0, AQUA), (away, 1 - p0, ORANGE)])
    st.markdown("##### Live: set the score and the clock")
    q, mins, marg = st.columns(3)
    period = q.selectbox("Quarter", [1, 2, 3, 4], index=3, key="bb_q")
    left = mins.slider("Minutes left in the quarter", 0.0, 12.0, 5.0, 0.5, key="bb_left")
    margin = marg.number_input(f"{home} lead (negative = behind)", -40, 40, -6, key="bb_m")
    t = (period - 1) * 720 + (720 - left * 60)
    f = bb.share_left(t, period)
    p = float(bb.win_prob(params, margin, f, gap))
    prob_bar([(home, p, AQUA), (away, 1 - p, ORANGE)])
    st.markdown("##### How big a lead do you need?")
    rows = []
    for m_left in np.arange(48, 0, -0.5):
        ff = m_left * 60 / bb.GAME_SECONDS
        for target in (0.9, 0.99):
            need = next((L for L in range(0, 61) if bb.win_prob(params, L, ff, 0) >= target), 60)
            rows.append({"minutes_left": m_left, "lead": need, "Chance": f"{target:.0%} sure"})
    d = pd.DataFrame(rows)
    chart(alt.Chart(d).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x=alt.X("minutes_left:Q", title="Minutes left", scale=alt.Scale(reverse=True, domain=[48, 0])),
        y=alt.Y("lead:Q", title="Home lead needed (points)"),
        color=alt.Color("Chance:N", scale=alt.Scale(range=[AQUA, ORANGE]), legend=alt.Legend(title=None)),
        tooltip=["minutes_left:Q", "lead:Q", "Chance:N"]), 280)
    st.caption("For two equal teams with the home team in front. ")
    res = info["results"]
    fin = res["+ Elo team strength (final)"]
    st.caption(f"Model: the final margin is treated as a random walk around the expected margin (from Elo and home court), with "
               f"uncertainty that shrinks with the square root of the time left. Trained on 2021-23 and tested on all "
               f"{info['test_games']:,} games of 2023-24: AUC {fin['AUC']:.2f}, log loss {fin['Log loss']:.3f} "
               f"(score and clock alone: {res['Score and clock only (no home edge, no team strength)']['Log loss']:.3f}). "
               f"It picks the winner {info['by_stage']['Tip-off']['Accuracy']:.0%} of the time at tip-off and "
               f"{info['by_stage']['Last 2 minutes']['Accuracy']:.0%} in the last two minutes.")


PAGES = {"Overview": overview, "Teams": teams, "Players": players, "Match replay": replay, "Predictor": predictor}
