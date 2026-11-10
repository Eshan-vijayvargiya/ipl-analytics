"""Football section: 49,940 international matches, 1872-2026."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import football as fb
import style
from ui import AQUA, BLUE, GOLD, MUTED, ORANGE, chart, pct, prob_bar

ROOT = Path(__file__).resolve().parent.parent
GREY = "#7d8aa5"


@st.cache_data(show_spinner="Loading 150 years of international football…")
def load():
    m = pd.read_parquet(fb.DATA / "matches.parquet")
    g = pd.read_parquet(fb.DATA / "goals.parquet")
    return m, g


@st.cache_resource
def model():
    return joblib.load(ROOT / "models" / "football_model.joblib"), json.loads((ROOT / "models" / "football_info.json").read_text())


@st.cache_data(show_spinner=False)
def ratings():
    m, _ = load()
    r = fb.ratings_now(m)
    last = pd.concat([m[["date", "home_team"]].rename(columns={"home_team": "team"}),
                      m[["date", "away_team"]].rename(columns={"away_team": "team"})]).groupby("team").date.max()
    active = last[last >= m.date.max() - pd.Timedelta(days=730)].index
    return r, r[r.index.isin(active)]


@st.cache_data(show_spinner=False)
def champions():
    m, _ = load()
    wc = m[m.tournament == "FIFA World Cup"].sort_values("date").groupby("year").tail(1)
    out = {}
    for r in wc.itertuples():
        out[r.year] = (r.home_team if r.home_score > r.away_score else r.away_team if r.away_score > r.home_score
                       else r.shootout_winner)
    out[1950] = "Uruguay"           # 1950 ended with a final group, not a final: Uruguay won it
    return pd.Series(out).sort_index()


def team_games(m, team):
    h = m[m.home_team == team].assign(opponent=lambda d: d.away_team, gf=lambda d: d.home_score, ga=lambda d: d.away_score,
                                      venue=np.where(m[m.home_team == team].neutral, "Neutral", "Home"))
    a = m[m.away_team == team].assign(opponent=lambda d: d.home_team, gf=lambda d: d.away_score, ga=lambda d: d.home_score,
                                      venue=np.where(m[m.away_team == team].neutral, "Neutral", "Away"))
    t = pd.concat([h, a]).sort_values("date")
    t["outcome"] = np.select([t.gf > t.ga, t.gf < t.ga], ["Won", "Lost"], "Drawn")
    return t


# ------------------------------------------------------------------ overview
def overview():
    m, g = load()
    style.hero("What 150 years of international football say",
               "Every men's international from the first match (Scotland 0-0 England, 1872) to the 2026 World Cup: "
               "how scoring changed, how much home advantage is worth, and who rules the world now.",
               eyebrow="⚽ Football · 1872-2026")
    nn = m[~m.neutral]
    r_all, r_active = ratings()
    style.kpis([("Matches", f"{len(m):,}", f"{m.home_team.nunique() + 0:,}+ national teams"),
                ("Goals", f"{int(m.home_score.sum() + m.away_score.sum()):,}", f"{(m.home_score + m.away_score).mean():.2f} a match"),
                ("Home team won", pct((nn.result == 'H').mean() * 100), "vs 44% at neutral grounds", "gold"),
                ("Draws", pct((m.result == 'D').mean() * 100), "about 1 match in 4"),
                ("World No. 1 now", r_active.index[0], f"Elo {r_active.iloc[0]:.0f}")])
    t1, t2, t3, t4 = st.tabs(["Goals over time", "Home advantage", "World Cup", "Best teams now"])
    m = m.assign(decade=m.year // 10 * 10)
    with t1:
        d = m.groupby("decade").apply(lambda x: (x.home_score + x.away_score).mean(), include_groups=False).rename("goals").reset_index()
        d["label"] = d.decade.astype(str) + "s"
        c = alt.Chart(d).mark_bar(color=BLUE, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=26).encode(
            x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelOverlap="parity")),
            y=alt.Y("goals:Q", title="Goals per match"), tooltip=[alt.Tooltip("label:N", title="Decade"), alt.Tooltip("goals:Q", format=".2f")])
        chart(c, 300)
        b = pd.cut(g.minute, [0, 15, 30, 45, 60, 75, 90, 125], labels=["1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "Extra time"])
        gm = g.groupby(b, observed=True).size().rename("goals").reset_index().rename(columns={"minute": "window"})
        gm["window"] = gm.window.astype(str)
        st.markdown("##### When goals are scored")
        c2 = alt.Chart(gm[gm.window != "Extra time"]).mark_bar(color=AQUA, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=40).encode(
            x=alt.X("window:N", sort=None, title="Minutes", axis=alt.Axis(labelAngle=0)), y=alt.Y("goals:Q", title="Goals"),
            tooltip=["window:N", alt.Tooltip("goals:Q", format=",")])
        chart(c2, 240)
        style.card("What changed", "<p>Early internationals were goal-fests: <b>5.6 goals a match in the 1880s</b>. Defending got "
                   "organised and the number fell to <b>about 2.7 since the 1980s</b>. Within a match, the <b>last 15 minutes are the "
                   "busiest</b> (tired legs, teams chasing a result): 1 goal in 5 comes after the 75th minute.</p>", insight=True)
    with t2:
        h = nn.assign(decade=nn.year // 10 * 10).groupby("decade").result.value_counts(normalize=True).unstack().reset_index()
        h = h[h.decade >= 1900].melt("decade", ["H", "D", "A"], var_name="r", value_name="share")
        h["Result"] = h.r.map({"H": "Home win", "D": "Draw", "A": "Away win"})
        h["share"] *= 100
        c = alt.Chart(h).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("decade:O", title=None, axis=alt.Axis(labelAngle=0, labelOverlap="parity")),
            y=alt.Y("share:Q", title="Share of matches (%)", scale=alt.Scale(domain=[0, 70])),
            color=alt.Color("Result:N", scale=alt.Scale(domain=["Home win", "Draw", "Away win"], range=[AQUA, GREY, ORANGE]), legend=alt.Legend(title=None)),
            tooltip=["decade:O", "Result:N", alt.Tooltip("share:Q", format=".0f")])
        chart(c, 300)
        info = model()[1]
        style.card("How much is home worth?", f"""<p>Home teams win <b>51%</b> and lose only <b>26%</b>; at neutral grounds the
        better-known 'home' side wins 44%. The goal model puts a number on it: playing at home multiplies a team's expected goals by
        about <b>{np.exp(info['coef']['at_home']):.2f}x</b> and cuts the visitors' by about <b>{1 / np.exp(info['coef']['away_side']):.2f}x</b>.
        That edge has hardly moved in 100 years.</p>""", insight=True)
    with t3:
        ch = champions()
        cells = "".join(f'<div><div class="y">{y}</div><div class="t" style="font-size:.92rem">{t}</div></div>' for y, t in ch.items())
        style.html(f'<div class="champ">{cells}</div>')
        top = ch.value_counts().rename("titles").reset_index().rename(columns={"index": "team"})
        top.columns = ["team", "titles"]
        c = alt.Chart(top).mark_bar(color=GOLD, cornerRadiusEnd=4, size=18).encode(
            y=alt.Y("team:N", sort="-x", title=None), x=alt.X("titles:Q", title="World Cups won", axis=alt.Axis(tickMinStep=1)),
            tooltip=["team:N", "titles:Q"])
        chart(c, 26 * len(top) + 30)
    with t4:
        top = r_active.head(20).rename("elo").reset_index().rename(columns={"index": "team"})
        top.columns = ["team", "elo"]
        c = alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=4, size=16).encode(
            y=alt.Y("team:N", sort="-x", title=None), x=alt.X("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
            tooltip=["team:N", alt.Tooltip("elo:Q", format=".0f")])
        chart(c, 26 * len(top))
        st.caption(f"Ranked among the {len(r_active)} teams that played in the last two years. Elo: every team starts at 1500 and gains "
                   "points by beating others; World Cup wins and big wins count more.")


# ------------------------------------------------------------------ teams
def teams():
    m, g = load()
    r_all, r_active = ratings()
    names = sorted(set(m.home_team) | set(m.away_team))
    team = st.selectbox("Team", names, index=names.index("India"), key="fb_team")
    t = team_games(m, team)
    rank = list(r_active.index).index(team) + 1 if team in r_active.index else None
    style.hero(team, f"{len(t):,} international matches since {t.date.min():%Y}", eyebrow="⚽ Team profile")
    w, d_, l = (t.outcome == "Won").sum(), (t.outcome == "Drawn").sum(), (t.outcome == "Lost").sum()
    eh = fb.elo_history(m, team)
    peak = eh.loc[eh.elo_before.idxmax()]
    style.kpis([("Played", f"{len(t):,}", f"{w}-{d_}-{l} (W-D-L)"), ("Won", pct(w / len(t) * 100), "of matches", "gold"),
                ("Goals", f"{t.gf.sum():,} - {t.ga.sum():,}", "scored - conceded"),
                ("Elo now", f"{r_all.get(team, np.nan):.0f}", f"No. {rank} of {len(r_active)} active" if rank else "not active lately"),
                ("Peak Elo", f"{peak.elo_before:.0f}", f"{peak.date:%b %Y}")])
    c = alt.Chart(eh).mark_line(color=GOLD, strokeWidth=1.6).encode(
        x=alt.X("date:T", title=None), y=alt.Y("elo_before:Q", title="Elo rating", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), "opponent:N", "scored:Q", "conceded:Q", "tournament:N",
                 alt.Tooltip("elo_before:Q", format=".0f", title="Elo before")])
    st.markdown("##### Elo rating over time")
    chart(c, 260)
    a, b = st.columns(2)
    with a:
        st.markdown("##### Record by type of match")
        k = t.groupby("kind").outcome.value_counts().unstack(fill_value=0).reindex(columns=["Won", "Drawn", "Lost"], fill_value=0)
        k["Played"] = k.sum(axis=1)
        k["Win %"] = k.Won / k.Played * 100
        k = k.reindex([x for x in reversed(fb.TYPES) if x in k.index]).reset_index().rename(columns={"kind": "Type"})
        st.dataframe(k, hide_index=True, width="stretch",
                     column_config={"Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.0f%%")})
    with b:
        st.markdown("##### Top scorers for the team")
        sc = g.merge(t[["match_id"]], on="match_id")
        sc = sc[(sc.team == team) & ~sc.own_goal].scorer.value_counts().head(8).rename("goals").reset_index()
        if sc.empty:
            st.info("No matches with full scorer data for this team.")
        else:
            st.dataframe(sc.rename(columns={"scorer": "Player", "goals": "Goals"}), hide_index=True, width="stretch")
            st.caption("Only matches with every scorer recorded (mostly tournaments and qualifiers).")
    st.markdown("#### Head to head")
    opps = t.opponent.value_counts()
    opp = st.selectbox("Opponent", opps.index.tolist(), index=0, key="fb_opp",
                       format_func=lambda x: f"{x} ({opps[x]} matches)")
    h = t[t.opponent == opp].sort_values("date", ascending=False)
    style.kpis([("Meetings", f"{len(h)}", f"since {h.date.min():%Y}"), (f"{team} won", f"{(h.outcome == 'Won').sum()}", "", "gold"),
                ("Drawn", f"{(h.outcome == 'Drawn').sum()}", ""), (f"{opp} won", f"{(h.outcome == 'Lost').sum()}", "")])
    show = h.head(12).assign(score=lambda d: d.gf.astype(str) + "-" + d.ga.astype(str))
    st.dataframe(show[["date", "score", "outcome", "tournament", "venue", "city"]], hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("Date", format="D MMM YYYY"), "score": f"{team} score",
                                "outcome": "Result", "tournament": "Competition", "venue": "Venue", "city": "City"})


# ------------------------------------------------------------------ players (scorers)
def players():
    m, g = load()
    style.hero("Goalscorers", "Every scorer and minute from the matches with full scorer data: 47,600 goals in 15,500 matches "
               "(mostly World Cups, continental cups and qualifiers; many old friendlies have no scorers recorded).",
               eyebrow="⚽ Players")
    g = g.merge(m[["match_id", "date", "kind", "tournament", "home_team", "away_team"]], on="match_id")
    kinds = ["All matches"] + [k for k in reversed(fb.TYPES) if k in g.kind.unique()]
    a, b = st.columns([1, 1])
    kind = a.selectbox("Matches", kinds, key="fb_kind")
    since = b.slider("From year", 1900, 2026, 1990, key="fb_since")
    gg = g[(g.date.dt.year >= since) & ~g.own_goal]
    if kind != "All matches":
        gg = gg[gg.kind == kind]
    top = gg.groupby(["scorer", "team"]).agg(goals=("minute", "size"), penalties=("penalty", "sum")).reset_index()
    top = top.sort_values("goals", ascending=False).head(20)
    top["label"] = top.scorer + " (" + top.team + ")"
    c = alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=4, size=15).encode(
        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("goals:Q", title="Goals"),
        tooltip=["scorer:N", "team:N", "goals:Q", "penalties:Q"])
    chart(c, 24 * len(top) + 20)
    names = gg.scorer.value_counts()
    names = names[names >= 3].index.tolist()
    if not names:
        return
    who = st.selectbox("Player profile", names, index=0, key="fb_player")
    p = gg[gg.scorer == who]
    style.kpis([("Goals", f"{len(p)}", f"for {p.team.mode().iat[0]}", "gold"), ("Penalties", f"{int(p.penalty.sum())}", pct(p.penalty.mean() * 100) + " of goals"),
                ("Matches scored in", f"{p.match_id.nunique()}", f"{len(p) / p.match_id.nunique():.2f} goals each"),
                ("First / last", f"{p.date.min():%Y} / {p.date.max():%Y}", "")])
    a, b = st.columns(2)
    with a:
        y = p.groupby(p.date.dt.year).size().rename("goals").reset_index().rename(columns={"date": "year"})
        st.markdown("##### Goals by year")
        chart(alt.Chart(y).mark_bar(color=AQUA, size=14, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0, labelOverlap="parity")), y=alt.Y("goals:Q", title="Goals"),
            tooltip=["year:O", "goals:Q"]), 220)
    with b:
        opp = np.where(p.home_team == p.team, p.away_team, p.home_team)
        o = pd.Series(opp).value_counts().head(8).rename("goals").reset_index()
        o.columns = ["opponent", "goals"]
        st.markdown("##### Favourite opponents")
        chart(alt.Chart(o).mark_bar(color=ORANGE, size=14, cornerRadiusEnd=3).encode(
            y=alt.Y("opponent:N", sort="-x", title=None), x=alt.X("goals:Q", title="Goals"), tooltip=["opponent:N", "goals:Q"]), 220)


# ------------------------------------------------------------------ match replay
def replay():
    m, g = load()
    (mdl, info) = model()
    style.hero("Match replay", "Pick a match and watch the chances of a home win, a draw and an away win change minute by minute.",
               eyebrow="⚽ Minute-by-minute win probability")
    tl = m[m.has_timeline]
    a, b, c = st.columns([1.3, 0.7, 2.4])
    tours = tl.tournament.value_counts()
    tours = ["FIFA World Cup"] + [t for t in tours.index if t != "FIFA World Cup" and tours[t] >= 20]
    tour = a.selectbox("Competition", tours, key="fb_rp_tour")
    tt = tl[tl.tournament == tour]
    years = sorted(tt.year.unique(), reverse=True)
    year = b.selectbox("Year", years, index=years.index(2022) if 2022 in years else 0, key=f"fb_rp_year_{tour}")
    ms = tt[tt.year == year].sort_values("date", ascending=False)
    labels = (ms.date.dt.strftime("%d %b") + " · " + ms.home_team + " " + ms.home_score.astype(str) + "-" +
              ms.away_score.astype(str) + " " + ms.away_team + np.where(ms.shootout_winner.notna(), " (pens: " + ms.shootout_winner.fillna("") + ")", "")).tolist()
    pick = c.selectbox("Match", labels, index=0, key=f"fb_rp_match_{tour}_{year}")
    row = ms.iloc[labels.index(pick)]
    (lh, la), w, gl = fb.worm(mdl["model"], row, g, mdl["share"], mdl["extra"], mdl["rho"])
    head = f"{row.home_team} {row.home_score}-{row.away_score} {row.away_team}"
    so = f" · {row.shootout_winner} won on penalties" if isinstance(row.shootout_winner, str) else ""
    style.card(head, f"<p>{row.date:%d %B %Y} · {row.tournament} · {row.city}, {row.country}{so}</p>"
               f"<p>Before kick-off: Elo {row.elo_home:.0f} vs {row.elo_away:.0f} → expected goals "
               f"<b>{lh:.2f} - {la:.2f}</b></p>")
    long = w.melt("minute", ["home_win", "draw", "away_win"], var_name="k", value_name="p")
    long["Outcome"] = long.k.map({"home_win": f"{row.home_team} win", "draw": "Draw", "away_win": f"{row.away_team} win"})
    long["p"] *= 100
    order = [f"{row.home_team} win", "Draw", f"{row.away_team} win"]
    area = alt.Chart(long).mark_area(interpolate="step-after", opacity=0.9).encode(
        x=alt.X("minute:Q", title="Minute", scale=alt.Scale(domain=[0, int(w.minute.max())], nice=False)),
        y=alt.Y("p:Q", stack="normalize", title="Chance (%)", axis=alt.Axis(format="%")),
        color=alt.Color("Outcome:N", sort=order, scale=alt.Scale(domain=order, range=[AQUA, GREY, ORANGE]), legend=alt.Legend(title=None)),
        order=alt.Order("order:Q"), tooltip=["minute:Q", "Outcome:N", alt.Tooltip("p:Q", format=".0f", title="%")]
    ).transform_calculate(order=f"indexof({json.dumps(order)}, datum.Outcome)")
    marks = gl.assign(label=gl.scorer.str.split().str[-1] + " " + gl.minute.astype(str) + "'" + np.where(gl.penalty, " (pen)", "") + np.where(gl.own_goal, " (og)", ""))
    rules = alt.Chart(marks).mark_rule(color="#f4f6fb", strokeDash=[3, 3], opacity=0.8).encode(x="minute:Q", tooltip=["scorer:N", "team:N", "minute:Q"])
    text = alt.Chart(marks).mark_text(color="#f4f6fb", angle=270, align="left", dx=4, dy=-4, fontSize=10).encode(x="minute:Q", y=alt.datum(0.02), text="label:N")
    chart(area + rules + text, 360)
    prob_bar([(f"{row.home_team}", float(w.home_win.iloc[0]), AQUA), ("Draw", float(w.draw.iloc[0]), GREY), (f"{row.away_team}", float(w.away_win.iloc[0]), ORANGE)])
    st.caption("The bar shows the chances at kick-off. White dashed lines are goals. Goals scored after minute 90 in a normal "
               "match are recorded as 90 (stoppage time).")


# ------------------------------------------------------------------ predictor
def predictor():
    m, _ = load()
    (mdl, info) = model()
    r_all, r_active = ratings()
    style.hero("Match predictor", "Pick any two national teams: the Elo ratings and the goal model give the chances of each result "
               "and the most likely scores.", eyebrow="⚽ Who wins?")
    names = r_active.index.tolist()
    a, b, c, d = st.columns([1.2, 1.2, 1, 1])
    t1 = a.selectbox("Team 1", names, index=names.index("India") if "India" in names else 0, key="fb_p1")
    t2 = b.selectbox("Team 2", names, index=names.index("Pakistan") if "Pakistan" in names else 1, key="fb_p2")
    venue = c.selectbox("Venue", ["Neutral", f"{t1} at home", f"{t2} at home"], key="fb_pv")
    kind = d.selectbox("Match type", list(reversed(fb.TYPES)), index=5, key="fb_pk")
    if t1 == t2:
        st.info("Pick two different teams.")
        return
    home, away = (t2, t1) if venue == f"{t2} at home" else (t1, t2)
    lh, la = fb.expected_goals(mdl["model"], r_all[home], r_all[away], venue == "Neutral", kind)
    grid = fb.score_grid(lh, la, rho=mdl["rho"])
    ph, pdr, pa = fb.outcome(grid)
    p1, p2 = (ph, pa) if home == t1 else (pa, ph)
    e1, e2 = (lh, la) if home == t1 else (la, lh)
    style.kpis([(f"{t1} Elo", f"{r_all[t1]:.0f}", f"No. {names.index(t1) + 1} active"), (f"{t2} Elo", f"{r_all[t2]:.0f}", f"No. {names.index(t2) + 1} active"),
                ("Expected goals", f"{e1:.2f} - {e2:.2f}", f"{t1} - {t2}", "gold")])
    prob_bar([(t1, p1, AQUA), ("Draw", pdr, GREY), (t2, p2, ORANGE)])
    left, right = st.columns([1.1, 1])
    with left:
        cells = []
        n = 6
        for i in range(n):
            for j in range(n):
                g1, g2 = (i, j)
                prob = grid[i, j] if home == t1 else grid[j, i]
                cells.append({"t1": g1, "t2": g2, "p": prob * 100})
        hm = pd.DataFrame(cells)
        heat = alt.Chart(hm).mark_rect(cornerRadius=3).encode(
            x=alt.X("t2:O", title=f"{t2} goals", axis=alt.Axis(labelAngle=0)), y=alt.Y("t1:O", title=f"{t1} goals", sort="descending"),
            color=alt.Color("p:Q", scale=alt.Scale(scheme="blues"), legend=None), tooltip=[alt.Tooltip("t1:O", title=t1), alt.Tooltip("t2:O", title=t2), alt.Tooltip("p:Q", format=".1f", title="Chance %")])
        txt = alt.Chart(hm).mark_text(fontSize=11).encode(x="t2:O", y=alt.Y("t1:O", sort="descending"), text=alt.Text("p:Q", format=".0f"),
                                                          color=alt.condition("datum.p > 8", alt.value("#ffffff"), alt.value("#0e1525")))
        st.markdown("##### Chance of each score (%)")
        chart(heat + txt, 300)
    with right:
        st.markdown("##### Live: what if the match is under way?")
        minute = st.slider("Minute", 0, 90, 60, key="fb_min")
        s1 = st.number_input(f"{t1} goals so far", 0, 15, 0, key="fb_s1")
        s2 = st.number_input(f"{t2} goals so far", 0, 15, 0, key="fb_s2")
        hg, ag = (s1, s2) if home == t1 else (s2, s1)
        lp = fb.in_play(lh, la, minute, hg, ag, mdl["share"], rho=mdl["rho"])
        q1, q3 = (lp[0], lp[2]) if home == t1 else (lp[2], lp[0])
        prob_bar([(t1, q1, AQUA), ("Draw", lp[1], GREY), (t2, q3, ORANGE)])
    res = info["results"]
    st.caption(f"Model: a Poisson goal model (Elo gap, home advantage and match type) trained on 1990-2021. On {info['test_matches']:,} "
               f"matches from 2022-2026 it never saw, it picked the right result {res['Poisson goal model (final)']['Accuracy']:.0%} of the "
               f"time (log loss {res['Poisson goal model (final)']['Log loss']:.3f} vs {res['Base rates (home win 48%)']['Log loss']:.3f} for always "
               f"guessing the usual split).")


PAGES = {"Overview": overview, "Teams": teams, "Players": players, "Match replay": replay, "Predictor": predictor}
