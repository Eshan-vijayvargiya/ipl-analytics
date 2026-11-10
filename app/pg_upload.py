"""'Your file' page: upload any sports file (or several), and the engine reads it, splits it by sport,
recognises each sport and kind of table, cleans it, and runs the analysis that fits."""
import hashlib
import traceback
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import analyse as an
import engine as en
import style
from ui import AQUA, BLUE, GOLD, MUTED, ORANGE, chart, pct, prob_bar, two_way_worm

HERE = Path(__file__).resolve().parent
ICON = {"Cricket": "🏏", "Badminton": "🏸", "Basketball": "🏀", "Football": "⚽", "Volleyball": "🏐", None: "❔", "Unknown": "❔"}
DEMOS = {"Mixed: 5 sports in one file": "mixed_sports_results.csv", "Messy football file": "india_football_messy.csv",
         "Excel: 3 sports, 3 sheets": "three_sports_workbook.xlsx", "IPL ball by ball": "ipl_2024_ball_by_ball.csv",
         "Badminton point by point": "badminton_sen_vs_loh_points.csv"}
ROLE_NAMES = {"side_a": "First team / player (home)", "side_b": "Second team / player (away)", "score_a": "First side's score",
              "score_b": "Second side's score", "score_text": "Score as text (e.g. 2-1)", "winner": "Winner", "date": "Date",
              "competition": "Competition", "season": "Season", "venue": "Venue", "neutral": "Neutral ground",
              "match_id": "Match ID", "period": "Innings / quarter / set / game", "minute": "Minute", "clock": "Clock",
              "over": "Over", "ball": "Ball", "batting_team": "Batting team", "bowling_team": "Bowling team", "batter": "Batter",
              "bowler": "Bowler", "runs": "Runs", "extras": "Extras", "wicket": "Wicket", "team": "Team", "player": "Player / name",
              "point_winner": "Point won by", "games_played": "Games played", "loser": "Loser"}
KIND_ROLES = {
    "results": ["side_a", "side_b", "score_a", "score_b", "score_text", "winner", "date", "competition", "neutral"],
    "events": ["match_id", "period", "over", "ball", "minute", "clock", "score_a", "score_b", "score_text", "batting_team",
               "batter", "bowler", "runs", "extras", "wicket", "team", "player", "point_winner", "side_a", "side_b"],
    "players": ["player", "team", "games_played"],
    "table": [],
}


@st.cache_data(show_spinner="Reading the file…", max_entries=8)
def parse(name, data):
    """Every table in the file, split by sport, with the engine's guesses."""
    parts = []
    for label, df in en.read_tables(name, data):
        split, col = en.split_by_sport(df)
        for sport, part in split:
            guess = en.detect_sport(part, hint=label)
            parts.append({"label": label, "split_col": col, "given": sport, "df": part, "guess": guess})
    return parts


def safe(title, fn, *args):
    """Run one section; if it fails on an unusual file, explain instead of crashing the page."""
    try:
        return fn(*args)
    except Exception as err:          # noqa: BLE001  (any failure becomes a message)
        st.warning(f"**{title}** couldn't be shown for this table: {type(err).__name__}. The rest of the analysis is unaffected. "
                   "You can fix the column choices under 'How I read it'.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception_only(type(err), err))[-1200:])
        return None


# ------------------------------------------------------------------ page
def render():
    style.hero("Analyse your own sports file", "Upload any sports data: CSV, Excel (every sheet), JSON, PDF, Word or a ZIP of "
               "files. It works out the sport, splits mixed files, recognises the kind of table, cleans it, and runs the "
               "matching analysis and model.", eyebrow="📂 Your file")
    style.html("""<div class="flow"><div><span>1</span> Read every table</div><div><span>2</span> Split mixed files by sport</div>
    <div><span>3</span> Recognise the sport</div><div><span>4</span> Recognise the table: results, play-by-play or player stats</div>
    <div><span>5</span> Map and clean the columns</div><div><span>6</span> Run the right model</div></div>""")
    state = st.session_state
    up = st.file_uploader("Upload one or more files", type=en.SUPPORTED, accept_multiple_files=True, key="up_files")
    st.caption("Or try a sample:")
    cols = st.columns(len(DEMOS))
    for col, (label, fname) in zip(cols, DEMOS.items()):
        if col.button(label, width="stretch", key=f"demo_{fname}"):
            state.up_demo = fname
    files = []
    if up:
        files = [(f.name, f.getvalue()) for f in up]
        state.pop("up_demo", None)
    elif state.get("up_demo"):
        p = HERE / "demo" / state.up_demo
        files = [(p.name, p.read_bytes())]
    if not files:
        st.info("Nothing uploaded yet. Pick a sample above to see how it works, or drop in your own file.")
        return
    parts = []
    for name, data in files:
        try:
            for p in parse(name, data):
                parts.append({**p, "file": name, "id": hashlib.md5(data).hexdigest()[:8]})
        except ValueError as err:
            st.error(f"**{name}**: {err}")
        except Exception as err:        # noqa: BLE001
            st.error(f"**{name}** couldn't be read ({type(err).__name__}). Try saving it again as CSV or Excel.")
    if not parts:
        return
    sports = {p["given"] or p["guess"].sport for p in parts}
    style.kpis([("Files", f"{len(files)}", ", ".join(n for n, _ in files)[:60]), ("Tables found", f"{len(parts)}", "sheets, tables or sport groups"),
                ("Sports recognised", f"{len([s for s in sports if s and s != 'Unknown'])}", " ".join(ICON.get(s, '') for s in sports if s)),
                ("Rows", f"{sum(len(p['df']) for p in parts):,}", "in all tables")])
    labels = []
    for i, p in enumerate(parts):
        sport = p["given"] or p["guess"].sport
        pl = en.plan_table(p["df"], sport if sport != "Unknown" else None)
        p["plan"] = pl
        how = f"rows where '{p['split_col']}' = {sport}" if p["given"] else p["label"]
        labels.append(f"{ICON.get(sport, '❔')} {sport or 'Unknown sport'} · {pl.label} · {len(p['df']):,} rows · {how}")
    pick = st.selectbox("Table to analyse", range(len(parts)), format_func=lambda i: labels[i], key=f"up_pick_{parts[0]['id']}")
    analyse_part(parts[pick], f"{parts[pick]['id']}_{pick}")


def analyse_part(p, key):
    df, guess = p["df"], p["guess"]
    sport = p["given"] if p["given"] and p["given"] != "Unknown" else guess.sport
    with st.expander("How I read it (change anything that's wrong)", expanded=sport is None):
        a, b = st.columns([1, 2])
        options = en.SPORTS + ["Not sure"]
        chosen = a.selectbox("Sport", options, index=options.index(sport) if sport in options else len(options) - 1, key=f"sp_{key}")
        if p["given"]:
            b.markdown(f"Taken from the **{p['split_col']}** column of the file.")
        elif guess.sport:
            b.markdown(f"Recognised as **{guess.sport}** ({guess.confidence:.0%} sure) from: " + "; ".join(guess.evidence))
        else:
            b.markdown("I couldn't tell the sport from this table, so please pick it. The general analysis below works either way.")
        sport = None if chosen == "Not sure" else chosen
        plan = en.plan_table(df, sport)
        kinds = ["results", "events", "players", "table"]
        kind = a.selectbox("Kind of table", kinds, index=kinds.index(plan.kind),
                           format_func=lambda k: {"results": "Match results", "events": f"Play-by-play ({en.EVENT_LABEL.get(sport, 'events')})",
                                                  "players": "Player or team stats", "table": "Other table"}[k], key=f"kind_{key}")
        roles = dict(plan.roles)
        wanted = KIND_ROLES[kind]
        if wanted:
            st.markdown("**Columns**")
            cols = st.columns(3)
            for i, role in enumerate(wanted):
                opts = ["(none)"] + list(df.columns)
                cur = roles.get(role)
                sel = cols[i % 3].selectbox(ROLE_NAMES.get(role, role), opts, index=opts.index(cur) if cur in opts else 0, key=f"role_{key}_{role}")
                if sel == "(none)":
                    roles.pop(role, None)
                else:
                    roles[role] = sel
        for n in plan.notes:
            st.caption(n)
    c = en.clean(df, {r: c_ for r, c_ in roles.items() if c_ in df.columns})
    safe("Data quality", quality, df, c, roles)
    if kind == "results":
        safe("Results analysis", results_view, c, sport, roles, key)
    elif kind == "events":
        safe("Play-by-play analysis", events_view, c, sport, roles, key)
    elif kind == "players":
        safe("Player analysis", players_view, df, roles, sport, key)
    else:
        safe("Table summary", table_view, df, key)
    st.download_button("Download the cleaned table (CSV)", c.df.to_csv(index=False).encode("utf-8"), f"cleaned_{key}.csv", "text/csv",
                       icon=":material/download:", key=f"dl_{key}")


# ------------------------------------------------------------------ sections
def quality(df, c, roles):
    fixed = int(c.report.Fixed.sum()) if len(c.report) else 0
    bad = int(c.report.Unreadable.sum()) if len(c.report) else 0
    blanks = int(c.report.Blank.sum()) if len(c.report) else 0
    style.kpis([("Rows read", f"{len(df):,}", f"{len(df.columns)} columns"), ("Rows kept", f"{len(c.df):,}", f"{c.dropped} empty or repeated removed"),
                ("Cells fixed", f"{fixed:,}", "e.g. 'two' → 2, 'INDIA' → India", "gold"), ("Cells unreadable", f"{bad:,}", f"{blanks:,} blank")])
    if fixed or bad:
        with st.expander("See the problem cells"):
            show = c.report[(c.report.Fixed > 0) | (c.report.Unreadable > 0) | (c.report.Blank > 0)]
            show = show.assign(Role=show.Role.map(lambda r: ROLE_NAMES.get(r, r)))
            st.dataframe(show, hide_index=True, width="stretch")
            idx = pd.Index([])
            for role, stt in c.status.items():
                idx = idx.union(stt.index[stt.isin(["fixed", "unreadable"])])
            rows = df.loc[idx[:300]]
            colour = pd.DataFrame("", index=rows.index, columns=rows.columns)
            for role, stt in c.status.items():
                col = roles.get(role)
                if col in colour:
                    s = stt.reindex(rows.index)
                    colour.loc[s == "fixed", col] = "background-color:#d6ecdc;color:#174a31"
                    colour.loc[s == "unreadable", col] = "background-color:#ffe795;color:#4d3a00;font-weight:600"
            st.caption("Green = read after fixing · yellow = couldn't be read (left blank). First 300 rows with a problem.")
            st.dataframe(rows.astype(str).style.apply(lambda _: colour, axis=None), width="stretch", height=280)


def results_view(c, sport, roles, key):
    d = c.df
    m = an.standard_results(d, sport, roles)
    if len(m) < 2:
        st.info("Fewer than two complete matches after cleaning, so there's nothing to rank yet.")
        return
    home = an.has_home(roles)
    m, ratings, played = an.elo_walk(m, sport, home)
    if ratings.empty:
        st.info("The matches were read, but none has a result I could use (both scores or a winner), so there is nothing to "
                "rank. Check the score columns under 'How I read it'.")
        st.dataframe(m[["date", "a", "b", "sa", "sb"]].head(200), hide_index=True, width="stretch")
        return
    ev = an.evaluate(m, sport)
    decided = m[m.outcome.isin(["A", "B", "D"])]
    first_label = "Home team won" if home else "First-named side won"
    span = f"{m.date.min():%b %Y} – {m.date.max():%b %Y}" if m.date.notna().any() else f"{len(m)} matches"
    style.kpis([("Matches", f"{len(m):,}", span), ("Teams / players", f"{len(ratings):,}", ""),
                (first_label, pct((decided.outcome == 'A').mean() * 100), f"{(decided.outcome == 'D').mean():.0%} drawn" if (decided.outcome == 'D').any() else ""),
                ("Top rated", str(ratings.index[0])[:26], f"Elo {ratings.iloc[0]:.0f}", "gold")])
    if ev:
        better = ev["log_loss"] < ev["baseline_log_loss"]
        style.card("A model trained on your file", f"""<p>Elo ratings learned match by match from your file, turned into win chances
        with the {sport.lower() if sport else 'generic'} model{' (Poisson goals: win, draw or loss)' if sport == 'Football' else ''}. Every
        prediction uses only earlier matches, so this is an honest test: after learning from the first {ev['learned_from']} matches it
        called <b>{ev['accuracy']:.0%}</b> of the next {ev['tested']} results correctly (log loss {ev['log_loss']:.3f}), against
        {ev['baseline_accuracy']:.0%} ({ev['baseline_log_loss']:.3f}) for {ev['baseline']}.
        {'The ratings beat the simple rule.' if better else 'Here the ratings did not beat the simple rule: the file is small or the teams are very even.'}</p>""", insight=True)
    t1, t2, t3 = st.tabs(["Table", "Ratings", "Head to head & predictor"])
    with t1:
        s = an.standings(m, sport)
        s["Elo"] = s.team.map(ratings)
        cfg = {"team": "Team / player", "Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.0f%%"),
               "Elo": st.column_config.NumberColumn("Elo", format="%.0f"), "For": st.column_config.NumberColumn("For", format="%d"),
               "Against": st.column_config.NumberColumn("Against", format="%d"), "Diff": st.column_config.NumberColumn("Diff", format="%+d")}
        st.dataframe(s, hide_index=True, width="stretch", height=min(420, 36 * len(s) + 40), column_config=cfg)
    with t2:
        top = ratings.head(20).rename("elo").reset_index()
        top.columns = ["team", "elo"]
        chart(alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=4, size=14).encode(
            y=alt.Y("team:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)), x=alt.X("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
            tooltip=["team:N", alt.Tooltip("elo:Q", format=".0f")]), 22 * len(top) + 30)
        st.caption("Everyone starts at 1500. Beating a stronger side earns more points than beating a weaker one.")
    with t3:
        names = ratings.index.tolist()
        if len(names) < 2:
            return
        a, b = st.columns(2)
        x = a.selectbox("Team / player 1", names, index=0, key=f"h2h_a_{key}")
        y = b.selectbox("Team / player 2", names, index=1, key=f"h2h_b_{key}")
        if x == y:
            st.info("Pick two different sides.")
            return
        ph, pdr, pa = an.win_chances(sport, ratings[x], ratings[y], home)
        parts = [(x, ph, AQUA)] + ([("Draw", pdr, "#7d8aa5")] if pdr > 0.001 else []) + [(y, pa, ORANGE)]
        prob_bar(parts)
        where = f"at {x}'s home" if home else "on a neutral ground"
        st.caption(f"{x} {where}, from the Elo ratings learned from your file.")
        h = m[((m.a == x) & (m.b == y)) | ((m.a == y) & (m.b == x))]
        if len(h):
            xw = ((h.a == x) & (h.outcome == "A")).sum() + ((h.b == x) & (h.outcome == "B")).sum()
            yw = ((h.a == y) & (h.outcome == "A")).sum() + ((h.b == y) & (h.outcome == "B")).sum()
            style.kpis([("Meetings", f"{len(h)}", ""), (f"{x} won", f"{xw}", "", "gold"), (f"{y} won", f"{yw}", ""), ("Drawn", f"{(h.outcome == 'D').sum()}", "")])
            st.dataframe(h[["date", "a", "sa", "sb", "b"]].rename(columns={"date": "Date", "a": "Side 1", "sa": "Score 1", "sb": "Score 2", "b": "Side 2"}),
                         hide_index=True, width="stretch")
        else:
            st.caption("They never met in this file.")


def events_view(c, sport, roles, key):
    d = c.df
    if sport == "Cricket":
        games, balls = an.cricket_events(d)
        if not games:
            st.info("No complete innings found.")
            return
        labels = [f"{g['teams'] or g['match']} · {g['first']}/{g['first_wkts']}" + (f" vs {g['second']}/{g['second_wkts']}" if g["second"] is not None else "") for g in games]
        i = st.selectbox("Match", range(len(games)), format_func=lambda j: labels[j], index=len(games) - 1, key=f"cm_{key}")
        g = games[i]
        if "worm" in g:
            start, w = g["worm"]
            w = pd.concat([pd.DataFrame({"ball_no": [0], "win_prob": [start], "score": [0], "wkts": [0], "batter": [""], "bowler": [""], "runs": [0], "wicket": [False]}), w], ignore_index=True)
            w["pct"] = w.win_prob * 100
            w["state"] = w.score.astype(str) + "/" + w.wkts.astype(str)
            tips = [alt.Tooltip("ball_no:Q", title="Ball"), alt.Tooltip("state:N", title="Score"), "batter:N", "bowler:N",
                    alt.Tooltip("pct:Q", format=".0f", title="Chasing side win %")]
            two_way_worm(w, "ball_no", "pct", f"Ball of the chase (target {g['first'] + 1})", "Chasing side's chance (%)", tips,
                         marks=w[w.wicket == True])
            st.caption("From the IPL chase model (trained on 2008-2021, tested on 2022-24). Orange dots are wickets.")
        else:
            st.info("This match has no second innings to replay.")
        a, b = st.columns(2)
        bat = balls.groupby("batter").batter_runs.sum().sort_values(ascending=False).head(10).reset_index()
        bowl = balls[balls.bowler_wicket].groupby("bowler").size().sort_values(ascending=False).head(10).rename("Wickets").reset_index()
        a.markdown("##### Top run scorers (whole file)")
        a.dataframe(bat.rename(columns={"batter": "Batter", "batter_runs": "Runs"}), hide_index=True, width="stretch")
        b.markdown("##### Top wicket takers (whole file)")
        b.dataframe(bowl.rename(columns={"bowler": "Bowler"}), hide_index=True, width="stretch")
        st.caption("Runs are the batter's own runs; run-outs aren't credited to the bowler.")
        return
    if sport == "Basketball":
        games = an.basketball_events(d)
        i = st.selectbox("Game", range(len(games)), format_func=lambda j: f"{games[j]['game']} · final {games[j]['final']}", key=f"bm_{key}")
        w = games[i]["worm"].assign(pct=lambda x: x.home_win * 100, score=lambda x: x.away.astype(int).astype(str) + "-" + x.home.astype(int).astype(str))
        two_way_worm(w, "minute", "pct", "Minutes played", "Home team's chance (%)", [alt.Tooltip("minute:Q", format=".1f"), "score:N", alt.Tooltip("pct:Q", format=".0f")],
                     x_max=max(48.0, float(w.minute.max())))
        st.caption("From the NBA random-walk model, treating the two teams as equally good (the file has no ratings).")
        return
    if sport == "Football":
        games = an.football_events(d, roles)
        i = st.selectbox("Match", range(len(games)), format_func=lambda j: f"{games[j]['home']} vs {games[j]['away']} · {games[j]['match']}", key=f"fm_{key}")
        g = games[i]
        long = g["worm"].melt("minute", ["home_win", "draw", "away_win"], var_name="k", value_name="p")
        order = [f"{g['home']} win", "Draw", f"{g['away']} win"]
        long["Outcome"] = long.k.map(dict(zip(["home_win", "draw", "away_win"], order)))
        chart(alt.Chart(long).mark_area(interpolate="step-after").encode(
            x=alt.X("minute:Q", title="Minute"), y=alt.Y("p:Q", stack="normalize", title="Chance", axis=alt.Axis(format="%")),
            color=alt.Color("Outcome:N", sort=order, scale=alt.Scale(domain=order, range=[AQUA, "#7d8aa5", ORANGE]), legend=alt.Legend(title=None)),
            tooltip=["minute:Q", "Outcome:N", alt.Tooltip("p:Q", format=".0%")]), 320)
        st.caption("From the Poisson goal model with average international scoring rates (the file has no team ratings).")
        return
    if sport == "Badminton":
        games = an.badminton_events(d, roles)
        i = st.selectbox("Match", range(len(games)), format_func=lambda j: f"{games[j]['match']} · games {games[j]['games']}", key=f"bdm_{key}")
        w = games[i]["worm"].assign(pct=lambda x: x.a_win * 100)
        first = (d.point_winner.dropna().iloc[0] if "point_winner" in d and d.point_winner.notna().any() else "First player")
        two_way_worm(w, "n", "pct", "Rally number", f"{first}'s chance of winning (%)", ["n:Q", "game:Q", "score:N", alt.Tooltip("pct:Q", format=".0f")])
        st.caption("From the badminton Markov chain: every rally starts 50/50 and the estimate learns from the rallies played.")
        return
    if sport == "Volleyball":
        games = an.volleyball_events(d, roles)
        names = {}
        if "match_id" in d and "side_a" in d:
            names = d.groupby("match_id").agg(a=("side_a", "first"), b=("side_b", "first")).astype(str)
            names = {str(k): f"{r.a} vs {r.b}" for k, r in names.iterrows()}
        i = st.selectbox("Match", range(len(games)), format_func=lambda j: f"{names.get(games[j]['match'], games[j]['match'])} · sets {games[j]['sets']}", key=f"vm_{key}")
        w = games[i]["worm"].assign(pct=lambda x: x.a_win * 100)
        chart(alt.Chart(w).mark_line(point=alt.OverlayMarkDef(size=80, filled=True, color=GOLD), color="#f4f6fb").encode(
            x=alt.X("after:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("pct:Q", title="First team's chance (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["after:N", alt.Tooltip("pct:Q", format=".0f")]), 300)
        return
    st.info("Pick the sport under 'How I read it' to run its play-by-play model.")


def players_view(df, roles, sport, key):
    t = an.player_table(df, roles, sport)
    stats = [c for c in t.columns if c not in ("name", "team") and pd.api.types.is_numeric_dtype(t[c]) and t[c].notna().sum() >= 3]
    if not stats:
        st.info("No number columns to rank by.")
        return
    rates = [r[0] for r in an.RATES.get(sport, []) if r[0] in t.columns]
    a, b = st.columns([2, 1])
    default = rates[0] if rates else stats[0]
    stat = a.selectbox("Rank by", stats, index=stats.index(default), key=f"pl_stat_{key}")
    n = b.slider("How many", 5, 40, 15, key=f"pl_n_{key}")
    low_first = bool(en.tokens(stat) & {"economy", "era", "conceded", "against", "errors", "turnovers", "tov", "fouls"})
    top = t.dropna(subset=[stat]).sort_values(stat, ascending=low_first).head(n)
    style.kpis([("Rows", f"{len(t):,}", "players / teams"), ("Number columns", f"{len(stats)}", ", ".join(stats[:4])),
                (f"Best {stat}"[:30], str(top.name.iloc[0])[:24], f"{top[stat].iloc[0]:,.2f}".rstrip('0').rstrip('.'), "gold")])
    chart(alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=4, size=14).encode(
        y=alt.Y("name:N", sort="x" if low_first else "-x", title=None, axis=alt.Axis(labelLimit=220)), x=alt.X(f"{stat}:Q", title=stat),
        tooltip=["name:N"] + ([alt.Tooltip("team:N")] if "team" in top else []) + [alt.Tooltip(f"{stat}:Q", format=",.2f")]), 22 * len(top) + 30)
    if rates:
        st.caption("Worked out from your columns: " + ", ".join(rates) + ".")
    if len(stats) >= 2:
        c1, c2 = st.columns(2)
        x = c1.selectbox("Compare: x", stats, index=0, key=f"pl_x_{key}")
        y = c2.selectbox("Compare: y", stats, index=1, key=f"pl_y_{key}")
        sc = t.dropna(subset=[x, y])
        chart(alt.Chart(sc).mark_circle(size=55, opacity=0.7, color=AQUA).encode(
            x=alt.X(f"{x}:Q", title=x), y=alt.Y(f"{y}:Q", title=y), tooltip=["name:N", alt.Tooltip(f"{x}:Q", format=",.2f"), alt.Tooltip(f"{y}:Q", format=",.2f")]), 280)
        if len(sc) >= 5:
            st.caption(f"Correlation r = {sc[[x, y]].corr().iloc[0, 1]:.2f}")
    st.dataframe(t, hide_index=True, width="stretch", height=320)


def table_view(df, key):
    nums = [c for c in df.columns if en.num_share(df[c]) >= 0.8]
    st.info("This table isn't match results, play-by-play or player stats, so here is a general summary. If it is one of those, "
            "set the sport and kind under 'How I read it'.")
    if nums:
        col = st.selectbox("Number column", nums, key=f"tb_{key}")
        v, _ = en.to_number(df[col])
        chart(alt.Chart(pd.DataFrame({"v": v.dropna()})).mark_bar(color=BLUE).encode(
            x=alt.X("v:Q", bin=alt.Bin(maxbins=30), title=col), y=alt.Y("count():Q", title="Rows")), 260)
    st.dataframe(df.head(500), width="stretch", height=320)
