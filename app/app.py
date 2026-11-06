"""IPL Analytics 2008-2024: an interactive Streamlit app.

Run from the project folder:  python -m streamlit run app/app.py
"""
import importlib
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import ipl_data
import stats
import style
import winprob

# After a git push, Streamlit Cloud re-runs this file but can keep OLD helper modules in memory.
# Reloading them (a few milliseconds) means an update never runs a mix of new and old code.
for _module in (ipl_data, stats, style, winprob):
    importlib.reload(_module)

ROOT = Path(__file__).resolve().parent.parent
st.set_page_config(page_title="IPL Analytics 2008-2024", page_icon="🏏", layout="wide",
                   initial_sidebar_state="expanded")
style.setup()
state = st.session_state
state.setdefault("page", "Overview")
BLUE, ORANGE, AQUA, GOLD, MUTED = style.BLUE, style.ORANGE, style.AQUA, style.GOLD, style.MUTED


# ------------------------------------------------------------------ data (loaded once, then cached)
@st.cache_data(show_spinner="Loading 17 seasons of ball-by-ball data…")
def load():
    return ipl_data.load()


@st.cache_resource
def model():
    m = joblib.load(ROOT / "models" / "win_model.joblib")
    info = json.loads((ROOT / "models" / "model_info.json").read_text())
    return m, info


@st.cache_data(show_spinner=False)
def in_range(lo, hi):
    m, b = load()
    return stats.seasons(m, lo, hi), stats.seasons(b, lo, hi)


@st.cache_data(show_spinner=False)
def season_table(lo, hi):
    return stats.season_summary(*in_range(lo, hi))


@st.cache_data(show_spinner=False)
def batting_table(lo, hi):
    return stats.batting(in_range(lo, hi)[1])


@st.cache_data(show_spinner=False)
def bowling_table(lo, hi):
    return stats.bowling(in_range(lo, hi)[1])


@st.cache_data(show_spinner=False)
def chase_states():
    m, b = load()
    return winprob.chase_states(m, b)


def margin_text(n, kind):
    """'won by 1 run' / 'won by 23 runs' / 'won by 1 wicket'."""
    if pd.isna(n) or not isinstance(kind, str):
        return "tied (super over)" if kind == "tie" else ""
    n = int(n)
    return f"{n} {kind[:-1] if n == 1 and kind.endswith('s') else kind}"


def pct(x, digits=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}%"


def chart(c, height=300):
    st.altair_chart(c.properties(height=height).configure_view(strokeWidth=0)
                    .configure_axis(gridColor="#23304d", domainColor="#3a4a6e", labelColor="#b7c1d6",
                                    titleColor="#9aa6bd", labelFontSize=11, titleFontSize=11.5)
                    .configure_legend(labelColor="#b7c1d6", titleColor="#9aa6bd", orient="top")
                    .configure(background="transparent"), width="stretch")


def season_x(title=None):
    return alt.X("season:O", title=title, axis=alt.Axis(labelAngle=0, labelOverlap="parity"))


PHASE_X = alt.X("phase:N", sort=["Powerplay", "Middle", "Death"], title=None, axis=alt.Axis(labelAngle=0))


def era_color():
    return alt.Color("era:N", scale=alt.Scale(domain=["Before 2023", "Impact Player era (2023+)"], range=[BLUE, ORANGE]),
                     legend=alt.Legend(title=None))


# ------------------------------------------------------------------ sidebar
def sidebar():
    with st.sidebar:
        style.html(f'<div class="brand">{style.BALL_SVG}<div><div class="name">IPL Analytics</div>'
                   f'<div class="sub">17 seasons · 260,759 balls</div></div></div>')
        for page, icon in (("Overview", ":material/insights:"), ("Teams", ":material/groups:"),
                           ("Players", ":material/sports_cricket:"), ("Match replay", ":material/replay:"),
                           ("Win predictor", ":material/online_prediction:")):
            if st.button(page, icon=icon, width="stretch", key=f"nav_{page}",
                         type="primary" if state.page == page else "secondary"):
                state.page = page
                st.rerun()
        st.divider()
        lo, hi = st.slider("Seasons", 2008, 2024, (2008, 2024), key="seasons",
                           help="Filters the Overview, Teams and Players pages.")
        style.html("""<div class="foot">Data: <a href="https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020"
        style="color:#9aa6bd">IPL Complete Dataset 2008-2024</a> (Kaggle), ball by ball.<br>Built by Eshan Vijayvargiya ·
        <a href="https://github.com/Eshan-vijayvargiya/ipl-analytics" style="color:#9aa6bd">code</a></div>""")
    return lo, hi


# ------------------------------------------------------------------ page: overview
def overview(lo, hi):
    style.hero("What actually wins IPL matches?",
               "Seventeen seasons of ball-by-ball data: the scoring explosion, whether the toss matters, which grounds "
               "favour the chase, and who wins titles.")
    m, b = in_range(lo, hi)
    s = season_table(lo, hi)
    style.kpis([("Matches", f"{len(m):,}", f"{lo}-{hi}"),
                ("Runs scored", f"{int(b.total_runs.sum()):,}", f"{b.six.sum():,} sixes"),
                ("Avg first-innings score", f"{s.avg_first_innings.mean():.0f}",
                 f"{s.avg_first_innings.iloc[-1]:.0f} in {s.season.iloc[-1]}", "gold"),
                ("200+ totals", f"{int(s.totals_200.sum()):,}", f"{int(s.totals_200.iloc[-1])} in {s.season.iloc[-1]}"),
                ("Chasing team won", pct(stats.venues(m, 0).pipe(lambda v: (v.chase_win_pct * v.matches).sum() / v.matches.sum())),
                 "of decided matches")])

    tab1, tab2, tab3, tab4 = st.tabs(["Scoring revolution", "Toss & chasing", "Grounds", "Champions"])
    with tab1:
        metric = st.segmented_control("Show", ["Average first-innings score", "200+ totals", "Sixes per match",
                                               "Run rate by phase"], default="Average first-innings score",
                                      key="ov_metric")
        s["era"] = np.where(s.season >= 2023, "Impact Player era (2023+)", "Before 2023")
        if metric == "Run rate by phase":
            pr = stats.phase_rates(b)
            c = alt.Chart(pr).mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2).encode(
                x=season_x(), y=alt.Y("run_rate:Q", title="Runs per over", scale=alt.Scale(zero=False)),
                color=alt.Color("phase:N", sort=["Powerplay", "Middle", "Death"],
                                scale=alt.Scale(domain=["Powerplay", "Middle", "Death"], range=[AQUA, BLUE, ORANGE]),
                                legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("season:O"), alt.Tooltip("phase:N"), alt.Tooltip("run_rate:Q", format=".2f", title="Run rate")])
            chart(c, 330)
        else:
            col = {"Average first-innings score": "avg_first_innings", "200+ totals": "totals_200",
                   "Sixes per match": "sixes_per_match"}[metric or "Average first-innings score"]
            c = alt.Chart(s).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=26).encode(
                x=season_x(), y=alt.Y(f"{col}:Q", title=metric),
                color=era_color(),
                tooltip=[alt.Tooltip("season:O"), alt.Tooltip(f"{col}:Q", format=".1f", title=metric)])
            keep = {s.season.iloc[0], s.season.iloc[-1], s.loc[s[col].idxmax(), "season"]}
            text = alt.Chart(s[s.season.isin(keep)]).mark_text(dy=-8, color="#e8edf5", fontSize=11).encode(
                x="season:O", y=f"{col}:Q", text=alt.Text(f"{col}:Q", format=".0f"))
            chart(c + text, 330)
        style.card("What changed in 2023", """<p>The <b>Impact Player</b> rule lets each team bring in a substitute
        mid-match, usually an extra batter. With more batting to come, top orders attack from ball one:
        the average first-innings score rose from <b>171 (2022) to 190 (2024)</b>, 200+ innings went from
        <b>18 to 41</b>, and powerplay scoring jumped from <b>7.9 to 9.1 runs an over</b>.</p>""", insight=True)
    with tab2:
        dec = m[~m.no_result & m.winner.notna()]
        tw = dec.toss_winner_won.astype(bool).mean() * 100
        by = dec.groupby("toss_decision").toss_winner_won.apply(lambda x: x.astype(bool).mean() * 100)
        style.kpis([("Toss winner won", pct(tw, 1), "a coin flip is 50%"),
                    ("…after choosing to field", pct(by.get("field"), 1), f"{(dec.toss_decision == 'field').sum():,} matches", "gold"),
                    ("…after choosing to bat", pct(by.get("bat"), 1), f"{(dec.toss_decision == 'bat').sum():,} matches")])
        a, c2 = st.columns(2)
        with a:
            c = alt.Chart(s).mark_line(point=True, color=BLUE, strokeWidth=2).encode(
                x=season_x(), y=alt.Y("field_first_pct:Q", title="Toss winners choosing to field (%)",
                                                         scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("season:O"), alt.Tooltip("field_first_pct:Q", format=".0f", title="% chose to field")])
            chart(c, 260)
        with c2:
            rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
            c = alt.Chart(s).mark_line(point=True, color=AQUA, strokeWidth=2).encode(
                x=season_x(), y=alt.Y("chase_win_pct:Q", title="Chasing team won (%)",
                                                         scale=alt.Scale(domain=[0, 100])),
                tooltip=[alt.Tooltip("season:O"), alt.Tooltip("chase_win_pct:Q", format=".0f", title="% won chasing")])
            chart(c + rule, 260)
        style.card("The verdict", """<p>Winning the toss on its own is a coin flip (50.8%, p = 0.61 across all
        seasons). Choosing to <b>field</b> works better (53.9% vs 45.4%), and captains have learned it: about 3 in 4
        now bowl first. But in the Impact Player era, chasing got harder (45% of chases won in 2023).</p>""", insight=True)
    with tab3:
        mn = st.slider("Minimum matches at the ground", 5, 60, 20, key="ov_min")
        v = stats.venues(m, mn).reset_index()
        if v.empty:
            st.info("No ground has that many matches in these seasons. Lower the minimum.")
        else:
            v["label"] = v.venue + np.where(v.city.notna(), " (" + v.city.fillna("") + ")", "")
            v["side"] = np.where(v.chase_win_pct >= 50, "Favours the chase", "Favours batting first")
            c = alt.Chart(v).mark_bar(cornerRadiusEnd=4, size=16).encode(
                y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=340)),
                x=alt.X("chase_win_pct:Q", title="Chasing team won (%)", scale=alt.Scale(domain=[0, 100]),
                        axis=alt.Axis(values=[0, 25, 50, 75, 100])),
                color=alt.Color("side:N", scale=alt.Scale(domain=["Favours the chase", "Favours batting first"],
                                                          range=[AQUA, ORANGE]), legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("venue:N"), alt.Tooltip("matches:Q"),
                         alt.Tooltip("chase_win_pct:Q", format=".0f", title="% chases won"),
                         alt.Tooltip("field_first_pct:Q", format=".0f", title="% chose to field")])
            rule = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(x="x:Q")
            chart(c + rule, max(220, 26 * len(v)))
            st.caption("Across 2008-2024, Chepauk (Chennai) is the hardest place to chase (43%), and Jaipur (65%) "
                       "and Sharjah (64%) the easiest.")
    with tab4:
        champs = s.dropna(subset=["champion"])
        cells = "".join(f'<div><div class="y">{int(r.season)}</div><div class="t">{r.champion}</div></div>'
                        for r in champs.itertuples())
        style.html(f'<div class="champ">{cells}</div>')
        rec = stats.team_record(m).reset_index()
        st.dataframe(rec[["team", "played", "won", "win_pct", "titles", "first", "last"]], hide_index=True,
                     width="stretch", column_config={
                         "team": "Team", "played": "Played", "won": "Won",
                         "win_pct": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.0f%%"),
                         "titles": "Titles", "first": st.column_config.NumberColumn("First season", format="%d"),
                         "last": st.column_config.NumberColumn("Last season", format="%d")})


# ------------------------------------------------------------------ page: teams
def teams(lo, hi):
    m, b = in_range(lo, hi)
    rec = stats.team_record(m)
    names = rec.sort_values("played", ascending=False).index.tolist()
    team = st.selectbox("Team", names, index=names.index("Chennai Super Kings") if "Chennai Super Kings" in names else 0)
    r = rec.loc[team]
    finals = m[(m.match_type == "Final") & (m.winner == team)].season.tolist()
    style.hero(team, f"Seasons {int(r['first'])}-{int(r['last'])} · " +
               (f"Champions in {', '.join(map(str, finals))}" if finals else "No titles in these seasons"),
               eyebrow="Team profile")
    g = stats.team_games(m)
    tg = g[g.team == team]
    style.kpis([("Played", f"{int(r.played)}", f"{lo}-{hi}"), ("Won", f"{int(r.won)}", f"{r.win_pct:.0f}% win rate", "gold"),
                ("Titles", f"{int(r.titles)}", ", ".join(map(str, finals)) or "none"),
                ("Won chasing", pct(tg[tg.result == "wickets"].won.sum() / max(1, len(tg[tg.result == "wickets"])) * 100)
                 if False else pct((tg.won & (tg.result == "wickets")).sum() / max(1, tg.won.sum()) * 100), "share of their wins"),
                ("Playoff record", f"{int(tg[tg.match_type != 'League'].won.sum())}-{int((~tg[tg.match_type != 'League'].won).sum())}", "won-lost")])
    by = tg.groupby("season").won.agg(["mean", "size"]).reset_index()
    by["win_pct"] = by["mean"] * 100
    by["champion"] = np.where(by.season.isin(finals), "Champions", "Season")
    c = alt.Chart(by).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24).encode(
        x=season_x(), y=alt.Y("win_pct:Q", title="Matches won (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("champion:N", scale=alt.Scale(domain=["Season", "Champions"], range=[BLUE, GOLD]),
                        legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("season:O"), alt.Tooltip("win_pct:Q", format=".0f", title="Win %"),
                 alt.Tooltip("size:Q", title="Matches")])
    rule = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    chart(c + rule, 280)

    a, c2 = st.columns(2)
    tb = b[b.batting_team == team]
    with a:
        st.markdown("##### Top run scorers for the team")
        bat = stats.batting(tb).head(8).reset_index()
        st.dataframe(bat[["batter", "runs", "strike_rate", "average", "sixes"]], hide_index=True, width="stretch",
                     column_config={"batter": "Batter", "runs": "Runs",
                                    "strike_rate": st.column_config.NumberColumn("Strike rate", format="%.1f"),
                                    "average": st.column_config.NumberColumn("Average", format="%.1f"), "sixes": "6s"})
    with c2:
        st.markdown("##### Top wicket takers for the team")
        bw = stats.bowling(b[b.bowling_team == team]).head(8).reset_index()
        st.dataframe(bw[["bowler", "wickets", "economy", "best"]], hide_index=True, width="stretch",
                     column_config={"bowler": "Bowler", "wickets": "Wickets",
                                    "economy": st.column_config.NumberColumn("Economy", format="%.2f"), "best": "Best"})

    st.markdown("#### Head to head")
    opps = tg.opponent.value_counts().index.tolist()
    if not opps:
        return
    default = "Mumbai Indians" if "Mumbai Indians" in opps else opps[0]
    opp = st.selectbox("Opponent", opps, index=opps.index(default))
    h = tg[tg.opponent == opp].sort_values("date", ascending=False)
    style.kpis([("Meetings", f"{len(h)}", f"{lo}-{hi}"), (f"{ipl_data.SHORT.get(team, team)} won", f"{int(h.won.sum())}", "", "gold"),
                (f"{ipl_data.SHORT.get(opp, opp)} won", f"{int((~h.won).sum())}", "")])
    show = h.head(10).assign(result_text=[f"{ipl_data.SHORT.get(w, w)} won by {margin_text(n, k)}"
                                          if k != "tie" else f"Tie, {ipl_data.SHORT.get(w, w)} won the super over"
                                          for w, n, k in zip(h.head(10).winner, h.head(10).result_margin, h.head(10).result)])
    st.dataframe(show[["date", "season", "match_type", "venue", "result_text"]], hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("Date", format="D MMM YYYY"),
                                "season": st.column_config.NumberColumn("Season", format="%d"),
                                "match_type": "Stage", "venue": "Ground", "result_text": "Result"})


# ------------------------------------------------------------------ page: players
def players(lo, hi):
    m, b = in_range(lo, hi)
    bat, bowl = batting_table(lo, hi), bowling_table(lo, hi)
    score = bat.runs.reindex(bat.index.union(bowl.index)).fillna(0) + 25 * bowl.wickets.reindex(bat.index.union(bowl.index)).fillna(0)
    names = score[score > 150].sort_values(ascending=False).index.astype(str).tolist()
    if not names:
        st.info("Not enough data in these seasons.")
        return
    player = st.selectbox("Player", names, index=names.index("V Kohli") if "V Kohli" in names else 0,
                          help="Players with at least 150 runs, or 6 wickets, in the chosen seasons. Type to search.")
    is_bowl = player in bowl.index and bowl.loc[player, "balls"] >= 120 and bowl.loc[player, "wickets"] >= 6
    is_bat = player in bat.index and (bat.loc[player, "runs"] >= 150 or (not is_bowl and bat.loc[player, "balls"] >= 30))
    role = " · ".join(x for x, ok in (("Batter", is_bat), ("Bowler", is_bowl)) if ok)
    style.hero(player, f"{role or 'Player'} · seasons {lo}-{hi}", eyebrow="Player profile")
    league_sr = b[b.faced].groupby("phase", observed=True).apply(lambda d: d.batter_runs.sum() / len(d) * 100, include_groups=False)
    league_econ = b.groupby("phase", observed=True).apply(lambda d: d.bowler_runs.sum() / d.legal.sum() * 6, include_groups=False)

    if is_bat:
        r = bat.loc[player]
        style.kpis([("Runs", f"{int(r.runs):,}", f"{int(r.matches)} matches", "gold"),
                    ("Average", "–" if np.isnan(r.average) else f"{r.average:.1f}", f"{int(r.outs)} dismissals"),
                    ("Strike rate", f"{r.strike_rate:.1f}", "runs per 100 balls"),
                    ("50s / 100s", f"{int(r.fifties)} / {int(r.hundreds)}", f"best {int(r.best)}"),
                    ("Fours / sixes", f"{int(r.fours)} / {int(r.sixes)}", "")])
        a, c2 = st.columns(2)
        with a:
            ss = stats.by_season(b, player, "bat")
            c = alt.Chart(ss).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=BLUE, size=20).encode(
                x=season_x(), y=alt.Y("runs:Q", title="Runs"),
                tooltip=["season:O", "runs:Q", alt.Tooltip("strike_rate:Q", format=".1f", title="Strike rate")])
            st.markdown("##### Runs by season")
            chart(c, 250)
        with c2:
            ph = stats.by_phase(b, player, "bat")
            ph["league"] = ph.phase.map(league_sr)
            base = alt.Chart(ph).encode(x=PHASE_X)
            c = base.mark_bar(color=AQUA, size=46, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                y=alt.Y("strike_rate:Q", title="Strike rate"),
                tooltip=["phase:N", "runs:Q", "balls:Q", alt.Tooltip("strike_rate:Q", format=".1f"),
                         alt.Tooltip("league:Q", format=".1f", title="League average")])
            tick = base.mark_tick(color=GOLD, thickness=3, size=58).encode(y="league:Q")
            st.markdown("##### Strike rate by phase  <span style='color:#f2b630;font-size:.8rem'>▬ league average</span>",
                        unsafe_allow_html=True)
            chart(c + tick, 250)
    if is_bowl:
        r = bowl.loc[player]
        style.kpis([("Wickets", f"{int(r.wickets)}", f"{int(r.matches)} matches", "gold"),
                    ("Economy", f"{r.economy:.2f}", "runs per over"),
                    ("Average", "–" if np.isnan(r.average) else f"{r.average:.1f}", "runs per wicket"),
                    ("Best", str(r.best), "in a match"),
                    ("Dot balls", f"{r.dot_pct:.0f}%", f"boundaries {r.boundary_pct:.0f}%")])
        a, c2 = st.columns(2)
        with a:
            ss = stats.by_season(b, player, "bowl")
            c = alt.Chart(ss).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=ORANGE, size=20).encode(
                x=season_x(), y=alt.Y("wickets:Q", title="Wickets"),
                tooltip=["season:O", "wickets:Q", alt.Tooltip("economy:Q", format=".2f")])
            st.markdown("##### Wickets by season")
            chart(c, 250)
        with c2:
            ph = stats.by_phase(b, player, "bowl")
            ph["league"] = ph.phase.map(league_econ)
            base = alt.Chart(ph).encode(x=PHASE_X)
            c = base.mark_bar(color=BLUE, size=46, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                y=alt.Y("economy:Q", title="Economy (runs per over)"),
                tooltip=["phase:N", "wickets:Q", "balls:Q", alt.Tooltip("economy:Q", format=".2f"),
                         alt.Tooltip("league:Q", format=".2f", title="League average")])
            tick = base.mark_tick(color=GOLD, thickness=3, size=58).encode(y="league:Q")
            st.markdown("##### Economy by phase  <span style='color:#f2b630;font-size:.8rem'>▬ league average</span>",
                        unsafe_allow_html=True)
            chart(c + tick, 250)

    st.markdown("#### Batter vs bowler")
    a, c2 = st.columns(2)
    batters = bat[bat.balls >= 200].index.astype(str).tolist()
    bowlers = bowl[bowl.balls >= 300].index.astype(str).tolist()
    if batters and bowlers:
        bt = a.selectbox("Batter", batters, index=batters.index("V Kohli") if "V Kohli" in batters else 0, key="mu_bat")
        bw = c2.selectbox("Bowler", bowlers, index=bowlers.index("JJ Bumrah") if "JJ Bumrah" in bowlers else 0, key="mu_bowl")
        mu = stats.matchup(b, bt, bw)
        if mu["balls"] == 0:
            st.info(f"{bt} never faced {bw} in these seasons.")
        else:
            style.kpis([("Balls", f"{mu['balls']}", ""), ("Runs", f"{mu['runs']}", f"strike rate {mu['strike_rate']:.0f}", "gold"),
                        ("Dismissals", f"{mu['outs']}", f"by {bw}"), ("Dots / boundaries", f"{mu['dots']} / {mu['boundaries']}", "")])

    with st.expander("Leaderboards for the chosen seasons"):
        a, c2 = st.columns(2)
        mb = a.slider("Minimum balls faced", 100, 3000, 500, step=100)
        top = bat[bat.balls >= mb].head(20).reset_index()
        a.dataframe(top[["batter", "runs", "average", "strike_rate", "sixes"]], hide_index=True, width="stretch",
                    column_config={"strike_rate": st.column_config.NumberColumn("SR", format="%.1f"),
                                   "average": st.column_config.NumberColumn("Avg", format="%.1f")})
        mw = c2.slider("Minimum balls bowled", 100, 3000, 600, step=100)
        topb = bowl[bowl.balls >= mw].head(20).reset_index()
        c2.dataframe(topb[["bowler", "wickets", "economy", "average", "best"]], hide_index=True, width="stretch",
                     column_config={"economy": st.column_config.NumberColumn("Econ", format="%.2f"),
                                    "average": st.column_config.NumberColumn("Avg", format="%.1f")})


# ------------------------------------------------------------------ page: match replay
def replay():
    m, b = load()
    mdl, info = model()
    style.hero("Match replay", "Pick any match and watch the chasing team's chance of winning change ball by ball, "
               "from the win-probability model.", eyebrow="Ball-by-ball win probability")
    a, c2 = st.columns([1, 3])
    season = a.selectbox("Season", sorted(m.season.unique(), reverse=True), index=sorted(m.season.unique(), reverse=True).index(2019))
    ms = m[m.season == season].sort_values("date", ascending=False)
    ms = ms.assign(label=ms.date.dt.strftime("%d %b") + " · " + ms.match_type + " · " + ms.bat_first_short + " vs " +
                   ms.bat_second_short + " · " + np.where(ms.no_result, "no result",
                                                          ms.winner_short.fillna("") + " won"))
    labels = ms.label.tolist()
    default = next((i for i, (t, l) in enumerate(zip(ms.match_type, labels)) if t == "Final"), 0)
    pick = c2.selectbox("Match", labels, index=default)
    row = ms.iloc[labels.index(pick)]
    mb = b[b.match_id == row.match_id]
    tot = stats.scorecard_totals(b, row.match_id)
    margin = ("No result" if row.no_result else f"Tie, {row.winner} won the super over" if row.result == "tie"
              else f"{row.winner} won by {margin_text(row.result_margin, row.result)}")
    chips = "".join(f'<span class="chip">{ipl_data.SHORT.get(t.batting_team, t.batting_team)} <b>{int(t.runs)}/{int(t.wkts)}</b> ({t.overs} ov)</span>'
                    for t in tot.itertuples())
    style.card(f"{row.bat_first} vs {row.bat_second}",
               f"<p>{chips}</p><p>{row.date:%d %B %Y} · {row.venue}, {row.city} · {row.match_type}</p>"
               f"<p><b style='color:#f2b630'>{margin}</b>"
               + (f" · Player of the match: {row.player_of_match}" if isinstance(row.player_of_match, str) else "") + "</p>")
    if row.no_result or row.dls or row.super_over or mb[mb.inning == 2].empty:
        st.info("This match had no result, a rain-revised (D/L) target or a super over, so the win-probability "
                "model doesn't replay it. The run charts below still work.")
    else:
        target = int(mb[mb.inning == 1].total_runs.sum() + 1)
        start, w = winprob.worm(mdl, mb, target, bool(row.impact_era))
        chaser = row.bat_second_short
        w = pd.concat([pd.DataFrame({"ball_no": [0], "over": [0], "ball": [0], "batter": [""], "bowler": [""], "runs": [0],
                                     "wicket": [False], "score": [0], "wkts": [0], "win_prob": [start]}), w],
                      ignore_index=True)
        w["pct"] = w.win_prob * 100
        w["up"], w["down"] = w.pct.clip(lower=50), w.pct.clip(upper=50)
        w["label"] = np.where(w.ball_no == 0, "Start of the chase", "Over " + (w.over - 1).astype(str) + "." + w.ball.astype(str))
        w["event"] = np.where(w.wicket, "WICKET", w.runs.astype(str) + " run(s)")
        w["state"] = chaser + " " + w.score.astype(str) + "/" + w.wkts.astype(str)
        w["change"] = w.pct.diff().fillna(0)
        tips = [alt.Tooltip("label:N", title="Ball"), alt.Tooltip("batter:N"), alt.Tooltip("bowler:N"),
                alt.Tooltip("event:N", title="What happened"), alt.Tooltip("state:N", title="Score"),
                alt.Tooltip("pct:Q", format=".0f", title=f"{chaser} win chance %")]
        base = alt.Chart(w).encode(x=alt.X("ball_no:Q", title=f"Ball of {chaser}'s chase (target {target})",
                                           scale=alt.Scale(domain=[0, int(w.ball_no.max())], nice=False)))
        up = base.mark_area(color=AQUA, opacity=0.25).encode(y=alt.Y("up:Q", title=f"{chaser}'s chance of winning (%)",
                                                                     scale=alt.Scale(domain=[0, 100])), y2=alt.datum(50))
        down = base.mark_area(color=ORANGE, opacity=0.25).encode(y="down:Q", y2=alt.datum(50))
        line = base.mark_line(color="#f4f6fb", strokeWidth=2).encode(y="pct:Q")
        wk = base.transform_filter("datum.wicket").mark_point(color=ORANGE, filled=True, size=90).encode(y="pct:Q", tooltip=tips)
        hover = base.mark_point(opacity=0, size=200).encode(y="pct:Q", tooltip=tips)
        mid = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
        chart(up + down + mid + line + wk + hover, 360)
        swings = w.loc[w.change.abs().sort_values(ascending=False).index].head(5)
        rows = "".join(f"<li><b>{r.label}</b>: {r.batter} vs {r.bowler}, {r.event.lower()} → {chaser} "
                       f"{'up' if r.change > 0 else 'down'} <b>{abs(r.change):.0f} points</b> to {r.pct:.0f}%</li>"
                       for r in swings.itertuples())
        style.card("Biggest swings", f"<ul style='margin:.2rem 0 0 1rem;padding:0'>{rows}</ul>")
    ob = stats.over_by_over(b, row.match_id)
    ob["team"] = ob.batting_team.map(ipl_data.SHORT)
    c = alt.Chart(ob).mark_bar(size=11, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("over:O", title="Over"), xOffset="team:N", y=alt.Y("runs:Q", title="Runs in the over"),
        color=alt.Color("team:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(title=None)),
        tooltip=["team:N", "over:O", "runs:Q", "wickets:Q", alt.Tooltip("cumulative:Q", title="Total")])
    st.markdown("##### Runs per over")
    chart(c, 240)


# ------------------------------------------------------------------ page: win predictor
def predictor():
    mdl, info = model()
    style.hero("Win predictor", "Set up any chase and get the chasing team's chance of winning, the same model "
               "that powers the match replays.", eyebrow="What are our chances?")
    a, b_, c, d, e = st.columns(5)
    target = a.number_input("Target", 50, 300, 190)
    score = b_.number_input("Score now", 0, 299, 120)
    wk = c.number_input("Wickets lost", 0, 10, 4)
    overs = d.number_input("Overs done", 0, 20, 14)
    balls_in = e.number_input("Balls into the over", 0, 5, 0)
    era = st.toggle("Impact Player era (2023 onwards)", value=True,
                    help="Scoring is higher since 2023, so the same chase is easier.")
    done = min(120, overs * 6 + balls_in)
    need, left, wl = target - score, 120 - done, 10 - wk
    if need <= 0:
        st.success("The target has already been reached: the chasing team has won.")
        return
    if left <= 0 or wl <= 0:
        st.error("No balls or wickets left: the chasing team has lost.")
        return
    p = winprob.probability(mdl, need, left, wl, target, era)
    color = AQUA if p >= 0.5 else ORANGE
    rr_need, rr_now = need / left * 6, (score / done * 6) if done else 0
    left_col, right_col = st.columns([1, 1.4])
    with left_col:
        style.html(f"""<div class="card"><div style="color:#9aa6bd;font-size:.85rem">Chasing team's chance of winning</div>
        <div class="bigprob" style="color:{color}">{p:.0%}</div>
        <div class="bar"><span style="width:{p * 100:.1f}%;background:{color}"></span></div>
        <p>Need <b>{need}</b> off <b>{left}</b> balls with <b>{wl}</b> wickets in hand.</p>
        <p>Required rate <b>{rr_need:.2f}</b> · current rate <b>{rr_now:.2f}</b></p></div>""")
        S = chase_states()
        sim = S[S.runs_needed.between(need - 4, need + 4) & S.balls_left.between(left - 3, left + 3) &
                S.wickets_left.between(wl - 1, wl + 1)]
        if len(sim) >= 10:
            n_matches = sim.match_id.nunique()
            real = sim.groupby("match_id").chase_won.first().mean()
            style.card("Real IPL chases like this", f"<p>In <b>{n_matches}</b> real chases that reached a similar point "
                       f"(±4 runs, ±3 balls, ±1 wicket), the chasing team won <b>{real:.0%}</b>.</p>")
        else:
            style.card("Real IPL chases like this", "<p>Very few real chases reached this exact point, so the model's "
                       "estimate is the best guide here.</p>")
    with right_col:
        if left >= 6:
            rows = []
            for runs in range(0, 25):
                for wicket, name in ((0, "No wicket"), (1, "Lose a wicket")):
                    if wl - wicket <= 0:
                        continue
                    rows.append({"runs": runs, "case": name,
                                 "prob": winprob.probability(mdl, need - runs, left - 6, wl - wicket, target, era) * 100})
            nx = pd.DataFrame(rows)
            c = alt.Chart(nx).mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=30)).encode(
                x=alt.X("runs:Q", title="Runs scored in the next over"),
                y=alt.Y("prob:Q", title="Chance of winning after it (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("case:N", scale=alt.Scale(domain=["No wicket", "Lose a wicket"], range=[AQUA, ORANGE]),
                                legend=alt.Legend(title=None)),
                tooltip=[alt.Tooltip("runs:Q", title="Runs in the over"), "case:N",
                         alt.Tooltip("prob:Q", format=".0f", title="Win chance %")])
            now = alt.Chart(pd.DataFrame({"y": [p * 100]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
            st.markdown("##### What the next over could do")
            chart(c + now, 330)
            st.caption("Dashed line: the chance right now. Where the lines cross it is how many runs the next over "
                       "needs to keep the chase on track.")
    t = info["test_on_unseen_2022_24"]
    st.caption(f"Model: {info['model']} trained on {info['states']:,} game states from {info['chases']:,} IPL chases. "
               f"On seasons it hadn't seen (2022-24) it scored AUC {t['Test AUC']:.2f} (vs {info['baseline_auc']:.2f} "
               f"using the required rate alone). Ties decided by a super over aren't counted as wins or losses.")


# ------------------------------------------------------------------ router
lo, hi = sidebar()
{"Overview": lambda: overview(lo, hi), "Teams": lambda: teams(lo, hi), "Players": lambda: players(lo, hi),
 "Match replay": replay, "Win predictor": predictor}[state.page]()
