"""Volleyball section: German Bundesliga, men and women, 2013/14 to 2018/19."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import rally
import style
import volleyball as vb
from ui import AQUA, BLUE, GOLD, MUTED, ORANGE, chart, pct, prob_bar

ROOT = Path(__file__).resolve().parent.parent
GREY = "#7d8aa5"
SCORES = ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]


@st.cache_data(show_spinner="Loading six Bundesliga seasons…")
def load():
    g = pd.read_parquet(vb.DATA / "matches.parquet")
    s = pd.read_parquet(vb.DATA / "sets.parquet")
    p = pd.read_parquet(vb.DATA / "player_stats.parquet")
    return g, s, p


@st.cache_resource
def model():
    return joblib.load(ROOT / "models" / "volleyball_model.joblib"), json.loads((ROOT / "models" / "volleyball_info.json").read_text())


@st.cache_data(show_spinner=False)
def champions():
    g, _, _ = load()
    out = []
    for (gender, season), d in g[g.stage == "Final"].groupby(["gender", "season"]):
        w = pd.concat([d[d.home_won].home, d[~d.home_won].away]).value_counts()
        out.append({"gender": gender, "season": season, "champion": w.index[0]})
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def ratings(gender):
    """Latest Elo of each team in one league (rating after its last match)."""
    g, _, _ = load()
    d = g[g.gender == gender]
    last = {}
    for r in d.itertuples():
        exp = 1 / (1 + 10 ** (-(r.elo_home + 40 - r.elo_away) / 400))
        mult = {3: 1.25, 2: 1.0, 1: 0.75}.get(abs(r.home_sets - r.away_sets), 1.0)
        ch = 30 * mult * ((1.0 if r.home_won else 0.0) - exp)
        last[r.home], last[r.away] = r.elo_home + ch, r.elo_away - ch
    return pd.Series(last).sort_values(ascending=False)


def team_rows(g, team):
    h = g[g.home == team].assign(opponent=lambda d: d.away, won=lambda d: d.home_won, venue="Home",
                                 sets=lambda d: d.home_sets.astype(str) + "-" + d.away_sets.astype(str))
    a = g[g.away == team].assign(opponent=lambda d: d.home, won=lambda d: ~d.home_won, venue="Away",
                                 sets=lambda d: d.away_sets.astype(str) + "-" + d.home_sets.astype(str))
    return pd.concat([h, a]).sort_values("date")


# ------------------------------------------------------------------ overview
def overview():
    g, s, p = load()
    (mdl, info) = model()
    style.hero("What wins a volleyball match?", "Six seasons of the German Bundesliga, men's and women's: 1,696 matches, every set "
               "score and every player's attacks, serves and blocks.", eyebrow="🏐 Volleyball · Bundesliga 2013-2019")
    hi = s[["home_pts", "away_pts"]].max(axis=1)
    style.kpis([("Matches", f"{len(g):,}", f"{len(s):,} sets"), ("Home team won", pct(g.home_won.mean() * 100), "men 59% · women 55%", "gold"),
                ("Straight-sets wins", pct(g.result.isin(['3-0', '0-3']).mean() * 100), "3-0 either way"),
                ("Sets that went past 25", pct(((s.set < 5) & (hi > 25)).mean() * 100), f"longest: {int(hi.max())} points"),
                ("Fans a match", f"{g.spectators.mean():,.0f}", f"{g.minutes.mean():.0f} minutes long")])
    t1, t2, t3 = st.tabs(["Final scores: model vs reality", "Champions", "Players who score"])
    with t1:
        fs = info["final_scores"]
        d = pd.DataFrame([{"score": k, "Model": v["predicted"] * 100, "Real": v["real"] * 100} for k, v in fs.items()])
        long = d.melt("score", var_name="Which", value_name="share")
        c = alt.Chart(long).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=22).encode(
            x=alt.X("score:N", sort=SCORES, title="Final score (home first)", axis=alt.Axis(labelAngle=0)), xOffset="Which:N",
            y=alt.Y("share:Q", title="Share of matches (%)"), color=alt.Color("Which:N", scale=alt.Scale(range=[GREY, GOLD]), legend=alt.Legend(title=None)),
            tooltip=["score:N", "Which:N", alt.Tooltip("share:Q", format=".1f")])
        chart(c, 300)
        style.card("Real matches are more one-sided than the maths expects", f"""<p>If every rally were an independent coin toss
        (weighted by team strength), {fs['3-0']['predicted']:.0%} of matches would end 3-0 to the home team. In reality
        <b>{fs['3-0']['real']:.0%}</b> do, and fewer go to five sets. Teams go on runs: a team that is on top tends to stay
        on top. That's a known limit of the rally (Markov) model, and an honest finding from the test season.</p>""", insight=True)
    with t2:
        ch = champions()
        for gender in ("Men", "Women"):
            cells = "".join(f'<div><div class="y">{r.season}</div><div class="t" style="font-size:.86rem">{r.champion.replace(" Volleys", "").replace("BERLIN RECYCLING", "Berlin")}</div></div>'
                            for r in ch[ch.gender == gender].itertuples())
            st.markdown(f"##### {gender}")
            style.html(f'<div class="champ" style="grid-template-columns:repeat(auto-fill,minmax(150px,1fr))">{cells}</div>')
    with t3:
        pos = p.groupby("position").agg(points=("points", "mean"), kills=("kills", "mean"), blocks=("blocks", "mean"), aces=("aces", "mean")).reset_index()
        long = pos.melt("position", ["kills", "blocks", "aces"], var_name="How", value_name="per_match")
        long["How"] = long.How.map({"kills": "Attack points", "blocks": "Block points", "aces": "Aces"})
        c = alt.Chart(long).mark_bar(cornerRadiusEnd=3, size=18).encode(
            y=alt.Y("position:N", sort="-x", title=None), x=alt.X("per_match:Q", title="Points per match", stack="zero"),
            color=alt.Color("How:N", scale=alt.Scale(range=[BLUE, AQUA, GOLD]), legend=alt.Legend(title=None)),
            tooltip=["position:N", "How:N", alt.Tooltip("per_match:Q", format=".2f")])
        chart(c, 240)
        st.caption("Where points come from, by position: outside spikers and diagonals (opposites) score with attacks, middle "
                   "blockers add blocks, and the libero never scores (it isn't allowed to attack above the net).")


# ------------------------------------------------------------------ teams
def teams():
    g, s, p = load()
    a, b = st.columns([1, 3])
    gender = a.segmented_control("League", ["Men", "Women"], default="Men", key="vb_gender") or "Men"
    gg = g[g.gender == gender]
    names = pd.concat([gg.home, gg.away]).value_counts()
    team = b.selectbox("Team", names.index.tolist(), index=0, key=f"vb_team_{gender}")
    t = team_rows(gg, team)
    r_ = ratings(gender)
    style.hero(team, f"{gender}'s Bundesliga · {len(t)} matches, {t.season.min()} to {t.season.max()}", eyebrow="🏐 Team profile")
    titles = (champions().query("gender == @gender").champion == team).sum()
    style.kpis([("Played", f"{len(t)}", f"{int(t.won.sum())} won · {int((~t.won).sum())} lost"), ("Won", pct(t.won.mean() * 100), "of matches", "gold"),
                ("Home / away", f"{t[t.venue == 'Home'].won.mean():.0%} / {t[t.venue == 'Away'].won.mean():.0%}", "win rate"),
                ("Titles", f"{titles}", "2014-2019"), ("Elo now", f"{r_.get(team, np.nan):.0f}", f"No. {list(r_.index).index(team) + 1} in the league")])
    e = pd.DataFrame({"date": t.date, "elo": np.where(t.venue == "Home", t.elo_home, t.elo_away), "opponent": t.opponent, "sets": t.sets})
    st.markdown("##### Elo rating, match by match")
    chart(alt.Chart(e).mark_line(color=GOLD, strokeWidth=1.8).encode(
        x=alt.X("date:T", title=None), y=alt.Y("elo:Q", title="Elo rating", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), "opponent:N", "sets:N", alt.Tooltip("elo:Q", format=".0f")]), 240)
    a, b = st.columns(2)
    with a:
        st.markdown("##### Season by season")
        ss = t.groupby("season").agg(Played=("won", "size"), Won=("won", "sum")).reset_index()
        ss["Win %"] = ss.Won / ss.Played * 100
        ch = champions().query("gender == @gender")
        ss["Champion"] = ss.season.map(dict(zip(ch.season, ch.champion))) == team
        st.dataframe(ss, hide_index=True, width="stretch", column_config={
            "season": "Season", "Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.0f%%"),
            "Champion": st.column_config.CheckboxColumn("Title")})
    with b:
        st.markdown("##### Top scorers")
        tp = p[(p.team == team) & (p.gender == gender)].groupby("name").agg(points=("points", "sum"), matches=("uid", "nunique"), position=("position", "first"))
        tp["per match"] = tp.points / tp.matches
        st.dataframe(tp.sort_values("points", ascending=False).head(8).reset_index(), hide_index=True, width="stretch",
                     column_config={"name": "Player", "points": st.column_config.NumberColumn("Points", format="%d"), "matches": "Matches",
                                    "position": "Position", "per match": st.column_config.NumberColumn("Per match", format="%.1f")})


# ------------------------------------------------------------------ players
def players():
    g, s, p = load()
    a, b, c = st.columns([1, 1.2, 1])
    gender = a.segmented_control("League", ["Men", "Women"], default="Men", key="vb_pl_gender") or "Men"
    pp = p[p.gender == gender]
    pos = b.selectbox("Position", ["All positions"] + sorted(pp.position.dropna().unique()), key=f"vb_pos_{gender}")
    if pos != "All positions":
        pp = pp[pp.position == pos]
    stat = c.selectbox("Rank by", ["Points", "Attack points", "Aces", "Blocks", "Attack success %"], key="vb_stat")
    agg = pp.groupby(["name", "team"]).agg(matches=("uid", "nunique"), points=("points", "sum"), kills=("kills", "sum"), attacks=("attacks", "sum"),
                                         aces=("aces", "sum"), blocks=("blocks", "sum"), height=("height", "first"), position=("position", "first")).reset_index()
    agg = agg.groupby("name").agg(team=("team", lambda x: " / ".join(dict.fromkeys(x))), matches=("matches", "sum"), points=("points", "sum"),
                                  kills=("kills", "sum"), attacks=("attacks", "sum"), aces=("aces", "sum"), blocks=("blocks", "sum"),
                                  height=("height", "first"), position=("position", "first")).reset_index()
    agg["Attack success %"] = agg.kills / agg.attacks.replace(0, np.nan) * 100
    col = {"Points": "points", "Attack points": "kills", "Aces": "aces", "Blocks": "blocks", "Attack success %": "Attack success %"}[stat]
    q = agg[agg.matches >= 20] if stat != "Attack success %" else agg[agg.attacks >= 300]
    top = q.sort_values(col, ascending=False).head(15)
    style.hero(f"Top {stat.lower()}", f"{gender}'s Bundesliga 2013-2019 · {pos.lower()}", eyebrow="🏐 Players")
    chart(alt.Chart(top).mark_bar(color=BLUE, cornerRadiusEnd=4, size=15).encode(
        y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)), x=alt.X(f"{col}:Q", title=stat),
        tooltip=["name:N", "team:N", "position:N", "matches:Q", alt.Tooltip(f"{col}:Q", format=",.1f")]), 24 * len(top) + 20)
    sc = agg[(agg.attacks >= 300) & agg.height.notna()]
    if len(sc) > 10:
        st.markdown("##### Does height help? Attack success vs height")
        chart(alt.Chart(sc).mark_circle(size=60, opacity=0.7, color=AQUA).encode(
            x=alt.X("height:Q", title="Height (cm)", scale=alt.Scale(zero=False)), y=alt.Y("Attack success %:Q", scale=alt.Scale(zero=False)),
            tooltip=["name:N", "team:N", "position:N", "height:Q", alt.Tooltip("Attack success %:Q", format=".1f"), "attacks:Q"]), 280)
        r = sc[["height", "Attack success %"]].corr().iloc[0, 1]
        st.caption(f"Correlation r = {r:.2f} among players with 300+ attacks. Height helps less than you might think once a "
                   f"player is already at Bundesliga level.")


# ------------------------------------------------------------------ match replay
def replay():
    g, s, p = load()
    (mdl, info) = model()
    style.hero("Match replay", "Pick a match and see the chance of winning before the first serve and after every set.",
               eyebrow="🏐 Set-by-set win probability")
    a, b, c = st.columns([0.8, 1, 2.6])
    gender = a.selectbox("League", ["Men", "Women"], key="vb_rp_gender")
    seasons = sorted(g.season.unique(), reverse=True)
    season = b.selectbox("Season", seasons, key=f"vb_rp_season_{gender}")
    gs = g[(g.gender == gender) & (g.season == season)]
    gs = gs.assign(rank=gs.stage.map({"Final": 0, "Semifinal": 1, "Quarterfinal": 2}).fillna(3)).sort_values(["rank", "date"], ascending=[True, False])
    labels = [f"{r.date:%d %b %Y} · {r.stage} · {r.home} {r.result} {r.away}" for r in gs.itertuples()]
    pick = c.selectbox("Match", labels, index=0, key=f"vb_rp_match_{gender}_{season}")
    row = gs.iloc[labels.index(pick)]
    rp = vb.replay(mdl["coef"], row, s, mdl["strength"])
    style.card(f"{row.home} {row.result} {row.away}", f"<p>{row.date:%d %B %Y} · {row.stage} · {int(row.spectators):,} fans · "
               f"{int(row.minutes)} minutes</p><p>Sets: <b style='color:#f2b630'>{row.score_line}</b> · before the first serve: Elo "
               f"{row.elo_home:.0f} vs {row.elo_away:.0f}, so {row.home} were expected to win <b>{rp.p_rally.iloc[0]:.1%}</b> of rallies</p>")
    rp["pct"] = rp.home_win * 100
    rp["step"] = np.arange(len(rp))
    c = alt.Chart(rp).mark_line(point=alt.OverlayMarkDef(size=90, filled=True, color=GOLD), color="#f4f6fb", strokeWidth=2.5).encode(
        x=alt.X("after:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct:Q", title=f"{row.home}'s chance of winning (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("after:N", title="After"), alt.Tooltip("pct:Q", format=".0f", title="Win %"),
                 alt.Tooltip("p_rally:Q", format=".1%", title="Rally win chance")])
    mid = alt.Chart(pd.DataFrame({"y": [50]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    chart(mid + c, 320)
    st.caption("The data has the score of every set, not every rally, so this replay moves set by set. After each set the "
               "rally estimate is nudged by the points actually won.")


# ------------------------------------------------------------------ predictor
def predictor():
    g, s, p = load()
    (mdl, info) = model()
    style.hero("Match predictor", "Pick two teams from the same league: Elo gives each side's chance of winning a rally, and the "
               "Markov chain turns it into the chance of every final score.", eyebrow="🏐 Who wins?")
    gender = st.segmented_control("League", ["Men", "Women"], default="Men", key="vb_p_gender") or "Men"
    r_ = ratings(gender)
    names = r_.index.tolist()
    a, b = st.columns(2)
    home = a.selectbox("Home team", names, index=0, key=f"vb_ph_{gender}")
    away = b.selectbox("Away team", names, index=1, key=f"vb_pa_{gender}")
    if home == away:
        st.info("Pick two different teams.")
        return
    pr = vb.rally_p(mdl["coef"], r_[home], r_[away])
    pm = rally.match(0, 0, pr, "volleyball")
    style.kpis([(f"{home} Elo", f"{r_[home]:.0f}", ""), (f"{away} Elo", f"{r_[away]:.0f}", ""),
                ("Rally win chance", f"{pr:.1%}", f"for {home}", "gold")])
    prob_bar([(home, pm, AQUA), (away, 1 - pm, ORANGE)])
    left, right = st.columns([1.2, 1])
    with left:
        sc = rally.set_score_chances(pr, "volleyball")
        d = pd.DataFrame({"score": list(sc), "p": [v * 100 for v in sc.values()]})
        d["side"] = np.where(d.score.str.startswith("3"), home, away)
        st.markdown("##### Chance of each final score")
        chart(alt.Chart(d).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=34).encode(
            x=alt.X("score:N", sort=SCORES, title=f"Sets ({home} first)", axis=alt.Axis(labelAngle=0)), y=alt.Y("p:Q", title="Chance (%)"),
            color=alt.Color("side:N", scale=alt.Scale(domain=[home, away], range=[AQUA, ORANGE]), legend=None),
            tooltip=["score:N", alt.Tooltip("p:Q", format=".1f")]), 260)
    with right:
        st.markdown("##### Live: during the match")
        sh = st.number_input(f"{home} sets", 0, 2, 1, key="vb_sh")
        sa_ = st.number_input(f"{away} sets", 0, 2, 1, key="vb_sa")
        c1, c2 = st.columns(2)
        ph_ = c1.number_input(f"{home} points", 0, 40, 20, key="vb_pts_h")
        pa_ = c2.number_input(f"{away} points", 0, 40, 22, key="vb_pts_a")
        live = rally.match(int(sh), int(sa_), pr, "volleyball", int(ph_), int(pa_))
        prob_bar([(home, live, AQUA), (away, 1 - live, ORANGE)])
    res = info["results"]["Rally Markov model (final)"]
    st.caption(f"Tested on the 2018/19 season after learning from 2013-2018: it picked the winner of {res['Accuracy']:.0%} of "
               f"{info['test_matches']} matches before the first serve (AUC {res['AUC']:.2f}) and "
               f"{info['after_sets']['after set 2']['Accuracy']:.0%} after two sets.")


PAGES = {"Overview": overview, "Teams": teams, "Players": players, "Match replay": replay, "Predictor": predictor}
