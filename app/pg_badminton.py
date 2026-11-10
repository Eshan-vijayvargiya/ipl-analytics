"""Badminton section: 94 top BWF singles matches (2018-2022), every rally and every shot."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import badminton as bd
import rally
import style
from ui import AQUA, BLUE, GOLD, MUTED, ORANGE, chart, pct, prob_bar, two_way_worm

ROOT = Path(__file__).resolve().parent.parent
GREY = "#7d8aa5"
MEN = "Men's singles"


@st.cache_data(show_spinner="Loading every shot…")
def load():
    m = pd.read_parquet(bd.DATA / "matches.parquet")
    r = pd.read_parquet(bd.DATA / "rallies.parquet")
    s = pd.read_parquet(bd.DATA / "strokes.parquet")
    return m, r, s


@st.cache_resource
def model():
    return joblib.load(ROOT / "models" / "badminton_model.joblib"), json.loads((ROOT / "models" / "badminton_info.json").read_text())


@st.cache_data(show_spinner=False)
def rally_view():
    """One row per rally with the names of who won it and who hit the last shot."""
    m, r, _ = load()
    d = r.merge(m[["match_id", "winner", "loser", "event", "tournament", "date"]], on="match_id")
    d["rally_winner"] = np.where(d.a_won, d.winner, d.loser)
    d["rally_loser"] = np.where(d.a_won, d.loser, d.winner)
    d["last_hitter"] = np.where(d.last_by_a, d.winner, d.loser)
    d["kind"] = np.where(d.ending.str.startswith("Winner"), "Winner", np.where(d.ending.str.startswith("Error") | (d.ending == "Fault"), "Error", "Not labelled"))
    return d


def match_label(r):
    flag = "🇮🇳 " if r.india else ""
    return f"{flag}{bd.short(r.winner)} beat {bd.short(r.loser)} {r.score_line} · {r.tournament} {r.round}"


def players_list():
    m, _, _ = load()
    p = pd.concat([m.winner, m.loser]).value_counts()
    return p


# ------------------------------------------------------------------ overview
def overview():
    m, r, s = load()
    d = rally_view()
    (mdl, info) = model()
    style.hero("How top badminton points are won", "94 BWF World Tour singles matches (2018-2022), labelled shot by shot from the "
               "broadcast video: 7,888 rallies and 81,939 shots, including Lakshya Sen, PV Sindhu, Srikanth, Prannoy and "
               "Sameer Verma.", eyebrow="🏸 Badminton · BWF singles")
    lab = d[d.kind != "Not labelled"]
    style.kpis([("Matches", f"{len(m)}", f"{(m.event == MEN).sum()} men's · {(m.event != MEN).sum()} women's"),
                ("Rallies", f"{len(r):,}", f"{len(s):,} shots"), ("Average rally", f"{r.shots.mean():.1f} shots", "median 8"),
                ("Points lost by an error", pct((lab.kind == 'Error').mean() * 100), "not won by a winner", "gold"),
                ("Server won the rally", pct((r.a_won == r.server_a).mean() * 100, 1), "no serving advantage")])
    t1, t2, t3, t4 = st.tabs(["How points end", "Shots that win and lose", "Pressure points", "Rally length"])
    with t1:
        e = lab.ending.value_counts(normalize=True).rename("share").reset_index()
        e["share"] *= 100
        e["kind"] = np.where(e.ending.str.startswith("Winner"), "Won by a winner", "Lost by an error")
        c = alt.Chart(e).mark_bar(cornerRadiusEnd=4, size=20).encode(
            y=alt.Y("ending:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)), x=alt.X("share:Q", title="Share of rallies (%)"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Won by a winner", "Lost by an error"], range=[AQUA, ORANGE]), legend=alt.Legend(title=None)),
            tooltip=["ending:N", alt.Tooltip("share:Q", format=".1f")])
        chart(c, 250)
        style.card("Two points in three are lost, not won", f"""<p>Even at the very top, only <b>{(lab.kind == 'Winner').mean():.0%}</b>
        of rallies end with a shot the opponent can't reach. The rest end with a mistake: the shuttle <b>hit out</b>
        ({(lab.ending == 'Error: hit out').mean():.0%}), <b>into the net</b> ({(lab.ending.isin(['Error: hit the net', "Error: didn't clear the net"])).mean():.0%})
        or <b>left alone</b> when it was landing in ({(lab.ending == 'Error: misjudged (left it, it landed in)').mean():.0%}).
        Forcing errors matters as much as hitting winners.</p>""", insight=True)
    with t2:
        uses = s.shot.value_counts()
        win = lab[lab.kind == "Winner"].last_shot.value_counts()
        err = lab[lab.kind == "Error"].last_shot.value_counts()
        t = pd.DataFrame({"uses": uses, "winners": win, "errors": err}).fillna(0)
        t = t[(t.uses >= 900) & (t.index != "Unknown")]
        t["Winner"] = t.winners / t.uses * 100
        t["Error"] = t.errors / t.uses * 100
        long = t.reset_index().rename(columns={"index": "shot"}).melt(["shot", "uses"], ["Winner", "Error"], var_name="Outcome", value_name="rate")
        order = t.sort_values("Winner", ascending=False).index.tolist()
        c = alt.Chart(long).mark_bar(cornerRadiusEnd=3, size=11).encode(
            y=alt.Y("shot:N", sort=order, title=None), x=alt.X("rate:Q", title="% of times the shot ended the rally"),
            yOffset="Outcome:N", color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Winner", "Error"], range=[AQUA, ORANGE]), legend=alt.Legend(title=None)),
            tooltip=["shot:N", "Outcome:N", alt.Tooltip("rate:Q", format=".1f"), alt.Tooltip("uses:Q", format=",", title="Times played")])
        chart(c, 30 * len(t) + 40)
        sm = t.loc["Smash"]
        rush = t.loc["Rush"] if "Rush" in t.index else None
        style.card("Attack wins points; the net is where errors live",
                   f"<p>A <b>rush</b> at the net is a winner <b>{rush.Winner:.0f}%</b> of the time and a <b>smash</b> {sm.Winner:.0f}%. "
                   f"Net shots, returns at the net and lobs almost never win the point outright, and they end in an error far more "
                   f"often than in a winner.</p>" if rush is not None else f"<p>A smash is a winner {sm.Winner:.0f}% of the time.</p>", insight=True)
    with t3:
        pv = pd.DataFrame([(a, b, rally.point_value(0.5, 21, 30, a, b) * 100) for a in range(21) for b in range(21)], columns=["a", "b", "v"])
        c = alt.Chart(pv).mark_rect().encode(
            x=alt.X("b:O", title="Opponent's score", axis=alt.Axis(labelAngle=0)), y=alt.Y("a:O", title="Your score", sort="descending"),
            color=alt.Color("v:Q", scale=alt.Scale(scheme="goldorange"), legend=alt.Legend(title="Swing %")),
            tooltip=[alt.Tooltip("a:O", title="You"), alt.Tooltip("b:O", title="Opponent"), alt.Tooltip("v:Q", format=".0f", title="Game swing (%)")])
        chart(c, 420)
        style.card("Why 19-all feels different", f"""<p>Each square is how much one rally changes the chance of winning the game
        (two equal players). At <b>2-2 a rally is worth {rally.point_value(0.5, 21, 30, 2, 2):.0%}</b> of the game; from <b>19-all
        onwards every rally is worth {rally.point_value(0.5, 21, 30, 19, 19):.0%}</b>, almost four times as much. This is the
        Markov chain that powers the match replays.</p>""", insight=True)
    with t4:
        rl = d.assign(bucket=pd.cut(d.shots, [0, 3, 6, 10, 15, 20, 30, 100], labels=["1-3", "4-6", "7-10", "11-15", "16-20", "21-30", "31+"]))
        rl = rl.groupby(["event", "bucket"], observed=True).size().rename("n").reset_index()
        rl["share"] = rl.n / rl.groupby("event").n.transform("sum") * 100
        c = alt.Chart(rl).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=16).encode(
            x=alt.X("bucket:N", sort=None, title="Shots in the rally", axis=alt.Axis(labelAngle=0)), xOffset="event:N",
            y=alt.Y("share:Q", title="Share of rallies (%)"), color=alt.Color("event:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(title=None)),
            tooltip=["event:N", "bucket:N", alt.Tooltip("share:Q", format=".1f")])
        chart(c, 280)
        st.caption(f"Men's rallies average {d[d.event == MEN].shots.mean():.1f} shots, women's "
                   f"{d[d.event != MEN].shots.mean():.1f}. The longest rally in the data lasted {int(r.shots.max())} shots.")


# ------------------------------------------------------------------ players
def players():
    m, r, s = load()
    d = rally_view()
    plist = players_list()
    names = plist.index.tolist()
    who = st.selectbox("Player", names, index=names.index("Lakshya SEN") if "Lakshya SEN" in names else 0,
                       format_func=lambda n: f"{'🇮🇳 ' if n in bd.INDIA else ''}{n} ({plist[n]} matches)", key="bd_player")
    mm = m[(m.winner == who) | (m.loser == who)]
    style.hero(who, f"{'Women' if who in bd.WOMEN else 'Men'}'s singles · {len(mm)} matches in the data", eyebrow="🏸 Player profile")
    pr = d[(d.winner == who) | (d.loser == who)]
    won = pr.rally_winner == who
    lab = pr[pr.kind != "Not labelled"]
    my_winners = ((lab.rally_winner == who) & (lab.kind == "Winner")).sum()
    my_errors = ((lab.rally_loser == who) & (lab.kind == "Error")).sum()
    games = len(pr.groupby(["match_id", "game"]))
    style.kpis([("Matches", f"{len(mm)}", f"{(mm.winner == who).sum()} won · {(mm.loser == who).sum()} lost"),
                ("Rallies won", pct(won.mean() * 100, 1), f"of {len(pr):,}", "gold"),
                ("Winners a game", f"{my_winners / games:.1f}", "shots the opponent couldn't reach"),
                ("Errors a game", f"{my_errors / games:.1f}", "out, net or misjudged"),
                ("Rally length", f"{pr.shots.mean():.1f}", "shots on average")])
    mine = s[s.player == who].shot.value_counts(normalize=True) * 100
    peers = s[s.player.isin(bd.WOMEN) == (who in bd.WOMEN)].shot.value_counts(normalize=True) * 100
    mix = pd.DataFrame({who: mine, "Everyone": peers}).fillna(0)
    mix = mix[mix.max(axis=1) >= 3].drop(index="Unknown", errors="ignore")
    long = mix.reset_index().rename(columns={"index": "shot"}).melt("shot", var_name="Who", value_name="share")
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("##### Shot mix compared with everyone")
        chart(alt.Chart(long).mark_bar(cornerRadiusEnd=3, size=9).encode(
            y=alt.Y("shot:N", sort=mix.sort_values(who, ascending=False).index.tolist(), title=None), x=alt.X("share:Q", title="% of shots"),
            yOffset=alt.YOffset("Who:N", sort=[who, "Everyone"]), color=alt.Color("Who:N", sort=[who, "Everyone"], scale=alt.Scale(range=[GOLD, GREY]), legend=alt.Legend(title=None)),
            tooltip=["shot:N", "Who:N", alt.Tooltip("share:Q", format=".1f")]), 26 * len(mix) + 50)
    with b:
        st.markdown("##### How their rallies end")
        won_how = lab[lab.rally_winner == who].kind.map({"Winner": "Won · hit a winner", "Error": "Won · opponent erred"})
        lost_how = lab[lab.rally_loser == who].kind.map({"Winner": "Lost · opponent's winner", "Error": "Lost · own error"})
        h = pd.concat([won_how, lost_how]).value_counts().rename("n").reset_index()
        h.columns = ["how", "n"]
        h["share"] = h.n / h.n.sum() * 100
        order = ["Won · hit a winner", "Won · opponent erred", "Lost · opponent's winner", "Lost · own error"]
        chart(alt.Chart(h).mark_bar(cornerRadiusEnd=3, size=22).encode(
            y=alt.Y("how:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)), x=alt.X("share:Q", title="% of rallies"),
            color=alt.Color("how:N", sort=order, scale=alt.Scale(domain=order, range=[AQUA, "#7fd3b4", ORANGE, "#f0a487"]), legend=None),
            tooltip=["how:N", "n:Q", alt.Tooltip("share:Q", format=".1f")]), 230)
    st.markdown("##### Matches")
    show = mm.assign(result=np.where(mm.winner == who, "Won", "Lost"), opponent=np.where(mm.winner == who, mm.loser, mm.winner))
    st.dataframe(show[["date", "tournament", "round", "opponent", "result", "score_line"]].sort_values("date", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("Date", format="D MMM YYYY"), "tournament": "Tournament",
                                "round": "Round", "opponent": "Opponent", "result": "Result", "score_line": "Score (winner first)"})


# ------------------------------------------------------------------ match replay
def replay():
    m, r, s = load()
    (mdl, info) = model()
    style.hero("Match replay", "Pick a match and watch the chance of winning change after every single rally, from the "
               "Markov-chain model.", eyebrow="🏸 Rally-by-rally win probability")
    ms = m.sort_values(["india", "date"], ascending=[False, False])
    labels = [match_label(x) for x in ms.itertuples()]
    default = next((i for i, x in enumerate(ms.itertuples()) if x.winner == "Lakshya SEN" and x.round == "Final"), 0)
    pick = st.selectbox("Match (Indian players first)", labels, index=default, key="bd_rp")
    row = ms.iloc[labels.index(pick)]
    rp = bd.replay(0.5, r[r.match_id == row.match_id], mdl["strength"])
    who = bd.short(row.winner)
    rp["pct"] = rp.a_win * 100
    rp["score"] = "Game " + rp.game.astype(str) + ": " + rp.score_a.astype(str) + "-" + rp.score_b.astype(str) + \
        " (games " + rp.games_a.astype(str) + "-" + rp.games_b.astype(str) + ")"
    rp["next"] = np.where(rp.a_won == True, f"{who} won it", np.where(rp.a_won == False, f"{bd.short(row.loser)} won it", "")) + \
        np.where(rp.last_shot != "", " · " + rp.last_shot.astype(str) + " · " + rp.ending.astype(str) + " · " + rp.shots.astype(str) + " shots", "")
    rp["value"] = rp.point_value * 100
    style.card(f"{row.winner} beat {row.loser}", f"<p>{row.date:%d %B %Y} · {row.tournament} · {row['round']} · {row.event}</p>"
               f"<p>Score: <b style='color:#f2b630'>{row.score_line}</b> · {len(rp) - 1} rallies</p>")
    tips = [alt.Tooltip("score:N", title="Score before the rally"), alt.Tooltip("next:N", title="What happened"),
            alt.Tooltip("pct:Q", format=".0f", title=f"{who} win %"), alt.Tooltip("value:Q", format=".0f", title="Rally worth (% of game)")]
    bounds = rp.groupby("game").n.agg(["min", "max"]).reset_index()
    two_way_worm(rp, "n", "pct", "Rally number", f"{who}'s chance of winning the match (%)", tips, height=340)
    st.caption(" · ".join(f"Game {int(g)}: rallies {int(lo)}-{int(hi)}" for g, lo, hi in zip(bounds.game, bounds["min"], bounds["max"])))
    rp["change"] = rp.pct.diff().shift(-1).fillna(0)
    sw = rp.loc[rp.change.abs().sort_values(ascending=False).index].head(5)
    items = "".join(f"<li><b>{x.score}</b>: {x.next} → {who} {'up' if x.change > 0 else 'down'} <b>{abs(x.change):.0f} points</b></li>" for x in sw.itertuples())
    runs = (rp.a_won.dropna() != rp.a_won.dropna().shift()).cumsum()
    streak = rp.a_won.dropna().groupby(runs).agg(["first", "size"]).sort_values("size", ascending=False).iloc[0]
    style.card("Turning points", f"<ul style='margin:.2rem 0 0 1rem;padding:0'>{items}</ul>"
               f"<p style='margin-top:.5rem'>Longest run: <b>{int(streak['size'])} points in a row</b> by "
               f"{who if streak['first'] else bd.short(row.loser)}.</p>")


# ------------------------------------------------------------------ predictor
def predictor():
    (mdl, info) = model()
    style.hero("Win calculator", "Set the games and the score: the Markov chain gives the chance of winning the game and the "
               "match, and how much the next rally is worth.", eyebrow="🏸 What are the chances?")
    a, b, c, d = st.columns(4)
    ga = a.number_input("Your games", 0, 1, 0, key="bd_ga")
    gb = b.number_input("Opponent's games", 0, 1, 0, key="bd_gb")
    sa = c.number_input("Your score", 0, 29, 18, key="bd_sa")
    sb = d.number_input("Opponent's score", 0, 29, 19, key="bd_sb")
    edge = st.slider("How often you win a rally against this opponent (%)", 40, 60, 50, key="bd_edge",
                     help="50% = evenly matched. Top players win about 52-55% of rallies against the rest of the top 20.")
    p = edge / 100
    over = (sa >= 21 or sb >= 21) and abs(sa - sb) >= 2 or sa >= 30 or sb >= 30
    if over:
        st.info("That game is already over: set the score of the game being played.")
        return
    g_win = rally.game(int(sa), int(sb), p, 21, 30)
    m_win = rally.match(int(ga), int(gb), p, "badminton", int(sa), int(sb))
    v = rally.point_value(p, 21, 30, int(sa), int(sb))
    style.kpis([("Chance of winning this game", f"{g_win:.0%}", f"at {sa}-{sb}", "gold"), ("Chance of winning the match", f"{m_win:.0%}", f"games {ga}-{gb}"),
                ("The next rally is worth", f"{v:.0%}", "of the game")])
    prob_bar([("You", m_win, AQUA), ("Opponent", 1 - m_win, ORANGE)])
    rows = [{"edge": e, "match": rally.match(0, 0, e / 100, "badminton") * 100} for e in np.arange(40, 60.5, 0.5)]
    st.markdown("##### Small edges add up: rallies won vs matches won")
    chart(alt.Chart(pd.DataFrame(rows)).mark_line(color=GOLD, strokeWidth=2.5).encode(
        x=alt.X("edge:Q", title="Rallies won (%)", scale=alt.Scale(domain=[40, 60])), y=alt.Y("match:Q", title="Matches won (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[alt.Tooltip("edge:Q", format=".1f"), alt.Tooltip("match:Q", format=".0f")]), 260)
    st.caption(f"Winning 53% of rallies wins about {rally.match(0, 0, 0.53, 'badminton'):.0%} of matches; 55% wins about "
               f"{rally.match(0, 0, 0.55, 'badminton'):.0%}. ")
    res = info["in_match"]
    fin = res["Score + rallies won so far (final)"]
    st.caption(f"How good is it? Tested on the 50 matches from 2022 after learning from 2018-21: rally by rally it named the eventual "
               f"winner {fin['Accuracy']:.0%} of the time, and {res['after game 1']['Accuracy']:.0%} after the first game. Player "
               f"ratings from 2018-21 did not help (only 14 of the 50 matches had both players in the earlier data), so the model "
               f"uses the score and the rallies won so far.")


PAGES = {"Overview": overview, "Players": players, "Match replay": replay, "Predictor": predictor}
