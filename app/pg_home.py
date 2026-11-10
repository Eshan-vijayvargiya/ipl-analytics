"""Home page: the five sports at a glance, the model behind each, and the upload engine."""
import json
from pathlib import Path

import streamlit as st

import style

ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def infos():
    out = {}
    for k, f in (("cricket", "model_info.json"), ("football", "football_info.json"), ("basketball", "basketball_info.json"),
                 ("badminton", "badminton_info.json"), ("volleyball", "volleyball_info.json")):
        try:
            out[k] = json.loads((ROOT / "models" / f).read_text())
        except Exception:
            out[k] = {}
    return out


def render(go):
    i = infos()
    style.hero("Five sports. One engine. Any file.",
               "Ball-by-ball, point-by-point and play-by-play data from five sports, a win-probability model for each one "
               "(tested on seasons it never saw), and an upload page that recognises the sport in any file and runs the right "
               "analysis.", eyebrow="🏆 Sports Analytics · built by Eshan Vijayvargiya")
    try:
        crk = f"AUC {i['cricket']['test_on_unseen_2022_24']['Test AUC']:.2f} on 2022-24"
    except Exception:
        crk = "tested on 2022-24"
    fb = i["football"].get("results", {}).get("Poisson goal model (final)", {})
    bk = i["basketball"].get("by_stage", {})
    bd = i["badminton"].get("in_match", {})
    vb = i["volleyball"].get("results", {}).get("Rally Markov model (final)", {})
    cards = [
        ("Cricket", "🏏", "IPL 2008-2024 · 260,759 balls", "Toss winners win just 50.8%; chasing 180 got 9 points easier after the Impact Player rule.",
         f"Chase model: {crk}"),
        ("Badminton", "🏸", "94 BWF singles matches · 81,939 shots", "2 points in 3 are lost by an error, not won by a winner.",
         f"Markov chain: {bd.get('after game 1', {}).get('Accuracy', 0):.0%} right after game 1"),
        ("Basketball", "🏀", "NBA 2021-24 · 3,690 games, every play", "A three and a two are worth the same per shot (1.10 vs 1.09 points).",
         f"Random walk: {bk.get('Tip-off', {}).get('Accuracy', 0):.0%} at tip-off, {bk.get('Last 2 minutes', {}).get('Accuracy', 0):.0%} in the last 2 min"),
        ("Football", "⚽", "49,940 internationals · 1872-2026", "Home teams win 51%; the last 15 minutes are the busiest for goals.",
         f"Poisson goals: {fb.get('Accuracy', 0):.0%} right before kick-off (2022-26)"),
        ("Volleyball", "🏐", "Bundesliga 2013-2019 · 6,330 sets", "Real matches are more one-sided than independent rallies predict.",
         f"Rally Markov: {vb.get('Accuracy', 0):.0%} right before the first serve"),
    ]
    rows = [cards[:3], cards[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (name, icon, data, fact, model) in zip(cols, row):
            with col:
                style.html(f'<div class="sport"><div class="ico">{icon}</div><h4>{name}</h4><p>{data}</p>'
                           f'<p style="color:#d5dcea;margin-top:.45rem">{fact}</p><p class="stat" style="margin-top:.45rem">{model}</p></div>')
                if st.button(f"Open {name}", key=f"home_{name}", width="stretch"):
                    go(name)
                    st.rerun()
        if row is rows[1]:
            with cols[2]:
                style.html('<div class="sport" style="border-color:#f2b630"><div class="ico">📂</div><h4>Your file</h4>'
                           '<p>Upload any sports data: CSV, Excel, JSON, PDF, Word or ZIP. One sport or many, clean or messy.</p>'
                           '<p style="color:#d5dcea;margin-top:.45rem">It recognises the sport and the kind of table, cleans it, and '
                           'runs the matching model.</p><p class="stat" style="margin-top:.45rem">5 sample files to try</p></div>')
                if st.button("Analyse a file", key="home_upload", width="stretch", type="primary"):
                    go("Your file")
                    st.rerun()
    st.markdown("#### The maths behind each sport")
    style.html("""<table class="mtab" style="margin-bottom:1rem">
    <tr><th>Sport</th><th>How it's scored</th><th>Model</th><th>Why this model</th></tr>
    <tr><td>🏏 Cricket</td><td>Runs vs balls and wickets left</td><td>Logistic regression on spline features</td><td>Smooth, explainable curves for "need X off Y with Z wickets"</td></tr>
    <tr><td>🏀 Basketball</td><td>Hundreds of small scoring chances</td><td>Random walk (normal model) + Elo</td><td>The margin drifts like a random walk; uncertainty shrinks with √time (fitted exponent: 0.50)</td></tr>
    <tr><td>⚽ Football</td><td>Few, rare goals</td><td>Poisson goals + Elo</td><td>Rare events are Poisson; gives win, draw and loss, and every scoreline</td></tr>
    <tr><td>🏸 Badminton</td><td>Rally point to 21, best of 3</td><td>Markov chain</td><td>Only the score matters: the exact chance from any score, and how much each point is worth</td></tr>
    <tr><td>🏐 Volleyball</td><td>Rally point to 25 (15), best of 5</td><td>Markov chain + Elo</td><td>Same chain as badminton with volleyball rules; Elo gives each side's rally chance</td></tr>
    </table>""")
    st.caption("Every model was trained on earlier seasons and tested on later ones it never saw. Data: IPL (Kaggle), ShuttleSet "
               "(MIT), NBA play-by-play (Apache-2.0), international football results (CC0), volleyball Bundesliga (CC0).")
