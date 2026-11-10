"""Sports Analytics: five sports, one engine. IPL cricket, badminton, basketball, football and volleyball,
plus an upload page that recognises the sport in any file and runs the right analysis.

Run from the project folder:  python -m streamlit run app/app.py
"""
import importlib

import streamlit as st

import badminton
import basketball
import football
import ipl_data
import rally
import stats
import style
import ui
import volleyball
import winprob

# After a git push, Streamlit Cloud re-runs this file but can keep OLD helper modules in memory.
# Reloading them (a few milliseconds) means an update never runs a mix of new and old code.
for _module in (style, ui, ipl_data, stats, winprob, rally, football, basketball, badminton, volleyball):
    importlib.reload(_module)

import pg_badminton  # noqa: E402  (page modules import the helpers above, so they load after the reload)
import pg_basketball  # noqa: E402
import pg_cricket  # noqa: E402
import pg_football  # noqa: E402
import pg_home  # noqa: E402
import pg_upload  # noqa: E402
import pg_volleyball  # noqa: E402

for _module in (pg_home, pg_cricket, pg_football, pg_basketball, pg_badminton, pg_volleyball, pg_upload):
    importlib.reload(_module)

st.set_page_config(page_title="Sports Analytics", page_icon="🏆", layout="wide", initial_sidebar_state="expanded")
style.setup()
state = st.session_state
state.setdefault("sport", "Home")
state.setdefault("page", "Overview")

SPORTS = {
    "Home": ("🏠", None),
    "Cricket": ("🏏", {"Overview": lambda lo, hi: pg_cricket.overview(lo, hi), "Teams": lambda lo, hi: pg_cricket.teams(lo, hi),
                      "Players": lambda lo, hi: pg_cricket.players(lo, hi), "Match replay": lambda lo, hi: pg_cricket.replay(),
                      "Predictor": lambda lo, hi: pg_cricket.predictor()}),
    "Badminton": ("🏸", pg_badminton.PAGES),
    "Basketball": ("🏀", pg_basketball.PAGES),
    "Football": ("⚽", pg_football.PAGES),
    "Volleyball": ("🏐", pg_volleyball.PAGES),
    "Your file": ("📂", None),
}
ICONS = {"Overview": ":material/insights:", "Teams": ":material/groups:", "Players": ":material/person:",
         "Match replay": ":material/replay:", "Predictor": ":material/online_prediction:"}


def go(sport, page="Overview"):
    state.sport, state.page = sport, page


def sidebar():
    lo, hi = 2008, 2024
    with st.sidebar:
        style.html(f'<div class="brand">{style.BALL_SVG}<div><div class="name">Sports Analytics</div>'
                   f'<div class="sub">5 sports · 1 engine · any file</div></div></div>')
        def sport_button(name, where):
            icon = SPORTS[name][0]
            if where.button(f"{icon}  {name}", key=f"sport_{name}", width="stretch",
                            type="primary" if state.sport == name else "secondary"):
                go(name)
                st.rerun()
        sport_button("Home", st)
        names = [n for n in SPORTS if n not in ("Home", "Your file")]
        for i in range(0, len(names), 2):
            cols = st.columns(2)
            for col, name in zip(cols, names[i:i + 2]):
                sport_button(name, col)
        sport_button("Your file", st)
        pages = SPORTS[state.sport][1]
        if pages:
            st.divider()
            st.caption(f"{SPORTS[state.sport][0]} {state.sport.upper()}")
            if state.page not in pages:
                state.page = "Overview"
            for page in pages:
                if st.button(page, icon=ICONS.get(page), key=f"page_{state.sport}_{page}", width="stretch",
                             type="primary" if state.page == page else "tertiary"):
                    state.page = page
                    st.rerun()
        if state.sport == "Cricket":
            st.divider()
            lo, hi = st.slider("Seasons", 2008, 2024, (2008, 2024), key="seasons",
                               help="Filters the Overview, Teams and Players pages.")
        style.html("""<div class="foot">Data: IPL (Kaggle) · ShuttleSet (BWF) · NBA play-by-play · international results
        1872-2026 · German Volleyball Bundesliga. All free to reuse.<br>Built by Eshan Vijayvargiya ·
        <a href="https://github.com/Eshan-vijayvargiya/ipl-analytics" style="color:#9aa6bd">code</a></div>""")
    return lo, hi


lo, hi = sidebar()
if state.sport == "Home":
    pg_home.render(go)
elif state.sport == "Your file":
    pg_upload.render()
elif state.sport == "Cricket":
    SPORTS["Cricket"][1][state.page](lo, hi)
else:
    SPORTS[state.sport][1][state.page]()
