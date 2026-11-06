"""Page design: a dark 'floodlit stadium' look, plus small HTML helpers for cards and headers."""
import streamlit as st

# Chart colours: validated dark-mode steps (colour-blind safe, 3:1 contrast on the page background)
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
GOLD, INK, MUTED, LINE, CARD = "#f2b630", "#e8edf5", "#9aa6bd", "#26334f", "#16213a"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, .stApp { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 15% -10%, #1d2b4d 0%, rgba(14,21,37,0) 60%), #0e1525; }
h1, h2, h3, .hero h1, .kpi .v { font-family: 'Outfit', 'Inter', 'Segoe UI', Arial, sans-serif !important; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #0b1120; border-right: 1px solid #1d2840; }
.block-container { padding-top: 3.4rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stBaseButton-primary"] { color: #0e1525 !important; font-weight: 600; }
[data-testid="stBaseButton-primary"] p { color: #0e1525 !important; font-weight: 600; }

.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 1.5rem 1.7rem 1.4rem; margin-bottom: 1.1rem;
  background: linear-gradient(120deg, #18264a 0%, #121b33 55%, #1b1838 100%); border: 1px solid #2a3a63; }
.hero:after { content: ""; position: absolute; right: -60px; top: -80px; width: 280px; height: 280px; border-radius: 50%;
  background: radial-gradient(circle, rgba(242,182,48,.28) 0%, rgba(242,182,48,0) 65%); }
.hero .eyebrow { color: #f2b630; font-weight: 600; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }
.hero h1 { color: #f4f6fb; font-size: 2.15rem; font-weight: 800; margin: .15rem 0 .25rem; line-height: 1.1; }
.hero p { color: #b7c1d6; margin: 0; font-size: 1rem; max-width: 760px; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: .8rem; margin: .2rem 0 1.1rem; }
.kpi { background: #16213a; border: 1px solid #26334f; border-radius: 16px; padding: .85rem 1rem; }
.kpi .l { color: #9aa6bd; font-size: .8rem; font-weight: 500; }
.kpi .v { color: #f4f6fb; font-size: 1.75rem; font-weight: 700; line-height: 1.15; margin-top: .15rem; }
.kpi .n { color: #8d99b1; font-size: .78rem; margin-top: .1rem; }
.kpi.gold .v { color: #f2b630; }

.card { background: #16213a; border: 1px solid #26334f; border-radius: 16px; padding: 1rem 1.15rem; margin-bottom: .9rem; }
.card h4 { font-family: 'Outfit', 'Segoe UI', Arial, sans-serif; color: #f4f6fb; margin: 0 0 .35rem; font-size: 1.08rem; }
.card p, .card li { color: #b7c1d6; font-size: .93rem; margin: .15rem 0; }
.insight { border-left: 4px solid #f2b630; }
.chip { display: inline-block; padding: .22rem .62rem; border-radius: 999px; font-size: .8rem; font-weight: 600;
  margin: 0 .3rem .35rem 0; background: #1f2c4a; color: #e8edf5; border: 1px solid #2c3b5f; }
.chip b { color: #f2b630; }
.champ { display: grid; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); gap: .45rem; margin-bottom: 1rem; }
.champ > div { background: #16213a; border: 1px solid #26334f; border-top: 3px solid var(--team, #f2b630); border-radius: 12px; padding: .45rem .5rem; text-align: center; }
.champ .y { color: #8d99b1; font-size: .75rem; } .champ .t { font-family: 'Outfit', 'Segoe UI', Arial, sans-serif; font-weight: 700; font-size: 1.05rem; color: #f4f6fb; }
.bigprob { font-family: 'Outfit', 'Segoe UI', Arial, sans-serif; font-size: 4.2rem; font-weight: 800; line-height: 1; }
.bar { height: 14px; border-radius: 99px; background: #26334f; overflow: hidden; margin: .5rem 0 .2rem; }
.bar span { display: block; height: 100%; border-radius: 99px; }
.brand { display: flex; gap: .65rem; align-items: center; margin: .2rem 0 1rem; }
.brand .name { font-family: 'Outfit', 'Segoe UI', Arial, sans-serif; font-weight: 800; font-size: 1.25rem; color: #f4f6fb; line-height: 1.1; }
.brand .sub { color: #8d99b1; font-size: .8rem; }
.foot { color: #6f7b93; font-size: .78rem; margin-top: 1.5rem; }
</style>
"""

BALL_SVG = """<svg width="38" height="38" viewBox="0 0 38 38"><circle cx="19" cy="19" r="17" fill="#c8312f"/>
<path d="M9 7c5 6 5 18 0 24M29 7c-5 6-5 18 0 24" stroke="#f6e7c8" stroke-width="1.6" fill="none" stroke-dasharray="2.2 2"/>
<circle cx="13" cy="12" r="4" fill="#fff" opacity=".18"/></svg>"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def html(text):
    st.markdown(" ".join(line.strip() for line in text.splitlines()), unsafe_allow_html=True)


def hero(title, subtitle, eyebrow="IPL Analytics · 2008-2024"):
    html(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>')


def kpis(items):
    """items: list of (label, value, note) or (label, value, note, 'gold')."""
    cells = "".join(f'<div class="kpi {i[3] if len(i) > 3 else ""}"><div class="l">{i[0]}</div>'
                    f'<div class="v">{i[1]}</div><div class="n">{i[2]}</div></div>' for i in items)
    html(f'<div class="kpis">{cells}</div>')


def card(title, body, insight=False):
    html(f'<div class="card {"insight" if insight else ""}"><h4>{title}</h4>{body}</div>')
