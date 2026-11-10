"""Shared page helpers: colours, number formats and the chart theme."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import style

BLUE, ORANGE, AQUA, GOLD, MUTED = style.BLUE, style.ORANGE, style.AQUA, style.GOLD, style.MUTED


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


def fmt_int(x):
    return "–" if x is None or pd.isna(x) else f"{int(round(x)):,}"


def flat(x):
    """A plain 1-D numpy array from a list, Series or array."""
    return np.asarray(x, dtype=float).ravel()


def two_way_worm(df, x, y, x_title, y_title, tips, marks=None, height=340, x_max=None):
    """Probability line with the area above 50% shaded aqua and below shaded orange (like a cricket 'worm').
    y is a 0-100 column. marks = optional DataFrame of events (same x, y columns) drawn as orange dots."""
    d = df.copy()
    d["up"], d["down"] = d[y].clip(lower=50), d[y].clip(upper=50)
    xs = alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[float(d[x].min()), float(x_max if x_max is not None else d[x].max())], nice=False))
    base = alt.Chart(d).encode(x=xs)
    up = base.mark_area(color=AQUA, opacity=0.25, interpolate="step-after").encode(y=alt.Y("up:Q", title=y_title, scale=alt.Scale(domain=[0, 100])), y2=alt.datum(50))
    down = base.mark_area(color=ORANGE, opacity=0.25, interpolate="step-after").encode(y="down:Q", y2=alt.datum(50))
    line = base.mark_line(color="#f4f6fb", strokeWidth=2, interpolate="step-after").encode(y=f"{y}:Q")
    hover = base.mark_point(opacity=0, size=220).encode(y=f"{y}:Q", tooltip=tips)
    mid = alt.Chart(pd.DataFrame({"m": [50]})).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="m:Q")
    layers = up + down + mid + line + hover
    if marks is not None and len(marks):
        layers = layers + alt.Chart(marks).mark_point(color=ORANGE, filled=True, size=90).encode(x=f"{x}:Q", y=f"{y}:Q", tooltip=tips)
    chart(layers, height)


def prob_bar(parts):
    """A horizontal 100% bar split into labelled parts: [(label, prob, colour), ...]."""
    cells = "".join(f'<div style="width:{p * 100:.2f}%;background:{c};" title="{l} {p:.0%}">'
                    f'{"" if p < 0.09 else f"{l} {p:.0%}"}</div>' for l, p, c in parts)
    style.html(f'<div class="pbar">{cells}</div>')


def big_number(label, value, sub="", color=None):
    style.html(f'<div class="card"><div style="color:#9aa6bd;font-size:.85rem">{label}</div>'
               f'<div class="bigprob" style="color:{color or "#f4f6fb"}">{value}</div>'
               f'<div style="color:#9aa6bd;font-size:.85rem;margin-top:.3rem">{sub}</div></div>')


def model_card(title, rows, columns, note=""):
    """A small table of test results: rows = {model name: {metric: value}}."""
    head = "".join(f"<th>{c}</th>" for c in ["Model"] + columns)
    body = ""
    for name, vals in rows.items():
        final = "final" in name.lower()
        tds = "".join(f"<td>{vals[c]:.3f}</td>" if isinstance(vals.get(c), float) and c not in ("Accuracy",)
                      else f"<td>{vals[c]:.0%}</td>" if c == "Accuracy" else f"<td>{vals.get(c, '')}</td>" for c in columns)
        body += f'<tr class="{"final" if final else ""}"><td>{name}</td>{tds}</tr>'
    style.card(title, f'<table class="mtab"><tr>{head}</tr>{body}</table><p style="margin-top:.5rem">{note}</p>')
