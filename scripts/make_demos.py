"""Sample files for the 'Your file' page, all cut from the real datasets and saved in everyday formats.

    python scripts/make_demos.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import ipl_data

OUT = APP / "demo"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

fm = pd.read_parquet(APP / "data/football/matches.parquet")
bg = pd.read_parquet(APP / "data/basketball/games.parquet")
vm = pd.read_parquet(APP / "data/volleyball/matches.parquet")
vs = pd.read_parquet(APP / "data/volleyball/sets.parquet")
bm = pd.read_parquet(APP / "data/badminton/matches.parquet")
br = pd.read_parquet(APP / "data/badminton/rallies.parquet")
cm, cb = ipl_data.load()

# 1. one results file, five sports, a 'Sport' column (the 'mixed file' case)
rows = []
for r in fm[fm.tournament == "FIFA World Cup"].tail(64).itertuples():
    rows.append(("Football", r.date, r.home_team, r.away_team, f"{r.home_score}-{r.away_score}", "FIFA World Cup 2026"))
for r in bg[bg.season == "2023-24"].tail(150).itertuples():
    rows.append(("Basketball", r.date, r.home, r.away, f"{r.home_score}-{r.away_score}", "NBA 2023-24"))
for r in vm[(vm.season == "2018/19") & (vm.gender == "Men")].itertuples():
    rows.append(("Volleyball", r.date, r.home, r.away, r.score_line, "Bundesliga men 2018/19"))
for r in bm.itertuples():
    rows.append(("Badminton", r.date, r.winner, r.loser, r.score_line, r.tournament))
tot = cb.groupby(["match_id", "inning"]).total_runs.sum().unstack()
wk = cb.groupby(["match_id", "inning"]).wicket.sum().unstack()
for r in cm[(cm.season == 2024) & ~cm.no_result].itertuples():
    rows.append(("Cricket", r.date, r.bat_first, r.bat_second,
                 f"{int(tot.loc[r.match_id, 1])}/{int(wk.loc[r.match_id, 1])} - {int(tot.loc[r.match_id, 2])}/{int(wk.loc[r.match_id, 2])}", "IPL 2024"))
mixed = pd.DataFrame(rows, columns=["Sport", "Date", "Team 1", "Team 2", "Score", "Competition"]).sample(frac=1, random_state=1)
mixed["Date"] = pd.to_datetime(mixed.Date).dt.strftime("%d/%m/%Y")
mixed.to_csv(OUT / "mixed_sports_results.csv", index=False)

# 2. a messy football file: title rows, mixed dates, '2 - 1' text scores, odd capitals, blanks, repeats, typos
ind = fm[(fm.home_team == "India") | (fm.away_team == "India")].tail(160).copy()
ind["Date"] = pd.to_datetime(ind.date).dt.strftime("%d-%m-%Y")
ind.loc[ind.sample(25, random_state=2).index, "Date"] = pd.to_datetime(ind.date).dt.strftime("%d %B %Y")
ind["Home"] = ind.home_team
ind["Away"] = ind.away_team
ind.loc[ind.sample(20, random_state=3).index, "Home"] = ind.home_team.str.upper()
ind.loc[ind.sample(15, random_state=4).index, "Away"] = " " + ind.away_team.str.lower() + " "
ind["Home Goals"] = ind.home_score.astype(str)
ind["Away Goals"] = ind.away_score.astype(str)
ind.loc[ind.sample(6, random_state=5).index, "Home Goals"] = ind.home_score.map({0: "nil", 1: "one", 2: "two", 3: "three"}).fillna("4")
ind.loc[ind.sample(4, random_state=6).index, "Away Goals"] = "N/A"
ind["Tournament"] = ind.tournament
messy = ind[["Date", "Home", "Away", "Home Goals", "Away Goals", "Tournament", "city"]].rename(columns={"city": "City"})
messy = pd.concat([messy, messy.sample(5, random_state=8)])
blank = pd.DataFrame([[None] * messy.shape[1]] * 3, columns=messy.columns)
messy = pd.concat([messy.iloc[:40], blank, messy.iloc[40:]])
buf = io.StringIO()
buf.write("India national football team - results export\nSource: international results database (messy copy, problems added on purpose)\n\n")
messy.to_csv(buf, index=False)
(OUT / "india_football_messy.csv").write_text(buf.getvalue())

# 3. IPL 2024 ball by ball, Cricsheet-style column names
b = cb[cb.season == 2024].drop(columns=["venue"], errors="ignore").merge(cm[["match_id", "date", "venue"]], on="match_id")
cs = pd.DataFrame({"match_id": b.match_id, "start_date": pd.to_datetime(b.date).dt.date, "venue": b.venue, "innings": b.inning,
                   "ball": (b.over - 1) + b.ball / 10, "batting_team": b.batting_team, "bowling_team": b.bowling_team,
                   "striker": b.batter, "bowler": b.bowler, "runs_off_bat": b.batter_runs, "extras": b.extra_runs,
                   "wicket_type": b.dismissal_kind.where(b.wicket), "player_dismissed": b.player_dismissed})
cs.to_csv(OUT / "ipl_2024_ball_by_ball.csv", index=False)

# 4. Excel workbook with three sports on three sheets
box = pd.read_parquet(APP / "data/basketball/box.parquet")
p = box[box.season == "2023-24"].groupby("player").agg(Team=("team", lambda x: x.mode().iat[0]), GP=("game_id", "nunique"),
                                                         PTS=("pts", "sum"), REB=("reb", "sum"), AST=("ast", "sum"),
                                                         STL=("stl", "sum"), BLK=("blk", "sum"), FGM=("fgm", "sum"), FGA=("fga", "sum"),
                                                         **{"3PM": ("fg3m", "sum")}).reset_index().rename(columns={"player": "Player"})
p = p[p.GP >= 20].sort_values("PTS", ascending=False)
vsets = vs.merge(vm[["uid", "date", "home", "away", "season", "gender", "stage"]], on="uid")
vsets = vsets[(vsets.season == "2018/19") & (vsets.stage.isin(["Final", "Semifinal"]))]
vsheet = pd.DataFrame({"Match": vsets.uid, "Date": vsets.date.dt.date, "League": vsets.gender, "Home team": vsets.home,
                       "Away team": vsets.away, "Set": vsets.set, "Home points": vsets.home_pts, "Away points": vsets.away_pts})
fsheet = fm[(fm.year >= 2024) & (fm.tournament.isin(["UEFA Euro", "Copa América", "FIFA World Cup"]))][
    ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]]
fsheet.columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Tournament", "Neutral"]
fsheet["Date"] = fsheet.Date.dt.date
with pd.ExcelWriter(OUT / "three_sports_workbook.xlsx") as xw:
    p.to_excel(xw, sheet_name="NBA players 2023-24", index=False)
    vsheet.to_excel(xw, sheet_name="Volleyball playoffs", index=False)
    fsheet.to_excel(xw, sheet_name="Football tournaments", index=False)

# 5. badminton point by point: the 2022 India Open final
mid = bm[(bm.winner == "Lakshya SEN") & (bm["round"] == "Final")].match_id.iat[0]
r = br[br.match_id == mid]
pts = pd.DataFrame({"Game": r.game, "Rally": r.rally, "Point won by": np.where(r.a_won, "Lakshya Sen", "Loh Kean Yew"),
                    "Sen score": r.score_a, "Loh score": r.score_b, "Shots in rally": r.shots, "Last shot": r.last_shot})
pts.to_csv(OUT / "badminton_sen_vs_loh_points.csv", index=False)
for f in sorted(OUT.iterdir()):
    print(f.name, f.stat().st_size // 1024, "KB")
