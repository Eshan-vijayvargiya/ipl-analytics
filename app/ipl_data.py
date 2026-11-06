"""Clean the raw IPL files (2008-2024) into two tidy tables, and load them for the app and notebook.

Raw problems this fixes:
  * every text field in the ball-by-ball file is padded with spaces, and missing values are the text "NA"
  * 19 team names for 15 franchises (Delhi Daredevils -> Delhi Capitals, Kings XI Punjab -> Punjab Kings, ...)
  * 58 venue spellings for 36 grounds ("M.Chinnaswamy Stadium", "M Chinnaswamy Stadium, Bengaluru", ...),
    including renamed grounds (Feroz Shah Kotla -> Arun Jaitley Stadium, Motera -> Narendra Modi Stadium)
  * super overs stored as innings 3-6, rain-shortened (D/L) matches, ties and no-results
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

TEAM = {  # old name -> current franchise name
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
SHORT = {"Chennai Super Kings": "CSK", "Mumbai Indians": "MI", "Kolkata Knight Riders": "KKR",
         "Royal Challengers Bengaluru": "RCB", "Delhi Capitals": "DC", "Punjab Kings": "PBKS",
         "Rajasthan Royals": "RR", "Sunrisers Hyderabad": "SRH", "Gujarat Titans": "GT",
         "Lucknow Super Giants": "LSG", "Deccan Chargers": "DCG", "Pune Warriors": "PWI",
         "Rising Pune Supergiant": "RPS", "Gujarat Lions": "GL", "Kochi Tuskers Kerala": "KTK"}
ACTIVE = ["CSK", "MI", "KKR", "RCB", "DC", "PBKS", "RR", "SRH", "GT", "LSG"]
COLOR = {"CSK": "#F9CD05", "MI": "#004BA0", "KKR": "#3A225D", "RCB": "#D11D2B", "DC": "#17479E",
         "PBKS": "#DD1F2D", "RR": "#EA1A85", "SRH": "#F26522", "GT": "#1B2133", "LSG": "#A72056",
         "DCG": "#8A8D8F", "PWI": "#2F9BE3", "RPS": "#6F61AC", "GL": "#E04F16", "KTK": "#7E3F98"}

VENUE = {  # many spellings -> one name per ground
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
    "Punjab Cricket Association Stadium, Mohali": "PCA IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium": "PCA IS Bindra Stadium",
    "Punjab Cricket Association Stadium": "PCA IS Bindra Stadium",
    "Sheikh Zayed Stadium": "Zayed Cricket Stadium",
    "Subrata Roy Sahara Stadium": "MCA Stadium",            # Pune's ground, renamed in 2016
    "Maharashtra Cricket Association Stadium": "MCA Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "ACA-VDCA Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Ekana Cricket Stadium",
    "Himachal Pradesh Cricket Association Stadium": "HPCA Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium": "Maharaja Yadavindra Singh Stadium",
    "Vidarbha Cricket Association Stadium": "VCA Stadium",
}
CITY_SUFFIXES = (", Bengaluru", ", Mumbai", ", Kolkata", ", Chennai", ", Delhi", ", Hyderabad", ", Jaipur",
                 ", Pune", ", Abu Dhabi", ", Mohali, Chandigarh", ", Mohali", ", Dharamsala", ", Visakhapatnam",
                 ", Lucknow", ", Ahmedabad", ", Guwahati", ", Jamtha", ", Mullanpur", ", Uppal", ", Chepauk")
VENUE_CITY = {"M Chinnaswamy Stadium": "Bengaluru", "Eden Gardens": "Kolkata", "Wankhede Stadium": "Mumbai",
              "Arun Jaitley Stadium": "Delhi", "MA Chidambaram Stadium": "Chennai",
              "Rajiv Gandhi International Stadium": "Hyderabad", "Sawai Mansingh Stadium": "Jaipur",
              "PCA IS Bindra Stadium": "Mohali", "Narendra Modi Stadium": "Ahmedabad", "MCA Stadium": "Pune",
              "Dubai International Cricket Stadium": "Dubai", "Zayed Cricket Stadium": "Abu Dhabi",
              "Sharjah Cricket Stadium": "Sharjah", "Dr DY Patil Sports Academy": "Navi Mumbai",
              "Brabourne Stadium": "Mumbai", "Ekana Cricket Stadium": "Lucknow", "HPCA Stadium": "Dharamsala",
              "ACA-VDCA Stadium": "Visakhapatnam"}
PHASES = [("Powerplay", 1, 6), ("Middle", 7, 15), ("Death", 16, 20)]
BOWLER_WICKETS = {"caught", "bowled", "lbw", "caught and bowled", "stumped", "hit wicket"}


def team(name):
    return TEAM.get(name, name)


def venue(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    for suffix in CITY_SUFFIXES:            # "Eden Gardens, Kolkata" -> "Eden Gardens"
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return VENUE.get(name, VENUE.get(name.split(",")[0], name))


def _strip(df):
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().replace({"NA": pd.NA, "": pd.NA})
    return df


def clean_matches(raw):
    m = _strip(raw.copy()).rename(columns={"id": "match_id"})
    for col in ["team1", "team2", "toss_winner", "winner"]:
        m[col] = m[col].map(lambda x: team(x) if pd.notna(x) else x)
    m["venue"] = m["venue"].map(venue)
    m["city"] = m["city"].fillna(m["venue"].map(VENUE_CITY))
    m["date"] = pd.to_datetime(m["date"])
    m["season"] = m["season"].astype(int)
    m["dls"] = m["method"].eq("D/L").fillna(False)
    m["super_over"] = m["super_over"].eq("Y")
    m["no_result"] = m["result"].eq("no result")
    m["playoff"] = ~m["match_type"].eq("League")
    # who batted first: the toss winner if they chose to bat, otherwise the other team
    other = np.where(m["toss_winner"] == m["team1"], m["team2"], m["team1"])
    m["bat_first"] = np.where(m["toss_decision"] == "bat", m["toss_winner"], other)
    m["bat_second"] = np.where(m["bat_first"] == m["team1"], m["team2"], m["team1"])
    decided = ~m["no_result"] & m["winner"].notna()
    m["chasing_won"] = pd.Series(np.where(decided, m["winner"] == m["bat_second"], np.nan), index=m.index)
    m["toss_winner_won"] = pd.Series(np.where(decided, m["winner"] == m["toss_winner"], np.nan), index=m.index)
    m["impact_era"] = m["season"] >= 2023            # the Impact Player rule started in 2023
    for col in ["team1", "team2", "winner", "toss_winner", "bat_first", "bat_second"]:
        m[col + "_short"] = m[col].map(SHORT)
    keep = ["match_id", "season", "date", "city", "venue", "match_type", "playoff", "team1", "team2",
            "toss_winner", "toss_decision", "bat_first", "bat_second", "winner", "result", "result_margin",
            "target_runs", "target_overs", "dls", "super_over", "no_result", "chasing_won", "toss_winner_won",
            "player_of_match", "impact_era"] + [c for c in m.columns if c.endswith("_short")]
    return m[keep].sort_values("date").reset_index(drop=True)


def clean_balls(raw, matches):
    b = _strip(raw.copy())
    b = b[b["inning"].isin([1, 2])].copy()           # innings 3-6 are super overs: kept out of normal stats
    for col in ["batting_team", "bowling_team"]:
        b[col] = b[col].map(team)
    b["over"] = b["over"].astype(int) + 1             # the raw file counts overs from 0
    b["extras_type"] = b["extras_type"].fillna("none")
    b["legal"] = ~b["extras_type"].isin(["wides", "noballs"])
    b["faced"] = ~b["extras_type"].eq("wides")        # a no-ball counts as a ball faced, a wide doesn't
    b["bowler_runs"] = b["batsman_runs"] + np.where(b["extras_type"].isin(["wides", "noballs"]), b["extra_runs"], 0)
    b["wicket"] = b["is_wicket"].astype(bool)
    b["bowler_wicket"] = b["wicket"] & b["dismissal_kind"].isin(BOWLER_WICKETS)
    b["four"] = b["batsman_runs"].eq(4)
    b["six"] = b["batsman_runs"].eq(6)
    b["dot"] = b["legal"] & b["total_runs"].eq(0)
    b["phase"] = pd.cut(b["over"], [0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"])
    b = b.merge(matches[["match_id", "season", "venue", "impact_era"]], on="match_id", how="left")
    b = b.rename(columns={"batsman_runs": "batter_runs"})
    cols = ["match_id", "season", "venue", "impact_era", "inning", "batting_team", "bowling_team", "over", "ball",
            "phase", "batter", "bowler", "non_striker", "batter_runs", "extra_runs", "total_runs", "extras_type",
            "legal", "faced", "bowler_runs", "wicket", "bowler_wicket", "player_dismissed", "dismissal_kind",
            "fielder", "four", "six", "dot"]
    b = b[cols].reset_index(drop=True)
    for col in ["venue", "batting_team", "bowling_team", "batter", "bowler", "non_striker", "extras_type",
                "player_dismissed", "dismissal_kind", "fielder"]:
        b[col] = b[col].astype("category")
    return b


def build(raw_dir, out_dir=DATA):
    """Read the raw CSVs and write the clean tables the app and notebook use."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    matches = clean_matches(pd.read_csv(raw_dir / "matches_2008-2024.csv"))
    balls = clean_balls(pd.read_csv(raw_dir / "deliveries_2008-2024.csv", low_memory=False), matches)
    out_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out_dir / "matches.csv", index=False)
    balls.to_parquet(out_dir / "balls.parquet", index=False, compression="zstd")
    return matches, balls


def load(data_dir=DATA):
    matches = pd.read_csv(Path(data_dir) / "matches.csv", parse_dates=["date"])
    balls = pd.read_parquet(Path(data_dir) / "balls.parquet")
    return matches, balls
