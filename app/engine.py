"""The 'any sports file' engine: read a file, find its tables, split them by sport, recognise the sport and the
kind of table, and map the columns to a standard layout so the right analysis can run.

Everything here is pure pandas (no Streamlit), so it can be tested on its own.

    tables = read_tables(name, data)            # [(label, DataFrame), ...]  every sheet / table in the file
    parts = split_by_sport(df)                  # [(sport or None, DataFrame), ...]  rows grouped by a 'sport' column
    guess = detect_sport(df)                    # Guess(sport, confidence, scores, evidence)
    plan = plan_table(df, sport)                # Plan(kind, roles {role: column}, notes)
"""
import io
import re
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPORTS = ["Cricket", "Badminton", "Basketball", "Football", "Volleyball"]
SUPPORTED = ["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "pdf", "docx", "zip"]
MAX_ROWS = 300_000


def key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def tokens(name):
    """'HomeTeamGoals' / 'home_team_goals' / 'Home Team (goals)' -> {'home', 'team', 'goals'}."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


# ================================================================== 1. reading any file into tables
def _header_row(rows, limit=30):
    """The row that looks most like column names: mostly short text, not numbers, followed by data."""
    best, best_score = 0, -1.0
    for i, row in enumerate(rows[:limit]):
        cells = [c for c in row if c is not None and str(c).strip() not in ("", "nan", "None")]
        if len(cells) < 2:
            continue
        texty = sum(bool(re.search(r"[A-Za-z]", str(c))) and len(str(c)) < 40 and not re.fullmatch(r"[\d\s.,:/-]+", str(c)) for c in cells)
        score = texty / max(len(row), 1) + 0.15 * min(len(cells), 8) / 8
        nxt = rows[i + 1] if i + 1 < len(rows) else []
        if not [c for c in nxt if str(c).strip() not in ("", "nan", "None")]:
            score -= 0.5
        if score > best_score + 0.05:
            best, best_score = i, score
        if texty >= 3 and texty / max(len(cells), 1) >= 0.8:
            return i
    return best


def _promote_header(raw):
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if raw.empty:
        return raw
    h = _header_row(raw.head(30).astype(str).values.tolist())
    out = raw.iloc[h + 1:].copy()
    out.columns = [str(c).strip() for c in raw.iloc[h]]
    return out.reset_index(drop=True)


def _delimiter(lines):
    best, best_score = ",", 0
    for d in [",", ";", "\t", "|"]:
        counts = [line.count(d) for line in lines if line.strip()]
        nonzero = [c for c in counts if c]
        if nonzero:
            mode = max(set(nonzero), key=nonzero.count)
            score = nonzero.count(mode) * (1 + min(mode, 50) / 100)
            if score > best_score:
                best, best_score = d, score
    return best


def _read_csv(data):
    import codecs
    import csv
    head = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(data[:65536])
    lines = head.lstrip("﻿").splitlines()[:30]
    sep = _delimiter(lines)
    rows = list(csv.reader(lines, delimiter=sep))
    skip = _header_row(rows)
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep, skiprows=skip, encoding=encoding, low_memory=False,
                               on_bad_lines="skip", nrows=MAX_ROWS, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue


def _flat(v):
    if isinstance(v, (list, dict, tuple, set)):
        import json
        try:
            return json.dumps(v, default=str)[:200]
        except Exception:
            return str(v)[:200]
    return v


def _tidy(df):
    df = df.map(_flat) if hasattr(df, "map") else df.applymap(_flat)
    out, seen = [], {}
    for i, c in enumerate(df.columns):
        c = str(c).strip()
        if c == "" or c.lower() in ("nan", "none") or c.startswith("Unnamed:"):
            c = f"Column {i + 1}"
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c} ({seen[c]})")
    df = df.copy()
    df.columns = out
    df = df.replace(r"^\s*$", pd.NA, regex=True).dropna(how="all")
    df = df.dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def _tables_from(name, data, depth=0):
    ext = Path(name).suffix.lower().lstrip(".")
    if ext in ("csv", "tsv", "txt"):
        return [(Path(name).name, _read_csv(data))]
    if ext in ("xlsx", "xlsm", "xls"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, nrows=MAX_ROWS, dtype=object)
        return [(f"{Path(name).name} › {sheet}", _promote_header(raw)) for sheet, raw in sheets.items() if raw.dropna(how="all").shape[0] > 1]
    if ext == "json":
        import json
        obj = json.loads(data.decode("utf-8-sig", errors="replace"))
        if isinstance(obj, dict):
            lists = {k: v for k, v in obj.items() if isinstance(v, list) and v and isinstance(v[0], dict)}
            if lists:
                return [(f"{Path(name).name} › {k}", pd.json_normalize(v)) for k, v in lists.items()]
            if all(isinstance(v, list) for v in obj.values()):
                return [(Path(name).name, pd.DataFrame(obj))]
            return [(Path(name).name, pd.json_normalize(obj))]
        return [(Path(name).name, pd.json_normalize(obj))]
    if ext == "pdf":
        import pdfplumber
        out = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:60]:
                tables = page.extract_tables() or page.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
                for t in tables:
                    df = pd.DataFrame(t)
                    if out and df.shape[1] == out[-1][1].shape[1]:       # a table continued from the last page
                        prev = out[-1][1]
                        first = [str(c).strip() for c in df.iloc[0]]
                        df = df.iloc[1:] if first == list(prev.columns) else df
                        df.columns = prev.columns
                        out[-1] = (out[-1][0], pd.concat([prev, df], ignore_index=True))
                    else:
                        out.append((f"{Path(name).name} › table {len(out) + 1}", _promote_header(df)))
        if not out:
            raise ValueError("No table found in this PDF. It needs a real table, not a scanned image.")
        return out
    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        if not doc.tables:
            raise ValueError("No table found in this Word file.")
        return [(f"{Path(name).name} › table {i + 1}", _promote_header(pd.DataFrame([[c.text for c in r.cells] for r in t.rows])))
                for i, t in enumerate(doc.tables)]
    if ext == "zip" and depth == 0:
        out = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist()[:40]:
                inner = info.filename
                if info.is_dir() or inner.startswith("__MACOSX") or Path(inner).suffix.lower().lstrip(".") not in SUPPORTED:
                    continue
                try:
                    out.extend(_tables_from(inner, z.read(info), depth + 1))
                except Exception:
                    continue
        if not out:
            raise ValueError("No readable tables inside this ZIP file.")
        return out
    raise ValueError(f"'.{ext}' files aren't supported. Use one of: {', '.join(SUPPORTED)}.")


def read_tables(name, data):
    """Every table in the file as (label, DataFrame of text), with blank rows/columns removed.
    Any problem becomes a ValueError with a plain-English reason."""
    if not data:
        raise ValueError("The file is empty.")
    try:
        tables = _tables_from(name, data)
    except ValueError:
        raise
    except Exception as err:
        raise ValueError("The file couldn't be read. It may be damaged, password-protected, or not really the type its "
                         "name says. Try saving it again as CSV or Excel.") from err
    out = []
    for label, df in tables:
        if df is None or df.shape[1] == 0:
            continue
        df = _tidy(df.astype(object))
        if len(df) and df.shape[1]:
            out.append((label, df))
    if not out:
        raise ValueError("No table with data was found in this file.")
    return out


# ================================================================== 2. which sport?
SPORT_WORDS = {
    "Cricket": {"cricket", "ipl", "t20", "odi", "bbl", "psl", "cpl", "ranji", "bcci", "icc", "test"},
    "Badminton": {"badminton", "shuttle", "shuttlecock", "bwf", "badmintion"},
    "Basketball": {"basketball", "nba", "wnba", "ncaab", "euroleague", "hoops", "bball"},
    "Football": {"football", "soccer", "fifa", "uefa", "epl", "isl", "laliga", "bundesliga1", "seriea", "ligue", "futbol", "fussball"},
    "Volleyball": {"volleyball", "volley", "fivb", "vnl", "beachvolleyball", "pvl"},
}
COLUMN_WORDS = {   # words in column names, with a weight
    "Cricket": {"over": 3, "overs": 3, "ball": 1, "balls": 2, "batter": 4, "batsman": 4, "batsmen": 4, "striker": 2, "nonstriker": 4,
                "bowler": 5, "bowling": 3, "batting": 3, "wicket": 5, "wickets": 5, "wkts": 5, "wkt": 5, "runs": 2, "innings": 3,
                "inning": 3, "extras": 4, "wides": 5, "wide": 3, "noballs": 5, "noball": 5, "byes": 4, "legbyes": 5, "dismissal": 4,
                "dismissed": 4, "toss": 3, "powerplay": 5, "sixes": 3, "fours": 3, "maiden": 5, "maidens": 5, "economy": 3, "century": 3,
                "fifties": 3, "duck": 2, "stumped": 5, "lbw": 5, "keeper": 1, "rr": 1, "sr": 1, "deliveries": 4, "delivery": 4},
    "Badminton": {"shuttle": 5, "rally": 3, "rallies": 3, "smash": 4, "smashes": 4, "drop": 2, "clear": 2, "lob": 3, "netshot": 4,
                  "net": 1, "serve": 1, "server": 1, "receiver": 2, "singles": 2, "doubles": 2, "mixed": 1, "ms": 1, "ws": 1, "md": 1,
                  "wd": 1, "xd": 2, "game1": 2, "game2": 2, "game3": 2, "roundscore": 4, "getpoint": 4, "landing": 3, "backhand": 2,
                  "forehand": 1, "shot": 1, "drive": 1},
    "Basketball": {"quarter": 4, "qtr": 4, "q1": 2, "q2": 2, "q3": 2, "q4": 2, "rebound": 5, "rebounds": 5, "reb": 5, "oreb": 5, "dreb": 5,
                   "assist": 1, "assists": 1, "ast": 3, "steal": 4, "steals": 4, "stl": 5, "blk": 4, "turnover": 4, "turnovers": 4,
                   "tov": 5, "fg": 5, "fga": 5, "fgm": 5, "fg3": 5, "fg3m": 5, "fg3a": 5, "3pt": 5, "3pm": 5, "3pa": 5, "ft": 1, "fta": 5,
                   "ftm": 5, "dunk": 5, "layup": 5, "possession": 1, "plusminus": 4, "pf": 1, "pts": 2, "ppg": 4, "rpg": 5, "apg": 5,
                   "period": 2, "eventmsgtype": 6, "pctimestring": 6, "scoremargin": 6, "tricode": 4, "conference": 1, "nba": 5},
    "Football": {"goal": 3, "goals": 3, "gf": 2, "ga": 2, "fthg": 6, "ftag": 6, "hthg": 6, "htag": 6, "ftr": 5, "htr": 4, "xg": 5,
                 "corner": 4, "corners": 4, "yellow": 4, "red": 2, "cards": 4, "card": 3, "offside": 5, "offsides": 5, "penalty": 2,
                 "penalties": 2, "pens": 3, "shootout": 4, "scorer": 4, "scorers": 4, "keeper": 2, "goalkeeper": 4, "cleansheet": 5,
                 "cleansheets": 5, "fouls": 2, "foul": 1, "possession": 1, "minute": 2, "matchday": 3, "fixture": 3, "draw": 2,
                 "draws": 2, "neutral": 2, "hs": 1, "hst": 4, "ast": 0, "hc": 3, "hy": 3, "hr": 2, "ar": 1, "ay": 3, "assist": 1},
    "Volleyball": {"set": 2, "sets": 3, "set1": 4, "set2": 4, "set3": 4, "set4": 5, "set5": 5, "ace": 4, "aces": 4, "kill": 4,
                   "kills": 4, "attack": 3, "attacks": 3, "dig": 5, "digs": 5, "block": 2, "blocks": 2, "reception": 5,
                   "receptions": 5, "rec": 2, "libero": 6, "setter": 5, "spiker": 6, "rotation": 4, "sideout": 6, "attackpct": 5,
                   "servicepoints": 5, "serv": 2, "serve": 1, "spike": 4, "spikes": 4, "tiebreak": 3},
}


@lru_cache(maxsize=1)
def known_names():
    """Team and player names from the five built-in datasets, plus some famous clubs. {key(name): sport}."""
    names = {}

    def add(values, sport):
        for v in values:
            k = key(v)
            if len(k) >= 3 and k not in names:
                names[k] = sport
    try:
        import ipl_data
        add(list(ipl_data.TEAM.keys()) + list(ipl_data.TEAM.values()) + list(ipl_data.SHORT.values()), "Cricket")
    except Exception:
        pass
    add(["India", "Australia", "England", "Pakistan", "South Africa", "New Zealand", "Sri Lanka", "West Indies", "Bangladesh",
         "Afghanistan"], "Cricket")   # (these are football countries too: country names alone are weak evidence)
    try:
        import basketball as bb
        add(list(bb.TEAM_NAMES.values()) + [n.split()[-1] for n in bb.TEAM_NAMES.values()], "Basketball")
        add([k for k in bb.TEAM_NAMES if len(k) == 3], "Basketball")
    except Exception:
        pass
    for f, sport, col in (("football/matches.parquet", "Football", ["home_team", "away_team"]),
                          ("volleyball/matches.parquet", "Volleyball", ["home", "away"]),
                          ("badminton/matches.parquet", "Badminton", ["winner", "loser"])):
        try:
            d = pd.read_parquet(HERE / "data" / f, columns=col)
            vals = pd.unique(d[col].to_numpy().ravel())
            if sport == "Football":
                vals = [v for v in vals if v not in ("India", "Australia", "England", "Pakistan", "South Africa", "New Zealand",
                                                      "Sri Lanka", "West Indies", "Bangladesh", "Afghanistan")]
            add(vals, sport)
        except Exception:
            pass
    add(["Arsenal", "Chelsea", "Liverpool", "Manchester United", "Man United", "Man Utd", "Manchester City", "Man City", "Tottenham",
         "Spurs", "Everton", "Newcastle", "Aston Villa", "West Ham", "Real Madrid", "Barcelona", "Atletico Madrid", "Bayern Munich",
         "Borussia Dortmund", "Juventus", "AC Milan", "Inter Milan", "Napoli", "PSG", "Paris Saint-Germain", "Ajax", "Benfica",
         "Porto", "Celtic", "Rangers", "Mohun Bagan", "East Bengal", "Bengaluru FC", "Kerala Blasters", "Mumbai City", "FC Goa",
         "Chennaiyin", "Hyderabad FC", "Odisha FC", "Jamshedpur FC", "NorthEast United", "Punjab FC", "Brighton", "Wolves",
         "Leicester", "Southampton", "Crystal Palace", "Fulham", "Brentford", "Nottingham Forest", "Bournemouth"], "Football")
    add(["Viktor Axelsen", "Kento Momota", "PV Sindhu", "P V Sindhu", "Saina Nehwal", "Lakshya Sen", "Kidambi Srikanth",
         "HS Prannoy", "Satwiksairaj Rankireddy", "Chirag Shetty", "Tai Tzu Ying", "Carolina Marin", "Chen Yufei", "An Se Young",
         "Akane Yamaguchi", "Lee Zii Jia", "Kunlavut Vitidsarn", "Shi Yuqi", "Anders Antonsen"], "Badminton")
    add(["Lakers", "Celtics", "Warriors", "Bulls", "Heat", "Knicks", "Nets", "Bucks", "Nuggets", "Suns", "Mavericks", "Clippers",
         "Grizzlies", "Pelicans", "Thunder", "Timberwolves", "Spurs Basketball", "LeBron James", "Stephen Curry", "Kevin Durant",
         "Luka Doncic", "Nikola Jokic", "Giannis Antetokounmpo", "Joel Embiid", "Jayson Tatum"], "Basketball")
    add(["Virat Kohli", "Rohit Sharma", "MS Dhoni", "Jasprit Bumrah", "Sachin Tendulkar", "KL Rahul", "Shubman Gill",
         "Hardik Pandya", "Ravindra Jadeja", "Rishabh Pant", "Suryakumar Yadav", "AB de Villiers", "Chris Gayle", "Rashid Khan"], "Cricket")
    return names


SCORE_PAIR = re.compile(r"(\d{1,3})\s*[-–:]\s*(\d{1,3})")
CRICKET_SCORE = re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,2}\b|\(\s*\d{1,2}(\.\d)?\s*(ov|overs)?\s*\)")


@dataclass
class Guess:
    sport: str | None
    confidence: float
    scores: dict
    evidence: list = field(default_factory=list)


def _value_sample(df, n=1500):
    return df.sample(min(n, len(df)), random_state=0) if len(df) > n else df


def detect_sport(df, hint=""):
    """Score each sport from column names, values (known names, keywords, score patterns) and the file name."""
    scores = {s: 0.0 for s in SPORTS}
    evidence = {s: [] for s in SPORTS}
    for col in df.columns:
        t = tokens(col) | {key(col)}
        for s in SPORTS:
            hit = [w for w in t if COLUMN_WORDS[s].get(w, 0) > 0]
            if hit:
                w = max(COLUMN_WORDS[s][h] for h in hit)
                scores[s] += w
                evidence[s].append(f"column '{col}'")
            if t & SPORT_WORDS[s]:
                scores[s] += 4
                evidence[s].append(f"column '{col}' names the sport")
    for s in SPORTS:
        if tokens(hint) & SPORT_WORDS[s]:
            scores[s] += 6
            evidence[s].append("the file or sheet name")
    sample = _value_sample(df)
    names = known_names()
    for col in sample.columns:
        v = sample[col].dropna().astype(str).str.strip()
        if v.empty or v.str.len().mean() > 60:
            continue
        uniq = v.unique()[:400]
        sport_hits = {}
        for u in uniq:
            k = key(u)
            sp = names.get(k)
            if sp:
                sport_hits[sp] = sport_hits.get(sp, 0) + 1
            for s in SPORTS:
                if k in SPORT_WORDS[s] or key(u).startswith(s.lower()):
                    sport_hits[s] = sport_hits.get(s, 0) + 3
        for s, n in sport_hits.items():
            share = n / max(len(uniq), 1)
            if n >= 2 or share >= 0.3:
                scores[s] += min(8.0, 2 + 10 * share)
                evidence[s].append(f"{n} known {s.lower()} name{'s' if n > 1 else ''} in '{col}'")
        text = " ".join(uniq[:200]).lower()
        for s, words in (("Cricket", ("wicket", "caught", "bowled", "lbw", "run out", "stumped", "six", "four")),
                         ("Basketball", ("dunk", "layup", "rebound", "3pt", "free throw", "jump shot")),
                         ("Football", ("goal", "penalty", "own goal", "yellow card", "corner", "offside")),
                         ("Badminton", ("smash", "net shot", "drop shot", "clear", "lob", "short serve")),
                         ("Volleyball", ("ace", "kill", "spike", "dig", "block", "libero"))):
            found = [w for w in words if re.search(rf"\b{re.escape(w)}\b", text)]
            if len(found) >= 2:
                scores[s] += 2 * len(found)
                evidence[s].append(f"words like '{found[0]}' in '{col}'")
        if CRICKET_SCORE.search(" ".join(uniq[:50])):
            scores["Cricket"] += 4
            evidence["Cricket"].append(f"cricket-style scores (180/5) in '{col}'")
        pairs = v.str.findall(SCORE_PAIR)
        multi = pairs.str.len()
        flat = [int(x) for p in pairs for pr in p for x in pr]
        if len(flat) >= 6 and (multi >= 1).mean() > 0.6:
            hi = np.percentile(flat, 95)
            several = (multi >= 2).mean() > 0.5
            if several and 19 <= hi <= 31 and max(flat) <= 30 and np.median(flat) <= 21:
                scores["Badminton"] += 6
                evidence["Badminton"].append(f"game scores up to 21 in '{col}'")
            elif several and 23 <= hi <= 40:
                scores["Volleyball"] += 6
                evidence["Volleyball"].append(f"set scores up to 25 in '{col}'")
            elif hi <= 8:
                scores["Football"] += 4
                evidence["Football"].append(f"low scores like '2-1' in '{col}'")
            elif 70 <= hi <= 170:
                scores["Basketball"] += 5
                evidence["Basketball"].append(f"scores around 100 in '{col}'")
    # numbers in score-like columns
    for col in df.columns:
        t = tokens(col)
        if not t & {"score", "scores", "points", "pts", "goals", "runs", "total", "sets"}:
            continue
        x = pd.to_numeric(sample[col], errors="coerce").dropna()
        if len(x) < 5:
            continue
        med, hi = x.median(), x.quantile(0.95)
        if t & {"runs"} or (110 <= med <= 260 and hi <= 320):
            scores["Cricket"] += 2
        if 80 <= med <= 140 and hi <= 180 and not t & {"runs"}:
            scores["Basketball"] += 3
            evidence["Basketball"].append(f"'{col}' looks like basketball scores (~{med:.0f})")
        if hi <= 8 and t & {"goals", "score", "scores"}:
            scores["Football"] += 2
        if hi <= 3 and t & {"sets"}:
            scores["Volleyball"] += 2
    total = sum(scores.values())
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    conf = 0.0 if total == 0 else min(0.99, (scores[best] / total) * min(1.0, scores[best] / 8) * (0.6 + 0.4 * min(1, margin / 6)))
    sport = best if scores[best] >= 4 and margin >= 1 else None
    return Guess(sport, round(conf, 2), scores, evidence.get(best, [])[:6] if sport else [])


SPORT_SYNONYMS = {key(w): s for s, ws in SPORT_WORDS.items() for w in ws} | {key(s): s for s in SPORTS} | {
    "shuttlebadminton": "Badminton", "soccer": "Football", "fooball": "Football", "footbal": "Football", "crickett": "Cricket",
    "basket": "Basketball", "basketbal": "Basketball", "vollyball": "Volleyball", "volleybal": "Volleyball", "volley": "Volleyball",
    "beachvolley": "Volleyball", "football(soccer)": "Football", "associationfootball": "Football", "t20cricket": "Cricket",
    "odicricket": "Cricket", "testcricket": "Cricket"}


def sport_of(value):
    k = key(value)
    if not k:
        return None
    if k in SPORT_SYNONYMS:
        return SPORT_SYNONYMS[k]
    for s in SPORTS:
        if s.lower()[:6] in k:
            return s
    return None


def sport_column(df):
    """A column that says which sport each row is (e.g. 'Sport' = Cricket / Football ...), or None."""
    best, best_share = None, 0.0
    for col in df.columns:
        v = df[col].dropna().astype(str)
        if v.empty or v.nunique() > 30:
            continue
        mapped = v.map(sport_of)
        share = mapped.notna().mean()
        named = bool(tokens(col) & {"sport", "sports", "game", "discipline", "category", "type", "event"})
        if mapped.nunique() >= 1 and (share >= 0.6 or (named and share >= 0.3)) and share > best_share:
            if mapped.nunique() >= 2 or named:
                best, best_share = col, share
    return best


def split_by_sport(df):
    """[(sport, rows)] using a sport column if there is one; otherwise [(None, df)]."""
    col = sport_column(df)
    if col is None:
        return [(None, df)], None
    sp = df[col].map(sport_of)
    parts = []
    for s in SPORTS:
        rows = df[sp == s].drop(columns=[col]).dropna(axis=1, how="all").reset_index(drop=True)
        if len(rows):
            parts.append((s, rows))
    other = df[sp.isna()].drop(columns=[col]).dropna(axis=1, how="all").reset_index(drop=True)
    if len(other):
        parts.append(("Unknown", other))
    return parts, col


# ================================================================== 3. what kind of table, and which column is which
G = lambda *w: set(w)        # noqa: E731  (a group of interchangeable words)
SIDE_A = G("home", "host", "hosts", "team1", "teama", "player1", "playera", "t1", "p1", "side1", "sidea", "first")
SIDE_B = G("away", "visitor", "visitors", "visiting", "guest", "opponent", "opposition", "opp", "team2", "teamb", "player2",
           "playerb", "t2", "p2", "side2", "sideb", "second", "vs", "against")
SCORE_W = G("score", "scores", "goals", "goal", "points", "pts", "runs", "sets", "games", "total", "gls", "pt", "fs", "ft")
ROLES = {
    "match_id": [(95, [G("matchid", "gameid", "fixtureid", "mid", "gid", "matchno", "matchnumber", "fixture")]),
                 (90, [G("match", "game", "fixture"), G("id", "no", "number", "code", "key", "num")]),
                 (65, [G("match", "game", "fixture", "matchup")])],
    "date": [(95, [G("date", "dates", "dt", "datetime", "kickoff", "gamedate", "matchdate", "startdate", "played", "day", "when")])],
    "season": [(90, [G("season", "seasons", "edition")]), (70, [G("year", "yr")])],
    "competition": [(90, [G("tournament", "competition", "league", "cup", "championship", "series", "comp", "division", "div", "event")])],
    "venue": [(80, [G("venue", "stadium", "ground", "arena", "city", "location", "place", "court")])],
    "neutral": [(95, [G("neutral")])],
    "side_a": [(97, [SIDE_A, G("team", "side", "club", "name", "player", "country", "nation")]), (92, [G("hometeam", "home", "team1", "teama", "player1", "playera", "host")]),
               (70, [G("team", "player", "side"), G("1", "a", "one")])],
    "side_b": [(97, [SIDE_B, G("team", "side", "club", "name", "player", "country", "nation")]),
               (92, [G("awayteam", "away", "visitor", "visitors", "opponent", "opposition", "opp", "team2", "teamb", "player2", "playerb", "guest")]),
               (70, [G("team", "player", "side"), G("2", "b", "two")])],
    "score_a": [(97, [SIDE_A, SCORE_W]), (96, [G("fthg", "hg", "homescore", "homegoals", "homepts", "hpts", "hscore", "score1", "goals1", "pts1", "runs1", "sets1", "scorea", "pointsa", "scorehome")]),
                (80, [SCORE_W, G("1", "a", "one", "for")])],
    "score_b": [(97, [SIDE_B, SCORE_W]), (96, [G("ftag", "ag", "awayscore", "awaygoals", "awaypts", "apts", "ascore", "score2", "goals2", "pts2", "runs2", "sets2", "scoreb", "pointsb", "scoreaway", "visitorscore")]),
                (80, [SCORE_W, G("2", "b", "two", "against")])],
    "score_text": [(90, [G("score", "result", "ft", "fulltime", "finalscore", "scoreline", "fts", "res", "final", "sets", "games", "setscores", "gamescores")])],
    "winner": [(92, [G("winner", "winners", "victor", "winning", "won", "matchwinner", "winningteam")])],
    "loser": [(90, [G("loser", "losers", "losing", "lost", "runnerup")])],
    "period": [(92, [G("quarter", "qtr", "period", "innings", "inning", "half", "set", "game", "inn")])],
    "minute": [(92, [G("minute", "min", "mins", "minutes", "time", "elapsed", "clock")])],
    "clock": [(95, [G("clock", "pctimestring", "timeleft", "timeremaining", "remaining", "gameclock")])],
    "over": [(95, [G("over", "overs", "ov", "overno", "overnumber")])],
    "ball": [(90, [G("ball", "balls", "delivery", "ballno", "ballnumber", "deliveryno")])],
    "batting_team": [(97, [G("batting", "bat", "battingteam", "inningsteam"), G("team", "side", "battingteam", "inningsteam")]), (90, [G("battingteam")])],
    "bowling_team": [(97, [G("bowling", "fielding", "field"), G("team", "side")]), (90, [G("bowlingteam", "fieldingteam")])],
    "batter": [(95, [G("batter", "batsman", "striker", "batsmen", "batterName", "batsmanname")])],
    "bowler": [(95, [G("bowler", "bowlername")])],
    "runs": [(92, [G("totalruns", "runstotal", "runsscored", "total")]), (90, [G("runs", "run", "runsoffbat", "batsmanruns", "batterruns", "runsbat")])],
    "extras": [(90, [G("extras", "extra", "extraruns", "extrasruns")])],
    "wicket": [(95, [G("wicket", "wickets", "dismissal", "dismissed", "iswicket", "playerdismissed", "wickettype", "wicketkind", "howout", "out")])],
    "team": [(85, [G("team", "club", "side", "country", "nation", "franchise", "tm", "squad", "teamname", "tricode")])],
    "player": [(92, [G("player", "playername", "athlete", "name", "fullname", "scorer", "goalscorer", "person")])],
    "point_winner": [(98, [G("pointwinner", "rallywinner", "getpoint", "getpointplayer", "pointto")]),
                     (97, [G("point", "rally", "points", "rallies"), G("winner", "won", "to", "by", "wonby")])],
    "games_played": [(90, [G("games", "matches", "gp", "played", "mp", "apps", "appearances", "inns", "g", "mat", "m")])],
}
NUMERIC_ROLES = {"score_a", "score_b", "period", "minute", "over", "ball", "runs", "extras", "games_played", "season"}
TEXT_ROLES = {"side_a", "side_b", "winner", "loser", "batting_team", "bowling_team", "batter", "bowler", "team", "player",
              "competition", "venue", "point_winner"}
ID_WORDS = G("id", "no", "number", "code", "key", "num")


def _rule_score(col, rules):
    t = tokens(col) | {key(col)}
    best = 0
    for score, groups in rules:
        if all(t & {g.lower() for g in grp} for grp in groups):
            best = max(best, score)
    return best


def num_share(s):
    x = pd.to_numeric(s.astype(str).str.replace(r"[,\s]", "", regex=True).str.extract(r"^(-?\d+(?:\.\d+)?)", expand=False), errors="coerce")
    v = s.notna()
    return float(x[v].notna().mean()) if v.any() else 0.0


def date_share(s):
    v = s.dropna().astype(str).head(300)
    if v.empty or v.str.fullmatch(r"\d{1,3}(\.\d+)?").mean() > 0.5:
        return 0.0
    d = pd.to_datetime(v, errors="coerce", dayfirst=True, format="mixed")
    return float(d.notna().mean())


@dataclass
class Plan:
    kind: str                      # 'events' | 'results' | 'players' | 'table'
    roles: dict                    # role -> column name
    label: str                     # plain-English name of the kind
    notes: list = field(default_factory=list)


KIND_LABEL = {"events": "play-by-play", "results": "match results", "players": "player or team stats", "table": "a table"}
EVENT_LABEL = {"Cricket": "ball-by-ball", "Basketball": "play-by-play", "Football": "goal timeline", "Badminton": "point-by-point",
               "Volleyball": "set scores"}


def find_roles(df):
    """Best column for each role (a column is used once). Scores = name rules + a check of the values."""
    cand = []
    for role, rules in ROLES.items():
        for col in df.columns:
            sc = _rule_score(col, rules)
            if not sc:
                continue
            s = df[col]
            t = tokens(col)
            if role in NUMERIC_ROLES and num_share(s) < 0.6:
                if not (role == "minute" and s.astype(str).str.contains(r"\d+\s*\+|\d+'", regex=True).mean() > 0.5):
                    continue
            if role in TEXT_ROLES and num_share(s) > 0.8:
                continue
            if role == "date" and date_share(s) < 0.6:
                continue
            if role == "period" and t & ID_WORDS:
                continue
            if role in ("side_a", "side_b", "player", "team") and t & (ID_WORDS | SCORE_W):
                continue
            if role == "score_text" and s.astype(str).str.contains(r"\d{1,3}\s*[-–:]\s*\d{1,3}", regex=True).mean() < 0.5:
                continue
            if role in ("winner", "loser") and t & G("point", "rally", "points", "rallies", "set", "game"):
                continue
            if role == "match_id" and sc < 90 and (s.nunique() > 0.9 * len(s) and num_share(s) < 0.5):
                continue
            if role == "minute" and t & G("minutes", "mins") and t & G("played", "mp", "per"):
                continue
            cand.append((sc, role, col))
    roles, used = {}, set()
    for sc, role, col in sorted(cand, key=lambda x: -x[0]):
        if role in roles or col in used:
            continue
        roles[role] = col
        used.add(col)
    return roles


def plan_table(df, sport):
    r = find_roles(df)
    notes = []
    if not ("score_a" in r and "score_b" in r):
        free = [c for c in df.columns if c not in r.values() and tokens(c) & SCORE_W and num_share(df[c]) >= 0.8]
        if len(free) >= 2 and "score_a" not in r and "score_b" not in r:
            r["score_a"], r["score_b"] = free[0], free[1]
            notes.append(f"Reading '{free[0]}' and '{free[1]}' as the two sides' scores.")
    n = len(df)
    per_match = None
    if "match_id" in r:
        per_match = n / max(df[r["match_id"]].nunique(), 1)
    has_sides = "side_a" in r and "side_b" in r
    has_scores = ("score_a" in r and "score_b" in r) or "score_text" in r or "winner" in r
    # play-by-play first (many rows per match)
    if sport == "Cricket" and ("over" in r or "ball" in r) and "runs" in r and ({"batter", "bowler", "batting_team"} & set(r)):
        return Plan("events", r, "ball-by-ball", notes)
    if sport == "Basketball" and "period" in r and ({"clock", "minute"} & set(r)) and ("score_a" in r and "score_b" in r or "score_text" in r):
        return Plan("events", r, "play-by-play", notes)
    if sport == "Football" and "minute" in r and ({"team", "player"} & set(r)) and (per_match or 0) >= 1 and \
            ("match_id" in r or "date" in r or has_sides):
        return Plan("events", r, "goal timeline", notes)
    if sport == "Badminton" and ("point_winner" in r or ("score_a" in r and "score_b" in r)) and "period" in r and \
            (per_match is None or per_match > 10):
        if not has_sides or (per_match or 0) > 10:
            return Plan("events", r, "point-by-point", notes)
    if sport == "Volleyball" and "period" in r and "score_a" in r and "score_b" in r and (per_match or 0) >= 3:
        return Plan("events", r, "set scores", notes)
    if has_sides and has_scores:
        return Plan("results", r, "match results", notes)
    if "winner" in r and "loser" in r:
        r["side_a"], r["side_b"] = r["winner"], r["loser"]
        return Plan("results", r, "match results", notes)
    nums = [c for c in df.columns if c not in r.values() and num_share(df[c]) >= 0.8]
    entity = r.get("player") or r.get("team")
    if entity is None:
        texty = [c for c in df.columns if num_share(df[c]) < 0.3 and df[c].nunique() > 0.5 * min(n, 500)]
        entity = texty[0] if texty else None
        if entity:
            r["player"] = entity
            notes.append(f"Using '{entity}' as the name column.")
    if entity and len(nums) >= 2:
        return Plan("players", r, "player or team stats", notes)
    return Plan("table", r, "a table", notes)


# ================================================================== 4. cleaning values
WORD_NUM = {"zero": 0, "nil": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "o": 0}
MISSING = {"", "na", "n/a", "nan", "none", "null", "-", "--", "—", "?", "nil.", "dnp", "tbd", "missing", "unknown", "#n/a"}


def to_number(s):
    """Numbers from messy text: '1,234', '21 pts', ' 7 ', 'two', '45+2' (-> 45), '78'' (-> 78), '12.5%'.
    Returns (numbers, status) where status is 'ok' | 'fixed' | 'blank' | 'unreadable' per cell."""
    raw = s.astype(object)
    txt = raw.astype(str).str.strip()
    blank = raw.isna() | txt.str.lower().isin(MISSING)
    direct = pd.to_numeric(txt.where(~blank), errors="coerce")
    out = direct.copy()
    need = out.isna() & ~blank
    if need.any():
        t = txt[need].str.lower().str.replace(",", "", regex=False)
        word = t.map(WORD_NUM)
        m = t.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
        out[need] = pd.to_numeric(word.fillna(m), errors="coerce")
    status = pd.Series("ok", index=s.index, dtype=object)
    status[blank] = "blank"
    status[need & out.notna()] = "fixed"
    status[need & out.isna()] = "unreadable"
    return out.astype(float), status


def to_date(s):
    v = s.astype(str).str.strip()
    a = pd.to_datetime(v, errors="coerce", dayfirst=True, format="mixed")
    b = pd.to_datetime(v, errors="coerce", dayfirst=False, format="mixed")
    iso = v.str.match(r"^\d{4}-\d{1,2}-\d{1,2}").mean() > 0.5
    d = b if iso or b.notna().sum() > a.notna().sum() else a
    status = pd.Series(np.where(s.isna(), "blank", np.where(d.isna(), "unreadable", "ok")), index=s.index)
    return d, status


def tidy_names(s):
    """Same name, different spelling of spaces/case ('INDIA', 'india ', 'India') -> the most common form."""
    v = s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    v = v.where(~v.str.lower().isin(MISSING))
    k = v.str.casefold()
    common = v.groupby(k).agg(lambda x: x.value_counts().index[0])
    out = k.map(common)
    changed = (out != s.astype(str)) & out.notna() & s.notna()
    return out, changed


@dataclass
class Clean:
    df: pd.DataFrame              # standardised columns named by role
    report: pd.DataFrame          # per role: column, blanks, fixed, unreadable
    dropped: int                  # rows removed (empty or duplicate)
    status: dict                  # role -> per-row status Series (for highlighting)


def clean(df, roles):
    out = pd.DataFrame(index=df.index)
    rows, status = [], {}
    for role, col in roles.items():
        s = df[col]
        if role in NUMERIC_ROLES and role != "season":
            if role == "minute":
                s = s.astype(str).str.extract(r"(\d+)", expand=False).where(s.notna())
            v, stt = to_number(s)
        elif role == "date":
            v, stt = to_date(s)
        elif role == "season":
            v = s.astype(str).str.strip().where(s.notna())
            stt = pd.Series(np.where(s.isna(), "blank", "ok"), index=s.index)
        elif role in TEXT_ROLES:
            v, changed = tidy_names(s)
            stt = pd.Series(np.where(s.isna() | v.isna(), "blank", np.where(changed, "fixed", "ok")), index=s.index)
        else:
            v = s
            stt = pd.Series(np.where(s.isna(), "blank", "ok"), index=s.index)
        out[role] = v
        status[role] = stt
        rows.append({"Role": role, "Column": col, "Blank": int((stt == "blank").sum()), "Fixed": int((stt == "fixed").sum()),
                     "Unreadable": int((stt == "unreadable").sum())})
    before = len(out)
    key_roles = [r for r in ("side_a", "side_b", "player", "team", "batter", "bowler", "match_id") if r in out]
    keep = out.notna().any(axis=1) if not key_roles else out[key_roles].notna().any(axis=1)
    dup = df.duplicated() & keep
    out = out[keep & ~dup]
    return Clean(out, pd.DataFrame(rows), before - len(out), {k: v[out.index] for k, v in status.items()})
