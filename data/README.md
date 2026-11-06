# Data

**Source:** [IPL Complete Dataset (2008-2024)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) on Kaggle: every IPL match and every ball from 2008 to 2024.

| File | Rows | What's in it |
|---|---|---|
| `matches_2008-2024.csv` | 1,095 | One row per match: teams, toss, venue, result, player of the match |
| `deliveries_2008-2024.csv` | 260,920 | One row per ball: batter, bowler, runs, extras, wickets |

The raw files are **not stored in this repo** (the ball-by-ball file is 69 MB). To get them, run:

```bash
python scripts/get_data.py
```

That downloads both files into this folder from a public GitHub mirror of the Kaggle dataset. Alternatively, download them from Kaggle and put them here.

**You only need them to re-run the notebook.** The app uses the cleaned tables in `app/data/` (`matches.csv` and `balls.parquet`), which `app/ipl_data.py` builds from these raw files.
