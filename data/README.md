# Data

The app ships with **cleaned** tables in `app/data/` (about 10 MB), so it runs without downloading anything.
The **raw** files are only needed to rebuild those tables and retrain the models. They are not stored in this repo
(the raw files are several hundred MB). `python scripts/get_data.py` downloads them all into `data/raw/`.

| Sport | Source | What's in it | Licence |
|---|---|---|---|
| Cricket | [IPL Complete Dataset 2008-2024](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) (Kaggle; downloaded from a public GitHub mirror) | 1,095 matches, 260,920 deliveries | public dataset |
| Badminton | [ShuttleSet + ShuttleSet22](https://github.com/wywyWang/CoachAI-Projects) (Wang et al., KDD 2023) | 94 BWF World Tour singles matches, every stroke labelled from video: shot type, landing area, who won the rally and why | MIT |
| Basketball | [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data) (`cdnnba_2021-2023`, `nbastats_2021-2023`) | NBA regular seasons 2021-22 to 2023-24, every play, with shot locations | Apache-2.0 |
| Football | [martj42/international_results](https://github.com/martj42/international_results) | 49,940 men's internationals 1872-2026, goalscorers with minutes, penalty shootouts | CC0 |
| Volleyball | [volleystat](https://github.com/bozhinvi/volleystat) R package data | German Bundesliga men and women 2013/14-2018/19: matches, set scores, player match stats | CC0 |

## Rebuild everything

```bash
pip install -r requirements-notebook.txt
python scripts/get_data.py                      # raw files -> data/raw/
python scripts/build_football.py   data/raw/football
python scripts/build_basketball.py data/raw/nba
python scripts/build_badminton.py  data/raw/CoachAI-Projects
python scripts/build_volleyball.py data/raw/volleystat/data
python scripts/make_demos.py                    # the sample files on the upload page
```

Each build script cleans the raw data, trains the model on earlier seasons, tests it on later ones, prints the scores,
and saves the final model and its test results to `models/`. The cricket tables are built by `app/ipl_data.py`
(see the notebook).

The sample files in `app/demo/` are cut from these datasets. The "messy football file" has problems added **on purpose**
(title rows, mixed date formats, scores written as words, odd capitals, blanks and repeated rows) to show the cleaning.
