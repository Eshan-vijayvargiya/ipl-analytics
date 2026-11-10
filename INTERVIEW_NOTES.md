# Interview notes: Sports Analytics (5 sports, 1 engine, any file)

Live app: https://eshan-sports-analytics.streamlit.app/

## The 2-minute story

> "It started as an IPL project: every ball from 2008 to 2024. I cleaned it, tested cricket beliefs with statistics (the toss is a coin flip; the ground matters more), and built a ball-by-ball win-probability model that I tested on seasons it had never seen.
>
> Then I asked a bigger question: can one analytics engine understand any sport? I added badminton, basketball, football and volleyball, all real professional data: every shot of 94 BWF matches, every play of three NBA seasons, 150 years of international football, and six seasons of German volleyball.
>
> The interesting part is that each sport is scored differently, so each needs different maths. Cricket and basketball use logistic regression and a random-walk model. Football goals are rare events, so a Poisson model. Badminton and volleyball are rally-point sports, so I used a Markov chain: the chance of winning from any score can be worked out exactly. Every model was trained on earlier seasons and tested on later ones.
>
> Finally, I built an upload page. You give it any sports file, even a messy Excel with several sports in it. It finds the tables, splits them by sport, recognises the sport and the kind of table, cleans the data, and runs the right model, including Elo ratings trained on that file. I stress-tested it with 188 random and broken files: zero crashes."

**Why I made it:** badminton is my hobby, I follow cricket like everyone in India, and I wanted one project that shows the full data-science workflow (messy real data, statistics, several kinds of model, honest testing) and a product someone can actually use.

---

## The engine (most likely questions)

**How does it know which sport a file is?**
A points system with evidence, in `engine.py`. Three kinds of clue:
1. **Column names**: `over`, `bowler`, `wicket` point to cricket; `REB`, `AST`, `FGA` to basketball; `FTHG` to football; `libero`, `aces` to volleyball; `rally`, `smash` to badminton. Each word has a weight.
2. **Values**: known team and player names from the five datasets (Mumbai Indians, Lakers, Brazil, Berlin Volleys, Viktor Axelsen) and words like "wicket", "dunk", "smash".
3. **Score patterns**: `21-19, 15-21` is badminton, `25-23` volleyball, `180/5` cricket, `112-108` basketball, `2-1` football.
It shows the evidence and a confidence, and the user can override it.

**Why rule-based and not a machine-learning classifier?**
There's no labelled dataset of "files and their sports" to train on, and rules are explainable: the app can say *why* it thinks a file is cricket. If it were used widely, the user corrections could become training data for a classifier later.

**What if one file has several sports?**
If a column names the sport (even spelt "FOOTBALL", "soccer", "Basketbal"), rows are split by sport and each part is analysed separately. Rows from sports it doesn't know (kabaddi, hockey) are set aside rather than forced into the wrong model. Excel files: every sheet is a separate table.

**How does it know what each column means?**
Each role (home team, away score, over, batter, minute...) has naming rules, and every candidate is checked against its values. A "score" column must be mostly numbers, a "date" column must parse as dates, a team column must be text. Then it decides the kind of table: results (two sides and scores), play-by-play (many rows per match with time or over), or player stats (a name plus number columns).

**How does it clean messy data?**
"two" → 2, "45+2'" → 45, "1,234" → 1234, "INDIA" / " india " → India, mixed date formats (day-first vs ISO), N/A and blanks, repeated rows and title rows above the header. Every fixed or unreadable cell is counted and highlighted, so nothing is changed silently.

**What model runs on an uploaded file?**
For results: **Elo ratings learned from that file**, turned into win chances with the sport's model (for football, the Poisson model gives win/draw/loss). I test it **walk-forward**: each match is predicted using only the matches before it, and I compare against a simple rule (always pick the most common result). For play-by-play: the sport's win-probability replay. For player tables: leaderboards with rates like strike rate or points a game.

**How did you make sure it doesn't crash?**
A fuzz test: 188 generated files, including random tables in 8 formats, empty files, binary garbage named `.xlsx`/`.pdf`/`.zip`, a web page saved as `.csv`, 150-column tables, `1e308` and `inf` scores, misspelt sports, and the real samples with columns shuffled, renamed and 10% of cells corrupted. Zero crashes; unreadable files get a plain-English reason. In the app every section is also wrapped, so if one part can't handle a strange table it explains instead of breaking the page.

---

## The models, one by one

**Why a different model for each sport?** Because they're scored differently, and the model should match how the score is made:

| Sport | How it's scored | Model | Why |
|---|---|---|---|
| Cricket | runs vs balls and wickets | logistic regression + splines | smooth, explainable curves for "need X off Y" |
| Basketball | hundreds of small scoring chances | random walk (normal model) | the margin drifts like a random walk |
| Football | few, rare goals | Poisson | rare events; gives draws and every scoreline |
| Badminton / volleyball | rally point to 21 / 25 | Markov chain | only the score matters; exact answer from any score |

**What is a Markov chain, in simple words?**
A process where the next step depends only on where you are now, not on how you got there. In badminton, if I know the score is 18-19 and how often each player wins a rally, I can calculate exactly the chance of winning the game. Win the rally → 19-19; lose it → 18-20. Work backwards from the end of the game.

**What did the Markov chain teach you?**
- **Small edges add up**: winning 52% of rallies wins about 65% of badminton matches and 70% of volleyball matches.
- **Pressure is real**: from 19-all one rally is worth 50% of the game, almost 4 times a rally at 2-2.

**Its limit?** It assumes rallies are independent. In volleyball it predicted 25% of matches would end 3-0 to the home team, but 31% did: teams go on runs. That's an honest finding, not something I hid.

**What is a Poisson distribution?**
The distribution of how many times a rare event happens in a fixed time, like goals in 90 minutes. If a team averages 1.4 goals against this opponent, Poisson gives the chance of 0, 1, 2, 3 goals. Combining both teams gives every scoreline and the chance of a win, draw or loss.

**Why a random walk for basketball?**
A game has hundreds of possessions, each adding a few points to one side, so the final margin is roughly normal around the expected margin. The uncertainty shrinks with the square root of the time left. I didn't just assume that: I let the model fit the exponent and it came out at **0.504**. Textbook confirmed by data.

**What is Elo?**
A rating system from chess. Everyone starts at 1500; after each match the winner takes points from the loser, and more points for beating a stronger side. I used the World Football Elo rules (bigger tournaments and bigger wins count more) and FiveThirtyEight's NBA version (margin of victory, carry-over between seasons).

**How did you test the models?**
Always **out of time**: train on earlier seasons, test on later ones, because that's how a model is used in real life. For cricket I also grouped cross-validation by match, so balls from one match are never on both sides of a split.

**Which metrics, and why log loss?**
These are probability models, so what matters is that "70%" really means 70%. **Log loss** and **Brier** measure that; accuracy only checks the side of 50%. I also report accuracy because it's easy to explain.

**Tell me about a model that didn't work.**
Badminton player ratings. I expected that knowing the players (Axelsen vs a qualifier) would help. With only 44 earlier matches, and only 14 of the 50 test matches having both players in that history, the ratings made predictions *worse* (log loss 0.494 vs 0.456). So the final model uses the score and the rallies already played in the match. I kept the failed version in the README, because knowing when not to use a feature is part of the job.

---

## Numbers to remember

| | |
|---|---|
| Data | IPL 260,759 balls · BWF 81,939 shots · NBA 3,690 games · football 49,940 matches · volleyball 6,330 sets |
| Cricket | toss winner won 50.8% (p = 0.61); chase model AUC 0.878 on 2022-24 |
| Badminton | 68% of rallies end with an error; 76% right rally by rally; 88% after game 1 |
| Basketball | three vs two: 1.10 vs 1.09 points a shot; AUC 0.868; 94% in the last 2 minutes |
| Football | home teams win 51%; 60% right before kick-off; 88% at 85 minutes |
| Volleyball | home teams win 57%; 78% right before the first serve; AUC 0.88 |
| Engine | 188-file fuzz test, 0 crashes; 5 sample files incl. a mixed 5-sport file |

## IPL questions (still asked often)

**How do you know the toss doesn't matter?** Binomial test: if the toss made no difference, seeing 50.8% or more in 1,090 matches by chance is very likely (p = 0.61). But choosing to field wins 53.9% vs 45.4% (chi-square p = 0.009).

**What is the Impact Player rule and what did it change?** Since 2023 a team can bring in a substitute mid-match, usually an extra batter. Average first-innings scores went from 171 (2022) to 190 (2024), 200+ totals from 18 to 41, and chasing 180 went from 42% to 51%.

**Dot balls vs boundaries?** A bowler's dot-ball % barely relates to his economy (r = -0.11); boundary % relates very strongly (r = 0.84). Avoid boundaries, don't chase dots.

## What I'd add next

Who is playing (a set batter, a star on the bench), doubles badminton, rally-level volleyball data, and turning the user's corrections on the upload page into training data for a sport classifier.
