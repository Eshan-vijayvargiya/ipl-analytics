# Interview notes: IPL Analytics 2008-2024

## The 2-minute story

> "I wanted a project on data that's messy, large and that every interviewer in India understands, so I took every IPL ball from 2008 to 2024: about 1,100 matches and 260,000 deliveries.
>
> First I cleaned it. The same franchise appears under several names (Delhi Daredevils became Delhi Capitals), there were 58 spellings for 36 grounds, and super overs and rain-shortened games were mixed in with normal innings.
>
> Then I tested what people believe. Winning the toss turned out to be a coin flip: 50.8%, with a p-value of 0.61. Choosing to field after the toss does help (54% vs 45%), and the ground matters far more. Only 43% of chases succeed at Chepauk, against 65% in Jaipur. I also measured the Impact Player rule from 2023: 200-plus totals went from 18 in 2022 to 41 in 2024.
>
> The main piece is a ball-by-ball win-probability model. Before every ball of a chase it estimates the chasing team's chance of winning from runs needed, balls left, wickets left and the era. I trained it on 2008 to 2021 and tested it on 2022 to 2024, so it was judged on seasons it had never seen. Logistic regression with spline features did best (AUC 0.88), and I checked that its percentages are honest: when it says 60%, teams win about 60% of the time.
>
> Finally I built a five-page Streamlit app: team and player profiles, a replay of any match with the win-probability chart, and a predictor where you set up a chase and see your chances."

## Questions you'll likely get

**Why IPL and not a business dataset?**
It's real, large (260,000 rows) and messy, and it has the same problems as business data: entities that change names, duplicate spellings, special cases and a rule change that shifts the data. Franchises also really use this kind of analysis for toss decisions, auction buys and match strategy.

**How do you know the toss doesn't matter? 50.8% is above 50%.**
I ran a binomial test: if the toss made no difference, how often would we see 50.8% or more in 1,090 matches just by chance? Very often (p = 0.61), so there's no evidence the toss itself helps. But the *decision* after the toss does matter: fielding first wins 53.9% vs 45.4% for batting first, and a chi-square test gives p = 0.009.

**What is a p-value, in simple words?**
The chance of seeing a result at least this extreme if there were really no effect. A small p-value (below 0.05) means "this would be surprising if nothing were going on".

**Why does chasing win more often?**
The chasing team knows exactly what it needs, so it can pace the innings. In night games, dew also makes the ball wet and harder to grip for the side bowling second. But it depends on the ground: at Chepauk the pitch slows down and spinners grip it, so defending is easier.

**What is the Impact Player rule, and how did you measure it?**
Since 2023 each team can bring in a substitute mid-match, usually an extra batter. I compared 2023-24 with 2018-22: average first-innings scores, 200+ totals, sixes per match and run rate by phase (powerplay, middle, death) all jumped, and the powerplay jumped most (7.9 to 9.1 runs an over). Teams attack earlier because they have more batting in reserve.

**Tell me about the dot-ball finding.**
Everyone says "bowl dot balls". I tested it: a bowler's dot-ball percentage barely relates to his economy (r = -0.11), but his boundary percentage relates very strongly (r = 0.84). Spinners bowl fewer dots than fast bowlers, but give away a boundary only once every 8 to 9 balls, which is why Narine and Ashwin are among the most economical. The lesson: avoid boundaries, don't chase dots.

**How does the win-probability model work?**
Every legal ball of a chase is one row: runs needed, balls left, wickets left, the target and whether it's the Impact Player era. The label is whether the chasing team won. From these I also build the required run rate and runs needed per wicket. The model is logistic regression on spline features. It outputs a probability between 0 and 1.

**Why logistic regression and not gradient boosting or a neural network?**
I compared them. Gradient boosting scored AUC 0.874 and logistic regression with splines 0.878 on the unseen seasons, with better log loss too. When a simple model is at least as good, I prefer it: it's explainable, it's fast, and its probabilities are well calibrated. Splines give it the flexibility it needs.

**What are splines?**
They let a linear model bend. Instead of saying "each extra run of required rate lowers the chance by the same amount", the model can learn that going from 8 to 10 an over hurts a little and going from 12 to 14 hurts a lot.

**Why test on 2022-2024 instead of a random split?**
Two reasons. First, balls from the same match are almost identical, so a random split would put near-copies on both sides and make the score look better than it is. Second, in real use you predict the future from the past. Testing on later seasons, which include the new Impact Player rule, is the honest version. My cross-validation is also grouped by match for the same reason.

**What are log loss, Brier score and AUC?**
- **AUC** (0.878): if you take a random winning situation and a random losing one, how often does the model rank the winning one higher? 0.5 is guessing, 1.0 is perfect.
- **Brier score** (0.144): the average squared gap between the predicted probability and what happened (1 or 0). Lower is better.
- **Log loss** (0.444): like Brier, but it punishes confident wrong answers much harder. Lower is better.
For a probability model, log loss and Brier matter more than AUC, because I care that "70%" really means 70%.

**What is calibration and how did you check it?**
A model is calibrated if, among all the times it says 70%, the team wins about 70% of the time. I grouped predictions into bins and compared them with the real win rate (the calibration chart), and I checked specific situations: "need 60 off 36 with 6 wickets" – the model says 51%, and 50% of 118 real chases were won.

**What is data drift, and did you see it?**
Drift is when the world changes after the model learns. After the Impact Player rule, big chases became easier, so a model trained only on old seasons was too pessimistic. My fix was to add the era as an input and retrain on all 17 seasons. Now chasing 180 is 42% before 2023 and 51% after.

**Why exclude D/L and super-over matches from the model?**
In a rain-shortened match the target and overs change mid-game, so "runs needed off balls left" doesn't mean the same thing. Super overs are a separate mini-match. There are only 35 such games, so leaving them out is cleaner than modelling them badly.

**What would you add with more time?**
Who is batting (a set batter vs a new one), the venue's typical score, toss and dew for night games, and first-innings win probability too. I'd also retrain each season to keep up with the game.

**Can it be used for betting?**
I built it for understanding strategy, not betting. And the model only knows the score, not the players, so a bookmaker's odds would beat it.

**What did you find hardest?**
The cleaning. The raw files looked tidy but had padded text, renamed teams and grounds, and super overs stored as extra innings. If I hadn't fixed those, the Mohali ground would have been split into two "different" venues and Delhi's record would be divided between two names.

## Numbers to remember

| | |
|---|---|
| Data | 1,095 matches · 260,759 balls · 2008-2024 |
| Toss winner won | 50.8% (p = 0.61) |
| Field first vs bat first after toss | 53.9% vs 45.4% (p = 0.009) |
| Chasing teams won | 54% overall · 43% Chepauk · 65% Jaipur |
| Avg first-innings score | 171 (2022) → 190 (2024) |
| 200+ totals | 18 (2022) → 41 (2024) |
| Titles | CSK 5 · MI 5 · KKR 3 |
| Model (unseen 2022-24) | AUC 0.878 · log loss 0.444 · Brier 0.144 (baseline AUC 0.839) |
| Training data | 118,186 game states from 1,055 chases |
