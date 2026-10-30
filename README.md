# Tactical Lab — luck or structure?

From StatsBomb open event data, build an xG model, turn it into expected points,
and separate teams that lose points through bad luck from teams with structural problems. Main data: four league seasons
(2015/16); the World Cups 2018 and 2022 and the Euros 2020 and 2024 are included as a test of whether the models carry over
to international football.

Data: StatsBomb Open Data (https://github.com/statsbomb/open-data). Attribution required: "data source: StatsBomb".

## How to reproduce

Python 3.12. From an empty `data/` folder (about 4 GB of downloads, then the whole pipeline):

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run_all.py
```

`run_all.py --skip-download` reuses `data/raw`. Every step logs to `reports/logs/`. The data is fetched from a pinned commit of
statsbomb/open-data (`COMMIT` in `src/download.py`), package versions are pinned in `requirements.txt`, and all random draws use
seed 7.

## Another club, several clubs, other seasons

The single-club analyses (`analyze_structure.py`, `opponent_types.py`, `gk_split.py`) default to Toulouse. Pick another club
with `--team`; add `--league` only if the name exists in two leagues. Names are listed in `reports/xpoints_table.csv`; a typo
gets a "did you mean" hint.

```
.venv\Scripts\python.exe -I -X utf8 src\analyze_structure.py --team "Leicester City"
.venv\Scripts\python.exe -I -X utf8 src\opponent_types.py --team "Leicester City"
.venv\Scripts\python.exe -I -X utf8 src\gk_split.py --team "Leicester City"
.venv\Scripts\python.exe run_all.py --skip-download --team "Leicester City"      # all three at once
```

Outputs are named after the club (`reports/leicester_city_structure.png`, `..._conceded_by_class.csv`, `..._score_state.csv`,
`..._by_opponent_type.csv`, `..._opponent_types.png`, `..._goalkeepers.png`). Every club is compared with the other teams of its
own league. The tests on leads and on shot-stopping follow the direction of the result, so an over-performing club gets the
right-hand tail ("P(>= 23 of 29 ...)") instead of the left one.

Several clubs side by side (`src/compare_clubs.py`; also the last step of `run_all.py`):

```
.venv\Scripts\python.exe -I -X utf8 src\compare_clubs.py --clubs "Leicester City,Toulouse,Sevilla,Aston Villa"
.venv\Scripts\python.exe -I -X utf8 src\compare_clubs.py --top 3     # default: 3 largest shortfalls and 3 largest surpluses
```

It writes `reports/compare_clubs.csv` (one row per club: points vs xPoints with the 95% band, finishing, defence, shot-stopping,
lead conversion, style type), `reports/compare_clubs_classes.csv` (conceded xG by attack type per club) and
`reports/compare_clubs.png`. A club's numbers are the same as in its single-club run (Toulouse was checked). Clubs from
different leagues are each measured against their own league. Choosing clubs because their gap is large makes plain p-values
too optimistic: `p_if_picked_from_80_clubs` multiplies the two-sided p by 80, which is deliberately conservative.

National teams from the four tournaments work the same way (names such as "Germany" exist in several tournaments, so `--league`
is needed; for `compare_clubs.py` use a prefix, `world_cup_2022:Morocco`):

```
.venv\Scripts\python.exe -I -X utf8 src\analyze_structure.py --team Morocco --league world_cup_2022
.venv\Scripts\python.exe -I -X utf8 src\compare_clubs.py --clubs "world_cup_2022:Morocco,euro_2024:Spain,Leicester City"
.venv\Scripts\python.exe -I -X utf8 src\compare_clubs.py --pool world_cup_2022 --top 2    # extremes inside one tournament
```

Other seasons: `reports/open_data_survey.csv` lists all 80 competition-seasons of the pinned data (`src/survey_open_data.py`).
There is no other full men's domestic league season: only 2015/16 of the four leagues, the four tournaments added in "Phase 6"
below, and single-club extracts such as Barcelona 2004/05-2020/21. Women's leagues look large but the fields the pipeline relies
on (shot freeze frames, provider xG) were not checked there. To add competitions, extend `COMPETITIONS` in `download.py` and the
label lists in `build_*.py` and `xpoints.py`, and check the team-name aliases in `build_shots.py`.

Parameters worth changing (re-run the listed step and what follows it): `TRAIN_LAST_WEEK` and the held-out league in
`xg_model.py`; `N_SIMS`, `HALF` in `xpoints.py`; the 12-second fast-attack rule in `source_class` (`analyze_structure.py`);
`MIN_SHOTS` in `gk_split.py`; `FEATURES` and the candidate `k` values in `opponent_types.py`; `LONG_PASS`, `OPP_ZONE_X`,
`PRESS_ZONE_X` in `build_style.py`. The numbers quoted in the phase sections below are for the defaults.

## What the open data really contains

The first draft of the plan assumed Ligue 1 2021/22 + 2022/23 (with 360 data) were full seasons. They are not:

| Competition / season | Matches | Note |
|---|---|---|
| Ligue 1 2021/22, 2022/23 | 26 / 32 | Paris Saint-Germain matches only |
| Bundesliga 2023/24 | 34 | Bayer Leverkusen matches only |
| La Liga 2020/21 and earlier | ~30-38 per season | Barcelona matches only |
| **Premier League 2015/16** | **380** | full season |
| **La Liga 2015/16** | **380** | full season |
| **Serie A 2015/16** | **380** | full season |
| **Ligue 1 2015/16** | **377** | full season (3 matches missing) |

So the working dataset is the 2015/16 season of four leagues (1517 matches). No StatsBomb 360 for these,
but every shot event carries `freeze_frame` (positions of the goalkeeper and visible players at the moment of the shot)
and the provider's own `statsbomb_xg`, which we use as a benchmark.

Consequences recorded for the write-up:
- one season per league -> no season-to-season validation; validation is by match week (train weeks 1-26, test 27-38)
  and by league (train on three leagues, test on the fourth);
- the provider xG may have been fitted on these same shots, so it is a reference ceiling, not a fair opponent;
- conclusions hold for 2015/16 only.

## xG model

Data: 37,888 shots, 3,869 goals, 1,517 matches, 80 teams (after merging two Ligue 1 clubs that appear under two names:
Caen, Marseille). No duplicates, no missing provider xG; the 354 shots without a freeze frame are all penalties.

Features are limited to what is known at the moment of the shot (position, angle, body part, shot type, technique,
play pattern, pressure, key-pass type/height/length, goalkeeper position, defenders/attackers in the shot triangle).
Left out on purpose: outcome, `end_location`, `deflected`, provider xG.

Log loss on held-out shots (lower is better), open play + free kicks, penalties excluded:

| Model | time split (weeks 27-38) | cross-league (test Ligue 1) |
|---|---|---|
| base rate | 0.3248 | 0.3200 |
| logistic, distance only | 0.2900 | 0.2888 |
| logistic, distance + angle | 0.2884 | 0.2860 |
| logistic, all features | 0.2610 | 0.2581 |
| LightGBM, all features | 0.2599 | 0.2553 |
| StatsBomb xG (benchmark) | 0.2581 | 0.2567 |

Bootstrap by match (95% CI) for the difference in log loss, negative = first model better:
- LightGBM vs distance+angle: -0.0285 [-0.0329, -0.0240] and -0.0307 [-0.0359, -0.0253] — clearly better;
- LightGBM vs logistic (all features): -0.0011 [-0.0030, +0.0008] and -0.0028 [-0.0049, -0.0005] — practically a tie,
  the boosted trees add little over a regularised logistic model;
- LightGBM vs StatsBomb xG: +0.0018 [-0.0010, +0.0044] and -0.0014 [-0.0049, +0.0018] — not distinguishable from the provider.

Caveats to keep in the write-up:
- in weeks 27-38 every model, the provider's included, expects about 6% fewer goals than were scored (1,205-1,227 vs 1,278
  incl. penalties), so league-wide scoring was higher late in the season than early; points-vs-xPoints gaps must be judged
  against this offset, not against zero;
- the provider's model may have been fitted on these same shots; our result is "on par", not "better";
- the top calibration bin mixes penalties (~0.76) with big chances.

## xPoints

Method (`src/xpoints.py`):
- every shot gets an out-of-fold xG (5 folds grouped by match, LightGBM), then one logit shift per league so that league xG
  equals league goals (absorbs the league-wide offset noted above; the shift is tiny, rank agreement with the unshifted
  version is 1.000);
- per match and team, the goal distribution is the exact Poisson-binomial of its shots (no sampling), which gives win/draw/loss
  probabilities and xPoints;
- the season is simulated 10,000 times from those distributions to get a 95% band for "what shot-outcome luck alone produces".

Points minus xPoints (team-season, 80 teams); `reports/xpoints_table.csv`, `reports/xpoints_gap.png`:

| Largest shortfall | pts | xPts | gap |
|---|---|---|---|
| Toulouse (Ligue 1) | 40 | 60.0 | -20.0 |
| Hellas Verona | 28 | 42.1 | -14.1 |
| Torino | 45 | 58.3 | -13.3 |
| Aston Villa | 17 | 29.3 | -12.3 |

Largest surpluses: Atlético Madrid +15.8, Villarreal +14.8, Real Madrid +14.4, Bastia +13.9, Leicester City +13.5.

What holds up and what does not:
- Toulouse is the data-driven focal team (largest shortfall, ranked 3rd in Ligue 1 by xPoints, finished 17th). Both finishing
  (43 goals vs 52.7 xG) and defence (54 conceded vs 40.9 xGA) were worse than the chances imply, and the gap shows in both halves
  of the season, mostly in weeks 1-19.
- But with 80 teams someone is always extreme: chance alone produces a shortfall of -20 or worse somewhere in 7.8% of
  simulated seasons, so the most extreme team is NOT significant at 5% (the same test for the biggest surplus: p = 0.52).
- 9 of 80 teams fall outside their own 95% band (about 4 expected), so there is some real signal beyond luck or the model
  misses something that varies between teams.
- Persistence between weeks 1-19 and 20-38 (Pearson r over 80 teams, 95% bootstrap CI): points - xPoints +0.04 [-0.16, +0.25],
  finishing +0.04 [-0.24, +0.27], defence -0.00 [-0.20, +0.23]. No evidence that over/under-performance carries over, but the
  intervals are wide.
- The gap is related to team strength: r = +0.27 (p = 0.017) between xPoints and the gap, about +3 points per 20 xPoints. Strong
  teams tend to beat their xPoints, weak ones to fall short; xG from independent shots does not model game state (leading teams
  protect the lead). The ranking of extreme teams barely changes after removing this trend (Toulouse -21.0).
- Robustness: gap ranks from our xG and the provider's xG agree (Spearman 0.987).

Limits: own goals (28-38 per league) count in actual points but are not in shot xG; shots in a match are treated as
independent; the 95% band reflects shot-outcome randomness only, not model uncertainty; 3 Ligue 1 matches are missing from the
data (affects Bastia, Bordeaux, Gazelec Ajaccio, PSG, Saint-Etienne, Troyes, not Toulouse).

## Toulouse: structure of conceded chances

`src/build_context.py` re-reads the full events: possession origin, score before each shot, goal timeline (it reproduces the
official score in all 3,034 team-matches, own goals included). `src/analyze_structure.py` compares Toulouse with the 19 other
Ligue 1 teams (bootstrap over matches, 95% CI). Outputs: `reports/toulouse_*.csv`, `reports/shot_stopping_80_teams.csv`,
`reports/toulouse_structure.png` (the file names carry the club name, see "Another club" above).

1. **Chances conceded are not the problem.** Toulouse conceded 420 shots, 54 goals, xG 40.9. By how the attack arose (built-up,
   fast attack, counter, corner, free kick, throw-in, restart from keeper) none shows more xG conceded per match than the peers;
   only fast attacks are lower (-0.055 xG/match, CI excludes 0 — one of 9 comparisons, so treat as weak). Where the attack started
   (own third / middle / final third) shows no difference either.
2. **Points were lost from winning positions, not by conceding more.** Toulouse led at some point in 23 matches and won 9
   (39%); Ligue 1 peers won 72% of such matches. That is 36 points against about 54.5 expected, roughly 18 points, and the
   rank is 19th of 20 teams. Tail probability 0.0018 against the peer rate; the chance that some team of 20 looks this extreme
   by luck alone is 0.019. While leading, Toulouse conceded fewer chances than the peers (0.94 vs 1.22 xG per 90) but more
   goals (1.41 vs 1.06 per 90). Coming back from behind is normal (27% took a point vs 30%). This is the same effect as the
   xPoints gap seen from another side, not independent evidence.
3. **Where the goals above xG came from: shot-stopping.** Post-shot xG (PSxG, adds end position of on-target shots, log loss
   0.402 vs 0.494 for pre-shot xG) puts Toulouse's non-penalty goals against at 54 vs 40.9 expected for the on-target shots:
   -13.1 goals, 79th of 80 teams (bootstrap 95% CI [-22.0, -3.8]). Of the +14.8 goals above xG against, +13.1 is shot-stopping and
   only +1.7 is better placement of opponents' shots.
   But: with no real skill differences, the chance that at least one of 80 teams is this bad is 0.21; the observed spread of
   shot-stopping across teams is 5.2 goals against 4.6 from chance alone; and the effect does not persist between halves across
   teams (r = +0.08). The league's worst is AFC Bournemouth (-16.7, chance probability 0.02).
4. **Attack side.** Non-penalty 39 goals vs 47.3 xG. Of the -8.3 difference, -5.2 is one player (Martin Braithwaite Christensen: 105
   shots, 9 goals, 14.2 xG) and -2.3 another (Tongo Hamed Doumbia); the main striker Wissam Ben Yedder is level (15 vs 15.6).
   League-wide, finishing does not persist between halves (r = +0.04), so this is most likely noise.

What this means for the coaching-staff question "what to test":
- Nothing here supports changing the defensive structure: the chances conceded are normal or lower, in every phase of play and
  in every score state.
- Hypothesis to test first: shot-stopping. Review on-target goals conceded (especially while leading) and split by goalkeeper;
  one season cannot separate a real weakness from bad luck, so ask for more seasons before acting on it.
- Hypothesis for the attack: check the shot profile of the two under-performing shooters on video; one season of finishing is
  not evidence of a skill problem.
- The honest conclusion for the project is mostly a negative one on tactics: the lost points look like conversion luck plus a
  possible goalkeeping issue, and the data cannot prove either.

Caveats: this is one team, one season, 38 matches; the Toulouse focus was chosen because it had the biggest gap, so every
test here is partly post-hoc (the 20-team lead test and the 80-team shot-stopping test account for that only within their
group); PSxG knows nothing about goalkeeper positioning or the identity of the goalkeeper (not yet split); "fast attack" means a
regular-play possession of at most 12 seconds, a rule of thumb.

## opponent types and Toulouse's results against them

`src/build_style.py` (events -> possession share, long-pass share, cross share, pressures per opponent pass in their own 60%)
and `src/opponent_types.py` (adds share of fast attacks + counters, share of set-piece xG; k-means on 6 standardised
features over all 80 teams). Outputs: `reports/team_style_types.csv`, `reports/toulouse_by_opponent_type.csv`,
`reports/toulouse_opponent_types.png`.

- **The structure is weak.** Silhouette is 0.23 for three types (below 0.25, the usual line for "no substantial structure");
  teams form a continuum along possession/long-ball (the two features correlate at -0.85; PC1 = 49% of variance). The number of types
  was fixed by a rule set before looking at results: among k = 3, 4, 5 take the one with the highest bootstrap stability
  (adjusted Rand index) -> k = 3 (ARI 0.57, 10th percentile 0.39: moderate). k = 2 is more stable (ARI 0.87) but would
  hardly beat "possession vs direct". Treat the types as a convenient grouping, not as natural classes.
- **The three types** (profiles in standard deviations): 1 "Possession + pressing" (36 teams, 10 in Ligue 1; possession +0.8,
  long balls -0.8, pressing +0.6; average xPoints 58); 2 "Direct, fast attacks" (27 teams, 7 in Ligue 1, Toulouse among them; fast
  attacks +1.0; xPoints 49); 3 "Direct, set-piece reliant" (17 teams, only Angers, Bordeaux, Montpellier in Ligue 1; set-piece
  xG share +1.25; xPoints 44). The types overlap with strength, which is why strength tiers are shown as a comparison.
- **Toulouse against each type** (points minus xPoints per match; difference from the other 19 Ligue 1 teams facing the same type,
  95% CI): possession + pressing -0.37 [-0.86, +0.08] over 20 matches; direct, fast attacks -0.79 [-1.39, -0.15] over 12;
  direct, set-piece reliant -0.58 [-1.46, +0.17] over 6. The shortfall is negative against every type; it is nominally
  significant only against type 2, which is one of three post-hoc comparisons.
- **No evidence that the shortfall depends on the type.** Permutation test over Toulouse's 38 matches: p = 0.73. By opponent
  strength (thirds of Ligue 1 by xPoints) p = 0.46. Taken with Phase 3 this fits a team-wide effect (conversion of leads and
  shot-stopping) rather than one tactical mismatch.
- Sanity check: across all Ligue 1 teams the mean gap against each type is about zero (-0.05, +0.03, -0.01), so xPoints does not
  misjudge a particular type for everybody.

Caveats: type 3 has 6 Toulouse matches; styles are season averages that include the match against Toulouse; the style features
are my choice (six of many possible), the PPDA-style pressing measure counts StatsBomb pressure events; one season, one team.

## shot-stopping per goalkeeper

`src/build_gk.py` identifies the goalkeeper for every shot two independent ways (starting XI + substitutions + tactical shifts,
and the goalkeeper in the shot's freeze frame): they agree on all 37,502 shots where both are known (0 disagreements), and the
lineup method covers all 37,888. `src/gk_split.py` reuses the PSxG model from Phase 3. Outputs: `reports/gk_shot_stopping.csv`,
`reports/toulouse_goalkeepers.png`.

- **The -13.1 goals are spread over three goalkeepers, none of them individually unusual.** Non-penalty on-target shots faced:
  Alban Lafont 78 shots, 27 goals vs 22.2 PSxG (-4.9, z = -1.46); Ali Nadhoim Ahamada 31 shots, 15 vs 10.4 (-4.6, z = -2.05);
  Mauro Daniel Goicoechea Furia 30 shots, 12 vs 8.3 (-3.7, z = -1.86). Only Lafont (rank 4th worst of 93 regular goalkeepers
  with at least 50 shots) is a regular; the other two are far too short to judge.
- **Little real skill spread exists to find.** Across the 93 regulars the variation is Q = 102.7 on 92 degrees of freedom, i.e.
  almost exactly what shot luck alone gives; the estimated true spread is 1.2 goals per 100 on-target shots. Between halves of
  the season, goalkeepers' results do not persist (r = +0.10 over 72 goalkeepers, p = 0.39).
- **Shrinking for noise takes Lafont back to average:** raw -6.2 goals per 100 shots becomes -0.2 [-2.4, +2.1]. The league's
  worst raw rate (Artur Boruc, AFC Bournemouth, -10.5, z = -3.10) shrinks to -0.9.
- **Changing the goalkeeper did not remove the problem:** weeks 1-19 mostly Ahamada and Goicoechea (-8.3 over 60 shots),
  weeks 20-38 Lafont (-5.3 over 56 shots), the shortfall stays negative with every keeper in goal. That points away from one
  individual and towards something shared (team defending, what PSxG cannot see) or towards luck.

What this changes in the recommendation from Phase 3: a goalkeeper change is not supported by this data. If the staff want to
check it, the 54 on-target goals conceded are a manageable video review, ideally against the goalkeepers' positioning, which PSxG
does not know about.

Caveats: PSxG uses shot placement but not goalkeeper position, shot pace or deflections; "regular" (at least 50 on-target shots) is
an arbitrary cut; shrinkage assumes a normal spread of true skill; one season.

## World Cup 2018 and 2022, Euro 2020 and 2024

**Data.** 230 matches (64 + 64 + 51 + 51), 5,458 regulation-time shots, from the same pinned commit. `match_week` is the stage
(1-3 group matchdays, 4 round of 16, 5 quarter-finals, 6 semi-finals, 7 final). The shots table has a `kind` column
(`league` / `tournament`) and a `stage` column.

**How tournaments are handled.**
- Regulation time only (periods 1-2). Extra time and shoot-outs are left out; 20 team-matches had extra-time goals. The score
  timeline rebuilt from events equals the official score minus extra-time goals in all 3,494 team-matches (leagues included).
- "Points" are 90-minute points: a knockout match level after 90 minutes counts as a draw.
- **No model is fitted on tournament data.** Everything is fitted on the four league seasons and then applied to the
  tournaments, so the league results are unchanged by construction (checked: 15 of 15 league tables identical after the
  full re-run) and the tournaments serve as a test of whether the models carry over to international football.

**xG carries over.** Fitted on the four leagues only, LightGBM xG on the 5,458 tournament shots (no penalties, log loss):
0.2482 against 0.2474 for StatsBomb's xG (difference +0.0008, 95% CI [-0.0025, +0.0043]) and 0.2768 for distance + angle
(-0.0286 [-0.0348, -0.0232]). Per tournament: World Cup 2018 0.2357 vs 0.2391 (provider), World Cup 2022 0.2649 vs 0.2701,
Euro 2020 0.2605 vs 0.2510, Euro 2024 0.2343 vs 0.2295. Total xG is 4% above goals (511 vs 492 non-penalty), 12% above in
World Cup 2018, 14% in Euro 2024 and 6% below in World Cup 2022, so xG is calibrated per tournament with one logit shift, as for
the leagues. Post-shot xG also carries over (log loss 0.4424 vs 0.5060 for pre-shot xG on 1,688 on-target tournament shots).

**Points against xPoints, tournaments.** Each tournament is simulated on its own (10,000 runs), `reports/xpoints_table_tournaments.csv`
and `reports/xpoints_tournaments.png`.
- *Over all matches the table is biased:* a team plays knockout matches only if it keeps winning, so teams that go deeper look
  better than their chances. Across the 112 teams the number of matches played and points - xPoints correlate at rho = +0.40
  (p < 0.0001). The biggest surpluses are all deep runs (France 2018 +7.3, Belgium 2018 +6.9, Spain 2024 +6.8 over 7 matches). Those
  numbers are kept in `xpoints_table_tournaments_all_matches.csv` but should not be used for comparisons.
- *Group stage only (3 matches each, no selection) is the fair comparison.* Largest gaps: World Cup 2018 Australia -4.3, Croatia
  +4.6; World Cup 2022 Canada -4.9 (0 points against 4.9 expected), Australia +4.5, Morocco +4.0 (7 points against 3.0); Euro
  2020 Denmark -4.0, Belgium +3.8; Euro 2024 Croatia -3.7, Spain +3.8 (9 points, no goal conceded). Only one gap is
  extreme against chance within its tournament (Canada, P = 0.025 for the worst of 32 teams); with eight such tests (two
  directions in four tournaments) one result at that level is unremarkable. Nothing here supports "a team under- or
  over-performed beyond luck".

**What works for national teams and what does not.**
- `analyze_structure.py`, `gk_split.py` and `compare_clubs.py` accept a national team. Shot-stopping is ranked among the teams of
  that tournament (24-32 teams, 10-30 on-target shots each); a national-team goalkeeper is shrunk towards the league mean using the
  league-estimated spread, and the comparison with league regulars is skipped. Intervals from 3-7 matches are very wide.
- `opponent_types.py` is skipped for tournaments (styles were fitted on 80 club teams), the persistence between halves is not
  computed (nothing to split), and the lead-conversion tests rest on a handful of matches.
- Neutral venues: `is_home` in tournament data only marks the listed home team.

Caveats: 3-7 matches per team is a tiny sample; the 360 data of World Cup 2022 and the Euros is not used; own goals are counted
in results but not in xG, as for the leagues.

## Layout

- `src/download.py` — fetches match lists and event files into `data/raw/` (untrusted, read with `python -I`)
- `src/build_shots.py` — one row per shot -> `data/processed/shots.parquet`
- `src/check_shots.py`, `src/check_teams.py` — data-quality checks on the shot table
- `src/xpoints.py` — out-of-fold xG, exact match probabilities, season simulation, halves -> `reports/xpoints_*`
- `src/build_context.py` — possession origin, score state, goal timeline from the full events
- `src/focal.py` — which club the single-club scripts are about (`--team`, `--league`; default Toulouse)
- `src/analyze_structure.py` — one club vs its league peers: attack types, score state, lead conversion, PSxG shot-stopping
- `src/build_style.py`, `src/opponent_types.py` — team style features, k-means types, the club's results by opponent type
- `src/build_gk.py`, `src/gk_split.py` — goalkeeper identity per shot, PSxG per goalkeeper, noise-adjusted skill
- `src/compare_clubs.py` — several clubs side by side (table + figure)
- `src/survey_open_data.py` — which competition-seasons the open data contains -> `reports/open_data_survey.csv`
- `src/check_candidates.py`, `src/check_gap_vs_strength.py` — checks behind the Phase 2 notes
- `src/xg_model.py` — baselines vs our models vs provider xG -> `reports/xg_metrics.csv`, `reports/xg_calibration.png`
- `AI_USAGE.md` — AI usage statement required by the lab; `PRESENTATION.md` — talk outline

Run (Windows): `.venv\Scripts\python.exe -I -X utf8`
