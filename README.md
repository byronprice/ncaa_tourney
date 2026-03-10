# NCAA Tournament Predictor (MVP)

This project builds practical bracket predictions using:

1. Team power rankings (ESPN BPI and/or KenPom-style inputs)
2. Official NCAA tournament bracket data when available
3. Monte Carlo simulations of all tournament games
4. Exported, fillable bracket pick recommendations

## Quick Start

For Selection Sunday copy/paste commands, see `docs/bracket_runbook.md`.

### 1) Create environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare rankings and tempo input

Because source websites can change markup and access rules, this repo supports both:

- Automatic pull from public ESPN BPI table parser
- Optional tempo pull from KenPom public table parser (`AdjT`)
- Manual CSV import fallback (recommended reliability)

Generate a CSV template:

```bash
python -m ncaa_tourney.cli init-rankings-template --out data/raw/rankings_manual.csv
```

Fill `Team,Rating,Source` rows in that file.

Generate a tempo CSV template (optional but recommended):

```bash
python -m ncaa_tourney.cli init-tempo-template --out data/raw/tempo_manual.csv
```

Fill `Team,AdjT` rows in that file.

### 3) Prepare bracket input

If the official bracket endpoint/page changes, use manual fallback:

```bash
python -m ncaa_tourney.cli init-bracket-template --out data/raw/bracket_manual.csv
```

Expected format is documented in `docs/bracket_format.md`.

### 4) Build unified dataset and run simulations

```bash
python -m ncaa_tourney.cli build-dataset \
  --rankings-source manual_csv \
  --rankings-path data/raw/rankings_manual.csv \
  --tempo-source manual_csv \
  --tempo-path data/raw/tempo_manual.csv \
  --team-match-threshold 0.86 \
  --default-tempo 68.0 \
  --bracket-source manual_csv \
  --bracket-path data/raw/bracket_manual.csv \
  --out-teams data/processed/teams.csv \
  --out-games data/processed/round1_games.csv

python -m ncaa_tourney.cli simulate \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --n-sims 20000 \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out-summary output/simulation_summary.csv \
  --out-brackets output/top_brackets.csv

python -m ncaa_tourney.cli make-picks \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out output/strategy_picks.csv

python -m ncaa_tourney.cli optimize-picks \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --pool-size 50 \
  --n-candidates 300 \
  --n-outcomes 2000 \
  --round-points 1,2,4,8,16,32 \
  --candidate-mix 0.34,0.33,0.33 \
  --opponent-mix 0.5,0.35,0.15 \
  --opponent-safe-seed-chalk-share 0.25 \
  --opponent-seed-popularity data/raw/espn_pick_popularity.csv \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out output/optimized_picks.csv \
  --out-summary output/optimized_picks_summary.csv

python -m ncaa_tourney.cli check-sources \
  --rankings-source espn_bpi \
  --tempo-source kenpom_html \
  --tempo-path data/raw/kenpom_page.html \
  --team-match-threshold 0.86 \
  --default-tempo 68.0 \
  --out-report output/source_link_report.csv \
  --out-unmatched output/source_link_report_unmatched.csv \
  --out-alias-suggestions output/source_link_report_alias_suggestions.csv
```

You can also try direct site pulls:

```bash
python -m ncaa_tourney.cli build-dataset \
  --rankings-source espn_bpi \
  --tempo-source kenpom_public \
  --tempo-url https://kenpom.com/ \
  --bracket-source manual_csv \
  --bracket-path data/raw/bracket_manual.csv \
  --out-teams data/processed/teams.csv \
  --out-games data/processed/round1_games.csv
```

If KenPom returns `403` in automated requests, use one of these fallback options:

1) Export an authenticated browser copy of the KenPom page and parse local HTML:

```bash
python -m ncaa_tourney.cli build-dataset \
  --rankings-source espn_bpi \
  --tempo-source kenpom_html \
  --tempo-path data/raw/kenpom_page.html \
  --bracket-source manual_csv \
  --bracket-path data/raw/bracket_manual.csv \
  --out-teams data/processed/teams.csv \
  --out-games data/processed/round1_games.csv
```

2) Or set a session cookie for direct pull:

```bash
export KENPOM_COOKIE='your_cookie_header_here'
```

## Outputs

- `output/simulation_summary.csv`: per-team advancement probabilities by round
- `output/top_brackets.csv`: likely full bracket paths and estimated likelihood
- `output/strategy_picks.csv`: three pick sheets (`safe`, `balanced`, `upset_heavy`) with game-by-game winners
- `output/optimized_picks.csv`: a single pool-optimized bracket (max first-place equity vs modeled field)
- `output/optimized_picks_summary.csv`: top candidate brackets ranked by first-place equity

## Strategy profiles

- `safe`: uses pure model probabilities (no strategy adjustment)
- `balanced`: pulls favorites partway toward 50/50 to increase variance
- `upset_heavy`: pulls favorites more strongly toward 50/50, creating the highest variance
- `optimize-picks` also supports `--opponent-safe-seed-chalk-share` so part of the simulated `safe` field can use seed-based chalk behavior.
- If you provide `--opponent-seed-popularity`, those seed-pair upset rates are taken from your file (instead of built-in priors) for rounds listed in the file.

### ESPN pick popularity file format

Use a CSV with columns:

- `Round` (e.g., `R64`, `R32`, `S16`)
- `SeedFavorite` (lower seed number)
- `SeedUnderdog` (higher seed number)
- `UnderdogPickRate` (either 0-1 or 0-100)

Example rows:

```csv
Round,SeedFavorite,SeedUnderdog,UnderdogPickRate
R64,5,12,35
R64,3,14,15
R32,2,10,33
S16,1,4,31
```

## Probability model

- The simulator converts rating edge to expected spread and pace-adjusts by possessions.
- Team tempo (`AdjT`) is used to estimate game possessions.
- Win probability uses the inverted spread model: `spread = a + b * z(fav)`, with CLI controls `--spread-a` and `--spread-b`.

## Source quality check

- Use `check-sources` to validate both data pulls and generate a per-team link report.
- The report includes `TempoMatchType` (`exact`, `subset`, `fuzzy`, `default`) and `TempoMatchScore` for quick audit.
- `--out-unmatched` writes only rows with `TempoMatchType=default`.
- `--out-alias-suggestions` writes best-match suggestions to speed manual alias updates.

## Notes on Sources

- ESPN BPI parser is included as best-effort and may need selector updates year-to-year.
- KenPom data is intentionally handled as a manual import workflow by default to avoid brittle scraping and access issues.

## Full model run with optimizer
conda run -n ncaab python -m ncaa_tourney.cli optimize-picks --teams data/processed/top64_teams.csv --games data/processed/top64_round1_games.csv --pool-size 50 --n-candidates 1000 --n-outcomes 500 --seed 42 --spread-a 0.0 --spread-b 12.1 --opponent-safe-seed-chalk-share 0.5 --out output/optimized_picks_seedchalk_smoke.csv --out-summary output/optimized_picks_seedchalk_summary_smoke.csv

