# Bracket Runbook (Selection Sunday)

This is the fastest path from bracket release to final predictions.

## 1) Prepare bracket file

Create starter template:

```bash
python -m ncaa_tourney.cli init-bracket-template --out data/raw/bracket_manual.csv
```

Fill all 32 first-round matchups in `data/raw/bracket_manual.csv` using this format:

- `Region,Round,GameId,TeamA,SeedA,TeamB,SeedB`
- `Round` should be `R64`
- one row per first-round game

## 2) Build tournament dataset

Use existing merged ratings/tempo from `output/source_link_report.csv`:

```bash
python -m ncaa_tourney.cli build-dataset \
  --rankings-source manual_csv \
  --rankings-path output/source_link_report.csv \
  --tempo-source none \
  --bracket-source manual_csv \
  --bracket-path data/raw/bracket_manual.csv \
  --out-teams data/processed/teams.csv \
  --out-games data/processed/round1_games.csv
```

## 3) Run tournament simulation

```bash
python -m ncaa_tourney.cli simulate \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --n-sims 50000 \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out-summary output/simulation_summary.csv \
  --out-brackets output/top_brackets.csv
```

## 4) Generate ready-to-use picks

```bash
python -m ncaa_tourney.cli make-picks \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out output/strategy_picks.csv
```

## 5) Use outputs

- `output/simulation_summary.csv` → team advancement/title probabilities
- `output/top_brackets.csv` → likely full bracket paths
- `output/strategy_picks.csv` → direct picks for `safe`, `balanced`, `upset_heavy`
