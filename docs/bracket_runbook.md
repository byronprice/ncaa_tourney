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

## 4) Generate strategy picks (optional baseline)

```bash
python -m ncaa_tourney.cli make-picks \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out output/strategy_picks.csv
```

## 5) Build ESPN popularity table (recommended)

Create `data/raw/espn_pick_popularity.csv` with columns:

- `Round` (`R64`, `R32`, `S16`)
- `SeedFavorite` (lower seed number)
- `SeedUnderdog` (higher seed number)
- `UnderdogPickRate` (either `0-1` or `0-100`)

Example:

```csv
Round,SeedFavorite,SeedUnderdog,UnderdogPickRate
R64,5,12,35
R64,3,14,15
R32,2,10,33
S16,1,4,31
```

## 6) Run pool-optimized picks

```bash
python -m ncaa_tourney.cli optimize-picks \
  --teams data/processed/teams.csv \
  --games data/processed/round1_games.csv \
  --pool-size 50 \
  --n-candidates 500 \
  --n-outcomes 5000 \
  --round-points 1,2,4,8,16,32 \
  --candidate-mix 0.34,0.33,0.33 \
  --opponent-mix 0.5,0.35,0.15 \
  --opponent-safe-seed-chalk-share 0.5 \
  --opponent-seed-popularity data/raw/espn_pick_popularity.csv \
  --seed 42 \
  --spread-a -0.78 \
  --spread-b 12.99 \
  --out output/optimized_picks.csv \
  --out-summary output/optimized_picks_summary.csv
```

### Common settings by pool size

Use these as starting points, then tune based on runtime and your pool tendencies.

| Pool Size | `--pool-size` | `--n-candidates` | `--n-outcomes` | `--opponent-mix` | `--opponent-safe-seed-chalk-share` |
| --- | ---: | ---: | ---: | --- | ---: |
| Small office (10–30) | `20` | `300` | `2000` | `0.6,0.3,0.1` | `0.5` |
| Medium (30–100) | `50` | `500` | `5000` | `0.5,0.35,0.15` | `0.5` |
| Large/public (100+) | `150` | `800` | `8000` | `0.4,0.35,0.25` | `0.4` |

Notes:

- Increase `--n-candidates` for broader bracket search.
- Increase `--n-outcomes` for more stable first-place equity estimates.
- If runtime is high, reduce `--n-outcomes` first.

## 7) Use outputs

- `output/simulation_summary.csv` → team advancement/title probabilities
- `output/top_brackets.csv` → likely full bracket paths
- `output/strategy_picks.csv` → direct picks for `safe`, `balanced`, `upset_heavy`
- `output/optimized_picks.csv` → single bracket chosen by first-place equity
- `output/optimized_picks_summary.csv` → top candidate brackets ranked by first-place equity
