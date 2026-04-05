"""Simulate pool winning-score distributions for each pure opponent strategy.

For each strategy in (safe, safe_seeded, balanced, upset_heavy, chalk_plus):
  - Run N_POOL_SIMS independent pool simulations.
  - Each pool has POOL_SIZE participants all using the same strategy.
  - Score each participant against the simulated tournament outcome.
  - Record the winning (max) score for that pool.

Outputs mean / std / max of winning scores across all pool sims, plus a CSV
with every winning score for histogram analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ncaa_tourney.simulation import (
    _sort_region_games,
    _build_tempo_map,
    _build_seed_map,
    _simulate_bracket_rows,
    _safe_plus_bracket_rows,
    _score_picks,
    DEFAULT_ROUND_POINTS,
    SPREAD_TO_Z_A,
    SPREAD_TO_Z_B,
)
from ncaa_tourney.power_model import simulate_forward_bracket


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEAMS_CSV   = REPO_ROOT / "data/processed/teams.csv"
GAMES_CSV   = REPO_ROOT / "data/processed/round1_games.csv"
POWER_CSV   = REPO_ROOT / "output/power_ratings.csv"
OUT_CSV     = REPO_ROOT / "output/opponent_strategy_winning_scores.csv"

STRATEGIES  = ["safe", "safe_seeded", "balanced", "upset_heavy", "chalk_plus"]
POOL_SIZE   = 130
N_POOL_SIMS = 1000
SEED        = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_power_ratings(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    val_col = "Value" if "Value" in df.columns else "Theta"
    return {str(r.Team): float(getattr(r, val_col)) for r in df.itertuples(index=False)}


def build_weight_vector(
    regions: list[str],
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    rng: np.random.Generator,
    round_points: dict[str, int],
) -> list[int]:
    _, _, rounds = _simulate_bracket_rows(
        regions, region_games, ratings, tempos, seeds,
        SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
        strategy=None, strategy_label="template",
    )
    return [round_points.get(r, 0) for r in rounds]


def simulate_truth(
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    region_games: dict[str, list[tuple[str, str]]],
    regions: list[str],
    rng: np.random.Generator,
) -> list[str]:
    _, picks, _ = _simulate_bracket_rows(
        regions, region_games, ratings, tempos, seeds,
        SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
        strategy=None, strategy_label="truth",
    )
    return list(picks)


def generate_bracket(
    strategy: str,
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    region_games: dict[str, list[tuple[str, str]]],
    regions: list[str],
    rng: np.random.Generator,
    power_ratings: dict[str, float],
) -> list[str]:
    if strategy == "safe_seeded":
        return simulate_forward_bracket(power_ratings, region_games, rng)
    if strategy == "safe_plus":
        _, picks, _ = _safe_plus_bracket_rows(
            regions, region_games, ratings, tempos, seeds,
            SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
        )
        return list(picks)
    if strategy == "chalk_plus":
        _, picks, _ = _simulate_bracket_rows(
            regions, region_games, ratings, tempos, seeds,
            SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
            strategy=None, strategy_label="chalk_plus", determ_thresh=0.6,
        )
        return list(picks)
    # safe, balanced, upset_heavy
    _, picks, _ = _simulate_bracket_rows(
        regions, region_games, ratings, tempos, seeds,
        SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
        strategy=strategy, strategy_label=strategy,
    )
    return list(picks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    teams_df      = pd.read_csv(TEAMS_CSV)
    games_df      = pd.read_csv(GAMES_CSV)
    power_ratings = load_power_ratings(POWER_CSV)

    ratings      = dict(zip(teams_df["Team"], teams_df["Rating"]))
    tempos       = _build_tempo_map(teams_df)
    seeds        = _build_seed_map(games_df)
    region_games = _sort_region_games(games_df)
    regions      = sorted(region_games.keys())
    round_points = DEFAULT_ROUND_POINTS

    rng = np.random.default_rng(SEED)

    weight_vector = build_weight_vector(
        regions, region_games, ratings, tempos, seeds, rng, round_points
    )

    print(f"Pool size: {POOL_SIZE}   Pool sims per strategy: {N_POOL_SIMS}\n")
    print(f"{'Strategy':<15}  {'Mean':>8}  {'Std':>7}  {'Max':>6}")
    print("-" * 45)

    all_rows: list[dict] = []

    for strategy in STRATEGIES:
        winning_scores: list[int] = []

        for sim_idx in range(N_POOL_SIMS):
            truth_picks = simulate_truth(
                ratings, tempos, seeds, region_games, regions, rng
            )
            pool_scores = [
                _score_picks(
                    generate_bracket(
                        strategy, ratings, tempos, seeds,
                        region_games, regions, rng, power_ratings,
                    ),
                    truth_picks,
                    weight_vector,
                )
                for _ in range(POOL_SIZE)
            ]
            winning_score = max(pool_scores)
            winning_scores.append(winning_score)
            all_rows.append({
                "Strategy": strategy,
                "SimIndex": sim_idx,
                "WinningScore": winning_score,
            })

        arr = np.array(winning_scores)
        print(f"{strategy:<15}  {arr.mean():>8.1f}  {arr.std():>7.1f}  {arr.max():>6}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
