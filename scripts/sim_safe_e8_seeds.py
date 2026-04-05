"""Check how often a 'safe' bracket picks specific E8 seed matchups.

Target: F4 = {1,1,2,3} seeds, with one E8 game being 3-over-5
and another being 1-over-6.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ncaa_tourney.simulation import (
    _sort_region_games,
    _build_tempo_map,
    _build_seed_map,
    _simulate_bracket_rows,
    SPREAD_TO_Z_A,
    SPREAD_TO_Z_B,
)

TEAMS_CSV = REPO_ROOT / "data/processed/teams.csv"
GAMES_CSV = REPO_ROOT / "data/processed/round1_games.csv"
N_SIMS    = 100_000
SEED      = 42


def main() -> None:
    teams_df = pd.read_csv(TEAMS_CSV)
    games_df = pd.read_csv(GAMES_CSV)

    ratings      = dict(zip(teams_df["Team"], teams_df["Rating"]))
    tempos       = _build_tempo_map(teams_df)
    seeds        = _build_seed_map(games_df)
    region_games = _sort_region_games(games_df)
    regions      = sorted(region_games.keys())

    rng = np.random.default_rng(SEED)

    # Counters
    n_f4_seeds_match   = 0  # F4 seeds exactly {1,1,2,3}
    n_has_3over5_e8    = 0  # any E8 game picks 3-seed over 5-seed
    n_has_1over6_e8    = 0  # any E8 game picks 1-seed over 6-seed
    n_both_e8          = 0  # both conditions together
    n_all_three        = 0  # all three conditions

    f4_seed_counts: Counter = Counter()

    for _ in range(N_SIMS):
        rows, _, _ = _simulate_bracket_rows(
            regions, region_games, ratings, tempos, seeds,
            SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
            strategy="safe", strategy_label="safe",
        )

        # Extract E8 rows
        e8_rows = [r for r in rows if r["Round"] == "E8"]

        # Extract F4 rows (the winners of E8 = F4 participants)
        f4_rows = [r for r in rows if r["Round"] == "F4"]
        f4_teams = [str(r["Pick"]) for r in f4_rows]
        # Also include the losers of F4 (the other two F4 teams)
        f4_all_teams = set()
        for r in f4_rows:
            f4_all_teams.add(str(r["TeamA"]))
            f4_all_teams.add(str(r["TeamB"]))

        f4_seeds = tuple(sorted(seeds.get(t, 99) for t in f4_all_teams))
        f4_seed_counts[f4_seeds] += 1

        seeds_match = (f4_seeds == (1, 1, 2, 3))

        # Check E8 seed matchups
        has_3over5 = False
        has_1over6 = False
        for r in e8_rows:
            ta, tb, pick = str(r["TeamA"]), str(r["TeamB"]), str(r["Pick"])
            sa, sb = seeds.get(ta, 99), seeds.get(tb, 99)
            winner_seed = seeds.get(pick, 99)
            loser_seed  = sb if pick == ta else sa
            # 3-seed beats 5-seed
            if winner_seed == 3 and loser_seed == 5:
                has_3over5 = True
            # 1-seed beats 6-seed
            if winner_seed == 1 and loser_seed == 6:
                has_1over6 = True

        if seeds_match:
            n_f4_seeds_match += 1
        if has_3over5:
            n_has_3over5_e8 += 1
        if has_1over6:
            n_has_1over6_e8 += 1
        if has_3over5 and has_1over6:
            n_both_e8 += 1
        if seeds_match and has_3over5 and has_1over6:
            n_all_three += 1

    print(f"Safe strategy bracket analysis over {N_SIMS:,} simulations\n")
    print(f"F4 seeds = {{1,1,2,3}}:              {n_f4_seeds_match:6,}  ({100*n_f4_seeds_match/N_SIMS:.2f}%)")
    print(f"Any E8 picks 3-seed over 5-seed:   {n_has_3over5_e8:6,}  ({100*n_has_3over5_e8/N_SIMS:.2f}%)")
    print(f"Any E8 picks 1-seed over 6-seed:   {n_has_1over6_e8:6,}  ({100*n_has_1over6_e8/N_SIMS:.2f}%)")
    print(f"Both E8 conditions together:        {n_both_e8:6,}  ({100*n_both_e8/N_SIMS:.2f}%)")
    print(f"All three conditions:               {n_all_three:6,}  ({100*n_all_three/N_SIMS:.2f}%)")
    print()
    print("Top 10 most common F4 seed combos:")
    for combo, count in f4_seed_counts.most_common(10):
        print(f"  {list(combo)}  {count:6,}  ({100*count/N_SIMS:.2f}%)")


if __name__ == "__main__":
    main()
