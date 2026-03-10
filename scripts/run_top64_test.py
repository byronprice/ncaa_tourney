from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ncaa_tourney.simulation import generate_strategy_brackets, simulate_tournament


def build_top64_bracket(source_report: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(source_report).sort_values("Rating", ascending=False).head(64).reset_index(drop=True)
    teams["OverallRank"] = teams.index + 1

    regions = ["East", "West", "South", "Midwest"]
    seed_lines = {seed: teams.iloc[(seed - 1) * 4 : seed * 4].copy() for seed in range(1, 17)}

    seeded_rows = []
    for seed in range(1, 17):
        line = seed_lines[seed]
        order = regions if seed % 2 == 1 else list(reversed(regions))
        for index, region in enumerate(order):
            row = line.iloc[index]
            seeded_rows.append(
                {
                    "Region": region,
                    "Seed": seed,
                    "Team": row["Team"],
                    "Rating": row["Rating"],
                    "Tempo": row["Tempo"],
                    "Source": row["Source"],
                    "TempoSource": row.get("TempoSource", "unknown"),
                    "OverallRank": int(row["OverallRank"]),
                }
            )

    seeded_df = pd.DataFrame(seeded_rows).sort_values("OverallRank").reset_index(drop=True)
    teams_df = seeded_df[["Team", "Rating", "Source", "Tempo", "TempoSource"]].drop_duplicates("Team")

    matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    games = []
    for region in regions:
        region_seeded = seeded_df[seeded_df["Region"] == region].set_index("Seed")
        for game_index, (seed_a, seed_b) in enumerate(matchups, start=1):
            games.append(
                {
                    "Region": region,
                    "Round": "R64",
                    "GameId": f"{region[0]}{game_index:02d}",
                    "TeamA": region_seeded.loc[seed_a, "Team"],
                    "SeedA": seed_a,
                    "TeamB": region_seeded.loc[seed_b, "Team"],
                    "SeedB": seed_b,
                }
            )

    games_df = pd.DataFrame(games)
    return teams_df, games_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run top-64 bracket test from source_link_report")
    parser.add_argument("--source-report", default="output/source_link_report.csv")
    parser.add_argument("--spread-a", type=float, default=-0.78)
    parser.add_argument("--spread-b", type=float, default=12.99)
    parser.add_argument("--n-sims", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    teams_df, games_df = build_top64_bracket(args.source_report)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(parents=True, exist_ok=True)

    teams_path = "data/processed/top64_teams.csv"
    games_path = "data/processed/top64_round1_games.csv"
    summary_path = "output/top64_simulation_summary_sigma12_1.csv"
    brackets_path = "output/top64_top_brackets_sigma12_1.csv"
    picks_path = "output/top64_strategy_picks_sigma12_1.csv"

    teams_df.to_csv(teams_path, index=False)
    games_df.to_csv(games_path, index=False)

    summary, top_brackets = simulate_tournament(
        teams_df,
        games_df,
        n_sims=args.n_sims,
        seed=args.seed,
        spread_a=args.spread_a,
        spread_b=args.spread_b,
    )
    picks = generate_strategy_brackets(
        teams_df,
        games_df,
        seed=args.seed,
        spread_a=args.spread_a,
        spread_b=args.spread_b,
    )

    summary.to_csv(summary_path, index=False)
    top_brackets.to_csv(brackets_path, index=False)
    picks.to_csv(picks_path, index=False)

    print(f"Wrote {teams_path}")
    print(f"Wrote {games_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {brackets_path}")
    print(f"Wrote {picks_path}")
    print("\nTop 10 title odds:")
    print(summary[["Team", "Win_Title", "Make_F4", "Make_NCG"]].head(10).to_string(index=False))
    print("\nStrategy champions:")
    print(picks[picks["Round"] == "Champ"][["Strategy", "Pick"]].to_string(index=False))


if __name__ == "__main__":
    main()
