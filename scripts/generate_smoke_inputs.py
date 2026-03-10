from __future__ import annotations

import pandas as pd


def main() -> None:
    regions = ["East", "West", "South", "Midwest"]
    seed_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    games = []
    teams = []
    for region in regions:
        for game_idx, (seed_a, seed_b) in enumerate(seed_pairs, start=1):
            team_a = f"{region} Seed {seed_a}"
            team_b = f"{region} Seed {seed_b}"
            games.append(
                {
                    "Region": region,
                    "Round": "R64",
                    "GameId": f"{region[0]}{game_idx:02d}",
                    "TeamA": team_a,
                    "SeedA": seed_a,
                    "TeamB": team_b,
                    "SeedB": seed_b,
                }
            )
            teams.append({"Team": team_a, "Rating": 100 - seed_a, "Source": "synthetic"})
            teams.append({"Team": team_b, "Rating": 100 - seed_b, "Source": "synthetic"})

    pd.DataFrame(teams).drop_duplicates(subset=["Team"]).to_csv("data/raw/rankings_smoke.csv", index=False)
    pd.DataFrame(games).to_csv("data/raw/bracket_smoke.csv", index=False)


if __name__ == "__main__":
    main()
