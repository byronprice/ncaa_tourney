from __future__ import annotations

import pandas as pd


def build_team_and_game_tables(rankings: pd.DataFrame, bracket: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    teams_in_bracket = set(bracket["TeamA"]).union(set(bracket["TeamB"]))
    known = set(rankings["Team"])

    missing = teams_in_bracket - known
    if missing:
        fallback_rows = [
            {
                "Team": team,
                "Rating": 0.0,
                "Source": "fallback_missing",
                "Tempo": 70.0,
                "TempoSource": "tempo_default",
                "TempoMatchType": "default",
                "TempoMatchScore": 0.0,
            }
            for team in sorted(missing)
        ]
        rankings = pd.concat([rankings, pd.DataFrame(fallback_rows)], ignore_index=True)

    rankings = rankings.copy()
    if "Tempo" not in rankings.columns:
        rankings["Tempo"] = 70.0
    rankings["Tempo"] = pd.to_numeric(rankings["Tempo"], errors="coerce").fillna(70.0)
    if "TempoSource" not in rankings.columns:
        rankings["TempoSource"] = "tempo_default"
    if "TempoMatchType" not in rankings.columns:
        rankings["TempoMatchType"] = "default"
    if "TempoMatchScore" not in rankings.columns:
        rankings["TempoMatchScore"] = 0.0

    teams = rankings.drop_duplicates(subset=["Team"], keep="first").reset_index(drop=True)
    games = bracket.copy().reset_index(drop=True)
    return teams, games
