from __future__ import annotations

from io import StringIO

import pandas as pd
import requests


REQUIRED_BRACKET_COLUMNS = ["Region", "Round", "GameId", "TeamA", "SeedA", "TeamB", "SeedB"]


def load_bracket_manual_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_BRACKET_COLUMNS)
    return _normalize_bracket(df)


def load_bracket_from_public_table(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise RuntimeError("No bracket-like table found")

    candidate = None
    for table in tables:
        cols = [str(c).lower() for c in table.columns]
        if any("team" in c for c in cols) and len(table.columns) >= 4:
            candidate = table
            break

    if candidate is None:
        raise RuntimeError("Could not parse bracket table from source")

    raise RuntimeError(
        "Auto bracket parser is intentionally conservative. "
        "Use init-bracket-template and fill manual CSV when official bracket is released."
    )


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_bracket(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Region", "Round", "GameId", "TeamA", "TeamB"]:
        out[col] = out[col].astype(str).str.strip()
    out["SeedA"] = pd.to_numeric(out["SeedA"], errors="coerce").astype("Int64")
    out["SeedB"] = pd.to_numeric(out["SeedB"], errors="coerce").astype("Int64")
    if out[["SeedA", "SeedB"]].isna().any().any():
        raise ValueError("SeedA/SeedB must be numeric for all rows")
    out = out[out["Round"].str.upper() == "R64"].copy()
    if len(out) != 32:
        print(f"Warning: expected 32 first-round games, found {len(out)}")
    return out.reset_index(drop=True)


def create_bracket_template(path: str) -> None:
    template = pd.DataFrame(
        [
            {
                "Region": "East",
                "Round": "R64",
                "GameId": "E01",
                "TeamA": "Team 1",
                "SeedA": 1,
                "TeamB": "Team 16",
                "SeedB": 16,
            },
            {
                "Region": "East",
                "Round": "R64",
                "GameId": "E02",
                "TeamA": "Team 8",
                "SeedA": 8,
                "TeamB": "Team 9",
                "SeedB": 9,
            },
        ]
    )
    template.to_csv(path, index=False)
