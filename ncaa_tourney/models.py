from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    name: str
    rating: float
    source: str


@dataclass(frozen=True)
class Matchup:
    region: str
    round_name: str
    game_id: str
    team_a: str
    seed_a: int
    team_b: str
    seed_b: int
