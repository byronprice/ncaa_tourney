from __future__ import annotations

import math
from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "NCG", "Champ"]

# Maps frozenset({team_a, team_b}) -> (favorite_name, p_favorite)
R64OddsTable = dict[frozenset[str], tuple[str, float]]

# SPREAD_TO_Z_A = -0.78
# SPREAD_TO_Z_B = 12.99
SPREAD_TO_Z_A = 0.0
SPREAD_TO_Z_B = 12.1
SIGMA_70 = 12.1
NATIONAL_AVG_TEMPO = 68.0

STRATEGY_RANDOMNESS = {
    "safe": {"R64": 0.0, "R32": 0.0, "S16": 0.0, "E8": 0.0, "F4": 0.0, "NCG": 0.0},
    "balanced": {"R64": 0.18, "R32": 0.16, "S16": 0.14, "E8": 0.12, "F4": 0.1, "NCG": 0.08},
    "upset_heavy": {"R64": 0.34, "R32": 0.3, "S16": 0.26, "E8": 0.22, "F4": 0.18, "NCG": 0.14},
}

DEFAULT_ROUND_POINTS = {
    "R64": 1,
    "R32": 2,
    "S16": 4,
    "E8": 8,
    "F4": 16,
    "NCG": 32,
}

PopularityTable = dict[str, dict[tuple[int, int], float]]

# Maps round -> {team_name -> public_pick_probability}
# Used for F4/NCG/Champ where seed-based lookup no longer identifies teams uniquely.
# F4 game: use NCG probabilities (% of public picking team to reach the title game)
# NCG game: use Champ probabilities (% of public picking team to win it all)
TeamPopularityTable = dict[str, dict[str, float]]

SEED_CHALK_UNDERDOG_PROBS_BY_ROUND = {
    "R64": {
        (1, 16): 0.01,
        (2, 15): 0.08,
        (3, 14): 0.15,
        (4, 13): 0.2,
        (5, 12): 0.35,
        (6, 11): 0.3,
        (7, 10): 0.4,
        (8, 9): 0.48,
    },
    "R32": {
        (1, 8): 0.2,
        (1, 9): 0.22,
        (1, 16): 0.06,
        (2, 7): 0.28,
        (2, 10): 0.33,
        (2, 15): 0.14,
        (3, 6): 0.34,
        (3, 11): 0.39,
        (3, 14): 0.22,
        (4, 5): 0.44,
        (4, 12): 0.47,
        (4, 13): 0.3,
        (5, 12): 0.49,
        (5, 13): 0.5,
        (6, 11): 0.46,
        (6, 14): 0.53,
        (7, 10): 0.47,
        (7, 15): 0.56,
        (8, 9): 0.5,
        (8, 16): 0.58,
    },
    "S16": {
        (1, 4): 0.31,
        (1, 5): 0.34,
        (1, 8): 0.2,
        (1, 9): 0.23,
        (1, 12): 0.38,
        (1, 13): 0.42,
        (1, 16): 0.08,
        (2, 3): 0.43,
        (2, 6): 0.31,
        (2, 7): 0.35,
        (2, 10): 0.39,
        (2, 11): 0.42,
        (2, 14): 0.5,
        (2, 15): 0.2,
        (3, 6): 0.39,
        (3, 7): 0.41,
        (3, 10): 0.45,
        (3, 11): 0.47,
        (3, 14): 0.56,
        (4, 5): 0.48,
        (4, 8): 0.39,
        (4, 9): 0.42,
        (4, 12): 0.5,
        (4, 13): 0.54,
        (5, 8): 0.44,
        (5, 9): 0.46,
        (5, 12): 0.53,
        (5, 13): 0.56,
        (6, 7): 0.49,
        (6, 10): 0.53,
        (6, 11): 0.55,
        (6, 14): 0.62,
        (7, 10): 0.54,
        (7, 11): 0.57,
        (7, 15): 0.66,
        (8, 9): 0.51,
        (8, 12): 0.58,
        (8, 13): 0.61,
        (8, 16): 0.7,
    },
}

# to run optimize-picks
#  python -m ncaa_tourney.cli optimize-picks --teams data/processed/teams.csv --opponent-teams output/source_link_report_espn.csv --games data/processed/round1_games.csv --r64-odds data/raw/round1_game_odds.csv --pool-size 30,30 --n-candidates 1000 --n-outcomes 5000 --round-points 1,2,4,8,16,32 --candidate-mix 0.20,0.30,0.50 --opponent-mix 0.5,0.4,0.1 --opponent-safe-seed-chalk-share 0.25 --opponent-seed-popularity data/raw/espn_pick_popularity.csv --opponent-team-popularity data/raw/espn_team_popularity.csv --seed 53 --out output/optimized_picks.csv --out-summary output/optimized_picks_summary.csv


@dataclass
class CandidateBracket:
    rows: list[dict[str, str | int | float]]
    picks: tuple[str, ...]
    rounds: list[str]
    strategy: str
    first_place_equity: float = 0.0
    win_rate: float = 0.0
    top_tie_rate: float = 0.0


def win_probability(
    rating_a: float,
    rating_b: float,
    tempo_a: float = 70.0,
    tempo_b: float = 70.0,
    sigma70: float = SIGMA_70,
    spread_a: float = SPREAD_TO_Z_A,
    spread_b: float = SPREAD_TO_Z_B,
    national_avg_tempo: float = NATIONAL_AVG_TEMPO,
) -> float:
    possessions = max(tempo_a + tempo_b - national_avg_tempo, 55.0)
    expected_spread = (rating_a - rating_b) * (possessions / 70.0)
    sigma = spread_b * math.sqrt(possessions / 70.0)

    fav_spread = abs(expected_spread)
    z_fav = (fav_spread - spread_a) / sigma
    p_fav = 0.5 * (1.0 + math.erf(z_fav / math.sqrt(2.0)))

    p_a = p_fav if expected_spread >= 0 else 1.0 - p_fav
    return float(np.clip(p_a, 0.01, 0.99))


def estimate_championship_total(
    team_a: str,
    team_b: str,
    teams_df: pd.DataFrame,
    efficiencies: dict[str, tuple[float, float]] | None = None,
    national_avg_eff: float = 103.0,
    national_avg_tempo: float = NATIONAL_AVG_TEMPO,
) -> dict:
    """Estimate the total score of a game for tiebreaker purposes.

    If `efficiencies` is provided (from load_kenpom_efficiencies), actual AdjO/AdjD
    per 100 possessions are used.  Scoring uses the standard KenPom formula:
        score_A = AdjO_A * (AdjD_B / national_avg_eff) * (possessions / 100)
    """
    def _get_row(name: str) -> pd.Series:
        rows = teams_df[teams_df["Team"] == name]
        if rows.empty:
            raise ValueError(f"Team not found: {name!r}")
        return rows.iloc[0]

    row_a, row_b = _get_row(team_a), _get_row(team_b)

    if efficiencies is None:
        raise ValueError("efficiencies dict is required — use load_kenpom_efficiencies()")

    from ncaa_tourney.rankings import _canonical_team_key, _resolve_alias

    def _lookup_eff(name: str) -> tuple[float, float]:
        key = _resolve_alias(_canonical_team_key(name))
        if key not in efficiencies:
            raise ValueError(f"No efficiency data for {name!r} (canonical key: {key!r})")
        return efficiencies[key]

    adj_o_a, adj_d_a = _lookup_eff(team_a)
    adj_o_b, adj_d_b = _lookup_eff(team_b)

    possessions = max(float(row_a["Tempo"]) + float(row_b["Tempo"]) - national_avg_tempo, 55.0)

    score_a = adj_o_a * (adj_d_b / national_avg_eff) * (possessions / 100.0)
    score_b = adj_o_b * (adj_d_a / national_avg_eff) * (possessions / 100.0)

    return {
        "team_a": team_a,  "score_a": round(score_a, 1),
        "team_b": team_b,  "score_b": round(score_b, 1),
        "total":  round(score_a + score_b, 1),
        "possessions": round(possessions, 1),
        "adj_o_a": round(adj_o_a, 1), "adj_d_a": round(adj_d_a, 1),
        "adj_o_b": round(adj_o_b, 1), "adj_d_b": round(adj_d_b, 1),
    }


def simulate_tournament(
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
    n_sims: int,
    seed: int = 42,
    sigma70: float = SIGMA_70,
    spread_a: float = SPREAD_TO_Z_A,
    spread_b: float = SPREAD_TO_Z_B,
    r64_odds: R64OddsTable | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ratings = dict(zip(teams_df["Team"], teams_df["Rating"]))
    tempos = _build_tempo_map(teams_df)

    advancement = {team: defaultdict(int) for team in ratings.keys()}
    path_counter: Counter[tuple[str, ...]] = Counter()

    region_games = _sort_region_games(games_df)
    regions = sorted(region_games.keys())
    if len(regions) != 4:
        print(f"Warning: expected 4 regions, found {len(regions)}")

    f4_region_order = ["East", "South", "Midwest", "West"]

    for _ in range(n_sims):
        champs_by_region: dict[str, str] = {}
        full_path = []

        for region in regions:
            first_round = region_games[region]
            r64_winners = []
            for game in first_round:
                winner = _simulate_game(game[0], game[1], ratings, tempos, sigma70, spread_a, spread_b, rng, r64_odds)
                r64_winners.append(winner)
                full_path.append(f"{region}:R64:{winner}")
                advancement[winner]["R32"] += 1

            r32_winners = _play_round(
                r64_winners,
                ratings,
                tempos,
                sigma70,
                spread_a,
                spread_b,
                rng,
                advancement,
                "S16",
                full_path,
                region,
                "R32",
            )
            s16_winners = _play_round(
                r32_winners,
                ratings,
                tempos,
                sigma70,
                spread_a,
                spread_b,
                rng,
                advancement,
                "E8",
                full_path,
                region,
                "S16",
            )
            e8_winner = _play_round(
                s16_winners,
                ratings,
                tempos,
                sigma70,
                spread_a,
                spread_b,
                rng,
                advancement,
                "F4",
                full_path,
                region,
                "E8",
            )
            champs_by_region[region] = e8_winner[0]

        regional_champs = [champs_by_region[r] for r in f4_region_order if r in champs_by_region]

        f4_winners = _play_round(
            regional_champs,
            ratings,
            tempos,
            sigma70,
            spread_a,
            spread_b,
            rng,
            advancement,
            "NCG",
            full_path,
            "FinalFour",
            "F4",
        )
        champion = _play_round(
            f4_winners,
            ratings,
            tempos,
            sigma70,
            spread_a,
            spread_b,
            rng,
            advancement,
            "Champ",
            full_path,
            "Final",
            "NCG",
        )
        if champion:
            full_path.append(f"Champion:{champion[0]}")

        path_counter[tuple(full_path)] += 1

    summary = _build_advancement_summary(advancement, n_sims)
    top_paths = _build_top_paths(path_counter, n_sims)
    return summary, top_paths


def generate_strategy_brackets(
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
    seed: int = 42,
    sigma70: float = 10.5,
    spread_a: float = SPREAD_TO_Z_A,
    spread_b: float = SPREAD_TO_Z_B,
    r64_odds: R64OddsTable | None = None,
) -> pd.DataFrame:
    ratings = dict(zip(teams_df["Team"], teams_df["Rating"]))
    tempos = _build_tempo_map(teams_df)
    region_games = _sort_region_games(games_df)
    regions = sorted(region_games.keys())

    rows: list[dict[str, str | int | float]] = []
    for offset, strategy in enumerate(["safe", "balanced", "upset_heavy"]):
        rng = np.random.default_rng(seed + offset)
        rows.extend(
            _run_strategy_once(strategy, regions, region_games, ratings, tempos, sigma70, spread_a, spread_b, rng, r64_odds)
        )

    return pd.DataFrame(rows)


def _greedy_portfolio_select(
    scores_matrix: np.ndarray,
    opp_max_matrix: np.ndarray,
    opp_ties_matrix: np.ndarray,
    pool_weights: list[float] | None = None,
) -> list[tuple[int, float, float]]:
    """Weighted global greedy bracket selection across all pools.

    At each step, picks the (pool, candidate) pair with the highest
    weight * marginal_equity, where weight = payout / pool_size.
    This ensures high-EV pools get priority access to the best candidates
    regardless of their position in the pool list.

    Returns a list of (candidate_idx, marginal_equity, individual_equity)
    ordered by pool index (not assignment order).
    """
    n_entries = opp_max_matrix.shape[0]
    n_outcomes = scores_matrix.shape[1]
    weights = pool_weights if pool_weights is not None else [1.0] * n_entries
    already_won = np.zeros(n_outcomes, dtype=bool)
    assigned: list[tuple[int, float, float] | None] = [None] * n_entries
    unassigned = list(range(n_entries))

    while unassigned:
        best_weighted = -1.0
        best_pool = -1
        best_candidate = -1
        best_marginal = 0.0
        best_individual = 0.0

        for k in unassigned:
            opp_max_k = opp_max_matrix[k]
            opp_ties_k = opp_ties_matrix[k]
            wins = scores_matrix > opp_max_k[np.newaxis, :]
            tie_mask = scores_matrix == opp_max_k[np.newaxis, :]
            tie_vals = np.where(tie_mask, 1.0 / (opp_ties_k[np.newaxis, :].astype(np.float64) + 1.0), 0.0)
            equity = wins.astype(np.float64) + tie_vals
            uncovered = ~already_won
            marginal = (equity * uncovered[np.newaxis, :]).mean(axis=1)
            individual = equity.mean(axis=1)
            best_c = int(np.argmax(marginal))
            weighted = weights[k] * float(marginal[best_c])
            if weighted > best_weighted:
                best_weighted = weighted
                best_pool = k
                best_candidate = best_c
                best_marginal = float(marginal[best_c])
                best_individual = float(individual[best_c])

        assigned[best_pool] = (best_candidate, best_marginal, best_individual)
        wins_k = scores_matrix[best_candidate] > opp_max_matrix[best_pool]
        already_won |= wins_k
        unassigned.remove(best_pool)

    return [a for a in assigned if a is not None]


def optimize_pool_bracket(
    teams_df: pd.DataFrame,
    games_df: pd.DataFrame,
    pool_sizes: list[int] | int = 50,
    pool_size: int | None = None,
    n_candidates: int = 300,
    n_outcomes: int = 2000,
    seed: int = 42,
    sigma70: float = 10.5,
    spread_a: float = SPREAD_TO_Z_A,
    spread_b: float = SPREAD_TO_Z_B,
    round_points: dict[str, int] | None = None,
    candidate_mix: dict[str, float] | None = None,
    opponent_mix: dict[str, float] | None = None,
    opponent_safe_seed_chalk_share: float = 0.0,
    opponent_seed_popularity: PopularityTable | None = None,
    r64_odds: R64OddsTable | None = None,
    opponent_teams_df: pd.DataFrame | None = None,
    opponent_team_popularity: TeamPopularityTable | None = None,
    pool_payouts: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Backward-compatible: old callers may pass pool_size= as a keyword arg
    if pool_size is not None:
        pool_sizes = pool_size
    _pool_sizes: list[int] = [pool_sizes] if isinstance(pool_sizes, int) else list(pool_sizes)
    n_entries = len(_pool_sizes)
    for ps in _pool_sizes:
        if ps < 2:
            raise ValueError("Each pool size must be at least 2")
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if n_outcomes < 1:
        raise ValueError("n_outcomes must be at least 1")
    if opponent_safe_seed_chalk_share < 0.0 or opponent_safe_seed_chalk_share > 1.0:
        raise ValueError("opponent_safe_seed_chalk_share must be between 0 and 1")

    ratings = dict(zip(teams_df["Team"], teams_df["Rating"]))
    tempos = _build_tempo_map(teams_df)
    opp_df = opponent_teams_df if opponent_teams_df is not None else teams_df
    opponent_ratings = dict(zip(opp_df["Team"], opp_df["Rating"]))
    opponent_tempos = _build_tempo_map(opp_df)
    seeds = _build_seed_map(games_df)
    region_games = _sort_region_games(games_df)
    regions = sorted(region_games.keys())

    round_points = round_points or DEFAULT_ROUND_POINTS
    strategy_names = list(STRATEGY_RANDOMNESS.keys())
    candidate_weights = _normalize_strategy_mix(candidate_mix, strategy_names)
    opponent_weights = _normalize_strategy_mix(opponent_mix, strategy_names)

    rng = np.random.default_rng(seed)
    candidate_store: dict[tuple[str, ...], CandidateBracket] = {}

    for _ in range(n_candidates):
        strategy = str(rng.choice(strategy_names, p=candidate_weights))
        rows, picks, rounds = _simulate_bracket_rows(
            regions,
            region_games,
            ratings,
            tempos,
            seeds,
            sigma70,
            spread_a,
            spread_b,
            rng,
            strategy,
            strategy,
            opponent_seed_popularity,
            r64_odds,
        )
        key = tuple(picks)
        if key not in candidate_store:
            candidate_store[key] = CandidateBracket(
                rows=rows,
                picks=key,
                rounds=rounds,
                strategy=strategy,
            )

    candidates = list(candidate_store.values())
    if not candidates:
        raise RuntimeError("No candidate brackets generated")

    rounds_template = list(candidates[0].rounds)
    weight_vector = [int(round_points.get(round_name, 0)) for round_name in rounds_template]

    n_cands = len(candidates)
    scores_matrix = np.empty((n_cands, n_outcomes), dtype=np.int32)
    opp_max_matrix = np.empty((n_entries, n_outcomes), dtype=np.int32)
    opp_ties_matrix = np.empty((n_entries, n_outcomes), dtype=np.int32)

    def _simulate_opponent(truth_picks: list[str]) -> int:
        """Simulate one opponent and return (score, picks) — inline helper."""
        opp_strategy = str(rng.choice(strategy_names, p=opponent_weights))
        if opp_strategy == "safe" and rng.random() < opponent_safe_seed_chalk_share:
            opp_strategy = "safe_seeded"
        _, opp_picks, _ = _simulate_bracket_rows(
            regions,
            region_games,
            opponent_ratings,
            opponent_tempos,
            seeds,
            sigma70,
            spread_a,
            spread_b,
            rng,
            strategy=opp_strategy,
            strategy_label=opp_strategy,
            seed_popularity=opponent_seed_popularity,
            r64_odds=None,
            team_popularity=opponent_team_popularity,
        )
        return _score_picks(opp_picks, truth_picks, weight_vector)

    for i in range(n_outcomes):
        _, truth_picks, _ = _simulate_bracket_rows(
            regions,
            region_games,
            ratings,
            tempos,
            seeds,
            sigma70,
            spread_a,
            spread_b,
            rng,
            strategy=None,
            strategy_label="truth",
            seed_popularity=None,
            r64_odds=r64_odds,
        )

        # Candidate scores for this outcome
        for c_idx, candidate in enumerate(candidates):
            scores_matrix[c_idx, i] = _score_picks(list(candidate.picks), truth_picks, weight_vector)

        # Per-pool opponent simulation (independent draw per pool)
        for k, ps in enumerate(_pool_sizes):
            opp_scores = [_simulate_opponent(truth_picks) for _ in range(ps - 1)]
            opp_max = max(opp_scores)
            opp_max_matrix[k, i] = opp_max
            opp_ties_matrix[k, i] = sum(1 for s in opp_scores if s == opp_max)

    # Accumulate per-candidate equity vs pool 0 for the summary ranking
    opp_max_0 = opp_max_matrix[0]
    opp_ties_0 = opp_ties_matrix[0]
    for c_idx, candidate in enumerate(candidates):
        cscores = scores_matrix[c_idx]
        wins = int(np.sum(cscores > opp_max_0))
        ties = np.where(cscores == opp_max_0)[0]
        candidate.first_place_equity = (wins + sum(1.0 / (opp_ties_0[t] + 1) for t in ties)) / n_outcomes
        candidate.win_rate = wins / n_outcomes
        candidate.top_tie_rate = len(ties) / n_outcomes

    _payouts = list(pool_payouts) if pool_payouts is not None else [1.0] * n_entries
    if len(_payouts) != n_entries:
        raise ValueError(f"pool_payouts length {len(_payouts)} must match number of pools {n_entries}")
    pool_weights = [p / ps for p, ps in zip(_payouts, _pool_sizes)]
    selected = _greedy_portfolio_select(scores_matrix, opp_max_matrix, opp_ties_matrix, pool_weights)

    cumulative = 0.0
    all_rows = []
    for entry_rank, (c_idx, marginal_eq, individual_eq) in enumerate(selected, start=1):
        candidate = candidates[c_idx]
        cumulative += marginal_eq
        for row in candidate.rows:
            out_row = dict(row)
            out_row["Strategy"] = "optimized"
            out_row["EntryRank"] = entry_rank
            out_row["PoolSize"] = _pool_sizes[entry_rank - 1]
            out_row["CandidateStrategy"] = candidate.strategy
            out_row["IndividualEquity"] = round(individual_eq, 6)
            out_row["MarginalPortfolioEquity"] = round(marginal_eq, 6)
            out_row["CumulativePortfolioEquity"] = round(cumulative, 6)
            all_rows.append(out_row)

    # Per-entry summary: one row per assigned entry showing pool context and champion pick
    entry_summary_rows = []
    for entry_rank, (c_idx, marginal_eq, individual_eq) in enumerate(selected, start=1):
        candidate = candidates[c_idx]
        ps = _pool_sizes[entry_rank - 1]
        payout = _payouts[entry_rank - 1]
        champ_row = next((r for r in candidate.rows if r.get("Round") == "Champ"), None)
        champion = champ_row["Pick"] if champ_row else ""
        entry_summary_rows.append(
            {
                "EntryRank": entry_rank,
                "PoolSize": ps,
                "Payout": payout,
                "EVWeight": round(payout / ps, 4),
                "Champion": champion,
                "CandidateStrategy": candidate.strategy,
                "IndividualEquity": round(individual_eq, 6),
                "MarginalPortfolioEquity": round(marginal_eq, 6),
                "CumulativePortfolioEquity": round(cumulative, 6),
            }
        )

    # Candidate summary: top 25 by first-place equity vs pool 0
    ranked = sorted(candidates, key=lambda row: row.first_place_equity, reverse=True)
    candidate_summary_rows = []
    for index, candidate in enumerate(ranked[:25], start=1):
        candidate_summary_rows.append(
            {
                "Rank": index,
                "FirstPlaceEquity": round(candidate.first_place_equity, 6),
                "WinOutrightRate": round(candidate.win_rate, 6),
                "TopTieRate": round(candidate.top_tie_rate, 6),
                "CandidateStrategy": candidate.strategy,
            }
        )

    return pd.DataFrame(all_rows), pd.DataFrame(entry_summary_rows), pd.DataFrame(candidate_summary_rows)


def _sample_forced_f4_teams(
    team_popularity: TeamPopularityTable,
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
    rng: np.random.Generator,
) -> tuple[dict[str, str], str | None]:
    """Backward-sample forced regional champions to match ESPN public pick popularity.

    Algorithm:
      1. Sample champion from Champ% (all tournament teams eligible; unlisted → BPI weight)
      2. Sample NCG opponent from NCG% restricted to the opposite bracket half
      3. Sample the two F4 losers from F4% restricted to their individual remaining regions

    Returns (forced_regional_champs, forced_champion):
      - forced_regional_champs: {region: team} for all 4 regions
      - forced_champion: the team that must also win F4 and NCG
    """
    from ncaa_tourney.rankings import _canonical_team_key, _resolve_alias

    region_to_teams: dict[str, set[str]] = {}
    for region, games in region_games.items():
        ts: set[str] = set()
        for a, b in games:
            ts.add(a)
            ts.add(b)
        region_to_teams[region] = ts

    all_game_teams = {t for ts in region_to_teams.values() for t in ts}
    canon_to_game: dict[str, str] = {}
    for t in all_game_teams:
        key = _resolve_alias(_canonical_team_key(t))
        canon_to_game[key] = t

    def get_weights(round_name: str, eligible: set[str]) -> dict[str, float]:
        pop = team_popularity.get(round_name, {})
        weights: dict[str, float] = {}
        listed_in: set[str] = set()
        listed_sum = 0.0
        for pop_name, prob in pop.items():
            key = _resolve_alias(_canonical_team_key(pop_name))
            game_name = canon_to_game.get(key)
            if game_name and game_name in eligible:
                weights[game_name] = float(prob)
                listed_sum += float(prob)
                listed_in.add(game_name)
        unlisted = eligible - listed_in
        if unlisted:
            residual = max(0.0, 1.0 - listed_sum)
            bpi_sum = sum(max(ratings.get(t, 0.0), 0.0) for t in unlisted)
            if residual > 0.0 and bpi_sum > 0.0:
                for t in unlisted:
                    weights[t] = (max(ratings.get(t, 0.0), 0.0) / bpi_sum) * residual
            elif bpi_sum > 0.0:
                # Listed teams exhausted the mass — give unlisted a tiny BPI-proportional weight
                for t in unlisted:
                    weights[t] = (max(ratings.get(t, 0.0), 0.0) / bpi_sum) * 1e-4
            else:
                for t in unlisted:
                    weights[t] = 1e-4 / len(unlisted)
        return weights

    def draw(weights: dict[str, float]) -> str:
        teams_list = list(weights.keys())
        if not teams_list:
            return ""
        w = np.array([weights[t] for t in teams_list], dtype=np.float64)
        total = w.sum()
        if total <= 0.0:
            return str(rng.choice(teams_list))
        w /= total
        return teams_list[int(rng.choice(len(teams_list), p=w))]

    half_a = ["East", "South"]
    half_b = ["Midwest", "West"]

    # Step 1: Champion (all teams eligible)
    champion = draw(get_weights("Champ", all_game_teams))
    champ_region = next((r for r, ts in region_to_teams.items() if champion in ts), None)
    if not champ_region:
        return {}, None

    champ_half = half_a if champ_region in half_a else half_b
    opp_half = half_b if champ_half == half_a else half_a

    # Step 2: NCG opponent from opposite half
    opp_half_teams = {t for r in opp_half for t in region_to_teams.get(r, set())}
    ncg_opponent = draw(get_weights("NCG", opp_half_teams))
    ncg_region = next((r for r, ts in region_to_teams.items() if ncg_opponent in ts), None)

    forced: dict[str, str] = {champ_region: champion}
    if ncg_region:
        forced[ncg_region] = ncg_opponent

    # Step 3: F4 losers — the remaining region in each half
    champ_other = next((r for r in champ_half if r != champ_region), None)
    opp_other = next((r for r in opp_half if r != ncg_region), None) if ncg_region else None

    if champ_other:
        forced[champ_other] = draw(get_weights("F4", region_to_teams.get(champ_other, set())))
    if opp_other:
        forced[opp_other] = draw(get_weights("F4", region_to_teams.get(opp_other, set())))

    return forced, champion


def _sort_region_games(games_df: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for region, group in games_df.groupby("Region"):
        ordered = group.sort_values("GameId")
        grouped[str(region)] = [(str(row.TeamA), str(row.TeamB)) for row in ordered.itertuples(index=False)]
    return grouped


def _build_tempo_map(teams_df: pd.DataFrame) -> dict[str, float]:
    if "Tempo" not in teams_df.columns:
        return {str(team): 70.0 for team in teams_df["Team"].tolist()}

    return {
        str(team): float(tempo) if pd.notna(tempo) else 70.0
        for team, tempo in zip(teams_df["Team"].tolist(), teams_df["Tempo"].tolist())
    }


def _build_seed_map(games_df: pd.DataFrame) -> dict[str, int]:
    required = {"TeamA", "TeamB", "SeedA", "SeedB"}
    if not required.issubset(set(games_df.columns)):
        return {}

    seed_map: dict[str, int] = {}
    for row in games_df.itertuples(index=False):
        seed_a = pd.to_numeric(getattr(row, "SeedA"), errors="coerce")
        seed_b = pd.to_numeric(getattr(row, "SeedB"), errors="coerce")
        if pd.notna(seed_a):
            seed_map[str(row.TeamA)] = int(float(seed_a))
        if pd.notna(seed_b):
            seed_map[str(row.TeamB)] = int(float(seed_b))
    return seed_map


def _normalize_strategy_mix(mix: dict[str, float] | None, strategy_names: list[str]) -> list[float]:
    if mix is None:
        return [1.0 / len(strategy_names)] * len(strategy_names)

    cleaned = [max(float(mix.get(name, 0.0)), 0.0) for name in strategy_names]
    total = sum(cleaned)
    if total <= 0:
        raise ValueError("Strategy mix must include at least one positive weight")
    return [value / total for value in cleaned]


def _score_picks(picks: list[str], truth_picks: list[str], weight_vector: list[int]) -> int:
    return int(sum(weight for pick, truth, weight in zip(picks, truth_picks, weight_vector) if pick == truth))


def _simulate_bracket_rows(
    regions: list[str],
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    strategy: str | None,
    strategy_label: str,
    seed_popularity: PopularityTable | None,
    r64_odds: R64OddsTable | None = None,
    team_popularity: TeamPopularityTable | None = None,
) -> tuple[list[dict[str, str | int | float]], list[str], list[str]]:
    rows: list[dict[str, str | int | float]] = []
    champs_by_region: dict[str, str] = {}

    # Backward-sample forced F4 teams for safe_seeded opponents when team popularity is available
    forced_regional_champs: dict[str, str] = {}
    forced_champion: str | None = None
    if strategy == "safe_seeded" and team_popularity is not None:
        forced_regional_champs, forced_champion = _sample_forced_f4_teams(
            team_popularity, region_games, ratings, rng
        )

    for region in regions:
        forced_for_region = forced_regional_champs.get(region)
        current_teams: list[str] = []
        for game_index, (team_a, team_b) in enumerate(region_games[region], start=1):
            winner, base_p_a, adjusted_p_a = _select_game_winner(
                team_a,
                team_b,
                ratings,
                tempos,
                seeds,
                strategy,
                "R64",
                sigma70,
                spread_a,
                spread_b,
                rng,
                seed_popularity,
                r64_odds,
                forced_winner=forced_for_region,
            )
            rows.append(
                _pick_row(
                    strategy_label,
                    "R64",
                    region,
                    game_index,
                    team_a,
                    team_b,
                    winner,
                    base_p_a,
                    adjusted_p_a,
                )
            )
            current_teams.append(winner)

        for round_name in ["R32", "S16", "E8"]:
            current_teams, new_rows = _run_round_pairs_for_strategy(
                strategy_label,
                strategy,
                round_name,
                region,
                current_teams,
                ratings,
                tempos,
                seeds,
                sigma70,
                spread_a,
                spread_b,
                rng,
                seed_popularity,
                forced_winner=forced_for_region,
            )
            rows.extend(new_rows)

        champs_by_region[region] = current_teams[0]

    f4_region_order = ["East", "South", "Midwest", "West"]
    regional_champs = [champs_by_region[r] for r in f4_region_order if r in champs_by_region]

    ff_winners, ff_rows = _run_round_pairs_for_strategy(
        strategy_label,
        strategy,
        "F4",
        "FinalFour",
        regional_champs,
        ratings,
        tempos,
        seeds,
        sigma70,
        spread_a,
        spread_b,
        rng,
        seed_popularity,
        forced_winner=forced_champion,
    )
    rows.extend(ff_rows)

    title_winner, ncg_rows = _run_round_pairs_for_strategy(
        strategy_label,
        strategy,
        "NCG",
        "Final",
        ff_winners,
        ratings,
        tempos,
        seeds,
        sigma70,
        spread_a,
        spread_b,
        rng,
        seed_popularity,
        forced_winner=forced_champion,
    )
    rows.extend(ncg_rows)

    if title_winner:
        rows.append(
            {
                "Strategy": strategy_label,
                "Round": "Champ",
                "Region": "Final",
                "GameIndex": 1,
                "TeamA": title_winner[0],
                "TeamB": "",
                "Pick": title_winner[0],
                "TeamA_WinProb_Base": 1.0,
                "TeamA_WinProb_Adjusted": 1.0,
            }
        )

    scored_rows = [row for row in rows if row["Round"] != "Champ"]
    picks = [str(row["Pick"]) for row in scored_rows]
    rounds = [str(row["Round"]) for row in scored_rows]
    return rows, picks, rounds


def _run_round_pairs_for_strategy(
    strategy_label: str,
    strategy: str | None,
    round_name: str,
    region: str,
    teams: list[str],
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    seed_popularity: PopularityTable | None,
    forced_winner: str | None = None,
) -> tuple[list[str], list[dict[str, str | int | float]]]:
    winners: list[str] = []
    rows: list[dict[str, str | int | float]] = []
    for i in range(0, len(teams), 2):
        team_a, team_b = teams[i], teams[i + 1]
        winner, base_p_a, adjusted_p_a = _select_game_winner(
            team_a,
            team_b,
            ratings,
            tempos,
            seeds,
            strategy,
            round_name,
            sigma70,
            spread_a,
            spread_b,
            rng,
            seed_popularity,
            forced_winner=forced_winner,
        )
        winners.append(winner)
        rows.append(
            _pick_row(
                strategy_label,
                round_name,
                region,
                (i // 2) + 1,
                team_a,
                team_b,
                winner,
                base_p_a,
                adjusted_p_a,
            )
        )

    return winners, rows


def _select_game_winner(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    strategy: str | None,
    round_name: str,
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    seed_popularity: PopularityTable | None,
    r64_odds: R64OddsTable | None = None,
    forced_winner: str | None = None,
) -> tuple[str, float, float]:
    base_p_a = win_probability(
        ratings.get(team_a, 0.0),
        ratings.get(team_b, 0.0),
        tempos.get(team_a, 70.0),
        tempos.get(team_b, 70.0),
        sigma70=sigma70,
        spread_a=spread_a,
        spread_b=spread_b,
    )
    if r64_odds is not None and round_name == "R64":
        key = frozenset({team_a, team_b})
        if key in r64_odds:
            fav_name, p_fav = r64_odds[key]
            base_p_a = p_fav if team_a == fav_name else 1.0 - p_fav

    if forced_winner is not None and forced_winner in (team_a, team_b):
        adj = 1.0 if forced_winner == team_a else 0.0
        return forced_winner, base_p_a, adj

    if strategy is None:
        winner = team_a if rng.random() < base_p_a else team_b
        return winner, base_p_a, base_p_a

    if strategy == "safe_seeded":
        adjusted_p_a = _seed_chalk_probability(team_a, team_b, round_name, base_p_a, seeds, seed_popularity)
        winner = team_a if rng.random() < adjusted_p_a else team_b
        return winner, base_p_a, adjusted_p_a

    randomness = STRATEGY_RANDOMNESS[strategy].get(round_name, 0.0)
    p_favorite = max(base_p_a, 1.0 - base_p_a)
    p_favorite_adj = (1.0 - randomness) * p_favorite + randomness * 0.5
    adjusted_p_a = p_favorite_adj if base_p_a >= 0.5 else 1.0 - p_favorite_adj
    adjusted_p_a = float(np.clip(adjusted_p_a, 0.01, 0.99))
    winner = team_a if rng.random() < adjusted_p_a else team_b
    return winner, base_p_a, adjusted_p_a


def _seed_chalk_probability(
    team_a: str,
    team_b: str,
    round_name: str,
    base_p_a: float,
    seeds: dict[str, int],
    seed_popularity: PopularityTable | None,
) -> float:
    seed_a = seeds.get(team_a)
    seed_b = seeds.get(team_b)
    if seed_a is None or seed_b is None:
        return base_p_a

    favorite_is_a = seed_a < seed_b
    seed_favorite = min(seed_a, seed_b)
    seed_underdog = max(seed_a, seed_b)

    round_probs = None
    if seed_popularity and round_name in seed_popularity:
        round_probs = seed_popularity[round_name]
    elif round_name in SEED_CHALK_UNDERDOG_PROBS_BY_ROUND:
        round_probs = SEED_CHALK_UNDERDOG_PROBS_BY_ROUND[round_name]

    if round_probs is not None:
        p_underdog = round_probs.get((seed_favorite, seed_underdog))
        if p_underdog is None:
            seed_gap = max(seed_underdog - seed_favorite, 0)
            p_underdog = float(np.clip(0.4 - (0.025 * seed_gap), 0.12, 0.5))
    else:
        seed_gap = max(seed_underdog - seed_favorite, 0)
        p_underdog = float(np.clip(0.45 - (0.03 * seed_gap), 0.12, 0.45))

    p_favorite = 1.0 - p_underdog
    p_a = p_favorite if favorite_is_a else p_underdog
    return float(np.clip(p_a, 0.01, 0.99))


def _play_round(
    teams: list[str],
    ratings: dict[str, float],
    tempos: dict[str, float],
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    advancement: dict[str, defaultdict],
    advances_to: str,
    full_path: list[str],
    region: str,
    current_round: str,
) -> list[str]:
    winners = []
    for i in range(0, len(teams), 2):
        winner = _simulate_game(teams[i], teams[i + 1], ratings, tempos, sigma70, spread_a, spread_b, rng)
        winners.append(winner)
        full_path.append(f"{region}:{current_round}:{winner}")
        advancement[winner][advances_to] += 1
    return winners


def _simulate_game(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    tempos: dict[str, float],
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    r64_odds: R64OddsTable | None = None,
) -> str:
    p_a = win_probability(
        ratings.get(team_a, 0.0),
        ratings.get(team_b, 0.0),
        tempos.get(team_a, 70.0),
        tempos.get(team_b, 70.0),
        sigma70=sigma70,
        spread_a=spread_a,
        spread_b=spread_b,
    )
    if r64_odds is not None:
        key = frozenset({team_a, team_b})
        if key in r64_odds:
            fav_name, p_fav = r64_odds[key]
            p_a = p_fav if team_a == fav_name else 1.0 - p_fav
    return team_a if rng.random() < p_a else team_b


def _simulate_game_with_strategy(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    tempos: dict[str, float],
    strategy: str,
    round_name: str,
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    r64_odds: R64OddsTable | None = None,
) -> tuple[str, float, float]:
    base_p_a = win_probability(
        ratings.get(team_a, 0.0),
        ratings.get(team_b, 0.0),
        tempos.get(team_a, 70.0),
        tempos.get(team_b, 70.0),
        sigma70=sigma70,
        spread_a=spread_a,
        spread_b=spread_b,
    )
    if r64_odds is not None and round_name == "R64":
        key = frozenset({team_a, team_b})
        if key in r64_odds:
            fav_name, p_fav = r64_odds[key]
            base_p_a = p_fav if team_a == fav_name else 1.0 - p_fav
    randomness = STRATEGY_RANDOMNESS[strategy].get(round_name, 0.0)

    p_favorite = max(base_p_a, 1.0 - base_p_a)
    p_favorite_adj = (1.0 - randomness) * p_favorite + randomness * 0.5
    p_a = p_favorite_adj if base_p_a >= 0.5 else 1.0 - p_favorite_adj
    p_a = float(np.clip(p_a, 0.01, 0.99))
    winner = team_a if rng.random() < p_a else team_b
    return winner, base_p_a, p_a


def _run_strategy_once(
    strategy: str,
    regions: list[str],
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
    tempos: dict[str, float],
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
    r64_odds: R64OddsTable | None = None,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    champs_by_region: dict[str, str] = {}

    for region in regions:
        current_teams: list[str] = []
        for game_index, (team_a, team_b) in enumerate(region_games[region], start=1):
            winner, base_p_a, adjusted_p_a = _simulate_game_with_strategy(
                team_a, team_b, ratings, tempos, strategy, "R64", sigma70, spread_a, spread_b, rng, r64_odds
            )
            rows.append(
                _pick_row(
                    strategy,
                    "R64",
                    region,
                    game_index,
                    team_a,
                    team_b,
                    winner,
                    base_p_a,
                    adjusted_p_a,
                )
            )
            current_teams.append(winner)

        current_teams, new_rows = _run_round_pairs(
            strategy, "R32", region, current_teams, ratings, tempos, sigma70, spread_a, spread_b, rng
        )
        rows.extend(new_rows)
        current_teams, new_rows = _run_round_pairs(
            strategy, "S16", region, current_teams, ratings, tempos, sigma70, spread_a, spread_b, rng
        )
        rows.extend(new_rows)
        current_teams, new_rows = _run_round_pairs(
            strategy, "E8", region, current_teams, ratings, tempos, sigma70, spread_a, spread_b, rng
        )
        rows.extend(new_rows)
        champs_by_region[region] = current_teams[0]

    f4_region_order = ["East", "South", "Midwest", "West"]
    regional_champs = [champs_by_region[r] for r in f4_region_order if r in champs_by_region]

    ff_winners, ff_rows = _run_round_pairs(
        strategy, "F4", "FinalFour", regional_champs, ratings, tempos, sigma70, spread_a, spread_b, rng
    )
    rows.extend(ff_rows)
    title_winner, ncg_rows = _run_round_pairs(
        strategy, "NCG", "Final", ff_winners, ratings, tempos, sigma70, spread_a, spread_b, rng
    )
    rows.extend(ncg_rows)

    if title_winner:
        rows.append(
            {
                "Strategy": strategy,
                "Round": "Champ",
                "Region": "Final",
                "GameIndex": 1,
                "TeamA": title_winner[0],
                "TeamB": "",
                "Pick": title_winner[0],
                "TeamA_WinProb_Base": 1.0,
                "TeamA_WinProb_Adjusted": 1.0,
            }
        )

    return rows


def _run_round_pairs(
    strategy: str,
    round_name: str,
    region: str,
    teams: list[str],
    ratings: dict[str, float],
    tempos: dict[str, float],
    sigma70: float,
    spread_a: float,
    spread_b: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[dict[str, str | int | float]]]:
    winners: list[str] = []
    rows: list[dict[str, str | int | float]] = []
    for i in range(0, len(teams), 2):
        team_a, team_b = teams[i], teams[i + 1]
        winner, base_p_a, adjusted_p_a = _simulate_game_with_strategy(
            team_a,
            team_b,
            ratings,
            tempos,
            strategy,
            round_name,
            sigma70,
            spread_a,
            spread_b,
            rng,
        )
        winners.append(winner)
        rows.append(
            _pick_row(
                strategy,
                round_name,
                region,
                (i // 2) + 1,
                team_a,
                team_b,
                winner,
                base_p_a,
                adjusted_p_a,
            )
        )
    return winners, rows


def _pick_row(
    strategy: str,
    round_name: str,
    region: str,
    game_index: int,
    team_a: str,
    team_b: str,
    winner: str,
    base_p_a: float,
    adjusted_p_a: float,
) -> dict[str, str | int | float]:
    return {
        "Strategy": strategy,
        "Round": round_name,
        "Region": region,
        "GameIndex": game_index,
        "TeamA": team_a,
        "TeamB": team_b,
        "Pick": winner,
        "TeamA_WinProb_Base": round(base_p_a, 4),
        "TeamA_WinProb_Adjusted": round(adjusted_p_a, 4),
    }


def _build_advancement_summary(advancement: dict[str, defaultdict], n_sims: int) -> pd.DataFrame:
    rows = []
    for team, counts in advancement.items():
        rows.append(
            {
                "Team": team,
                "Make_R32": counts["R32"] / n_sims,
                "Make_S16": counts["S16"] / n_sims,
                "Make_E8": counts["E8"] / n_sims,
                "Make_F4": counts["F4"] / n_sims,
                "Make_NCG": counts["NCG"] / n_sims,
                "Win_Title": counts["Champ"] / n_sims,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("Win_Title", ascending=False).reset_index(drop=True)


def _build_top_paths(path_counter: Counter[tuple[str, ...]], n_sims: int, top_n: int = 25) -> pd.DataFrame:
    rows = []
    for idx, (path, count) in enumerate(path_counter.most_common(top_n), start=1):
        rows.append(
            {
                "Rank": idx,
                "Likelihood": count / n_sims,
                "BracketPath": " | ".join(path),
            }
        )
    return pd.DataFrame(rows)
