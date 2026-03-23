"""Evaluate IndividualEquity for multiple bracket sets against the same opponent pool.

Evaluates:
  - Optimized brackets (from optimized_picks_upset.csv)
  - Competitor brackets (from competitor_picks_full_names.csv)
  - Deterministic ESPN BPI bracket  (always pick higher BPI-rated team)
  - Deterministic KenPom bracket    (always pick higher KenPom-rated team)

For each bracket, IndividualEquity = P(bracket score > max opponent score)
is computed separately for each distinct pool size from the original run.

Output: output/bracket_equity_comparison.csv
  Columns: Label, Group, Champion, IndividualEquity_{poolsize}, ...
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from ncaa_tourney.simulation import (
    _simulate_bracket_rows,
    _safe_plus_bracket_rows,
    _score_picks,
    _sort_region_games,
    _build_tempo_map,
    _build_seed_map,
    _normalize_strategy_mix,
    SPREAD_TO_Z_A,
    SPREAD_TO_Z_B,
    SIGMA_70,
)
from ncaa_tourney.power_model import simulate_forward_bracket

# ── Configuration ─────────────────────────────────────────────────────────────
# Match the parameters from the actual optimize-picks run.
TEAMS_PATH          = ROOT / "data/processed/teams.csv"
ESPN_RATINGS_PATH   = ROOT / "output/source_link_report_espn.csv"
KENPOM_RATINGS_PATH = ROOT / "output/source_link_report.csv"
GAMES_PATH          = ROOT / "data/processed/round1_games.csv"
R64_ODDS_PATH       = ROOT / "data/raw/round1_game_odds.csv"
POWER_RATINGS_PATH  = ROOT / "output/power_ratings.csv"
OPTIMIZED_PATH      = ROOT / "output/optimized_picks_upset.csv"
COMPETITOR_PATH     = ROOT / "data/processed/competitor_picks_full_names.csv"
OUT_PATH            = ROOT / "output/bracket_equity_comparison.csv"

# Pool sizes from the original run (distinct sizes for evaluation)
POOL_SIZES   = [11, 20, 122]
N_OUTCOMES   = 2500
SEED         = 99   # fresh seed — independent of the optimization run

ROUND_POINTS = {"R64": 1, "R32": 2, "S16": 4, "E8": 8, "F4": 16, "NCG": 32}

OPPONENT_MIX = {"safe": 0.30, "safe_seeded": 0.40, "balanced": 0.10,
                "upset_heavy": 0.10, "safe_plus": 0.10}

# Always include one deterministic (KenPom/teams.csv) bracket as a fixed opponent
# in every pool. Remaining ps-2 slots are drawn from OPPONENT_MIX.
INCLUDE_DET_OPPONENT = True

# F4 region pairing (must match simulation.py _F4_REGION_ORDER)
_F4_REGION_ORDER = ["East", "South", "Midwest", "West"]


# ── Weight vector (63 picks in simulation order) ───────────────────────────────
def _make_weight_vector() -> list[int]:
    """Build weight_vector in the same order as _simulate_bracket_rows picks."""
    wv = []
    for _ in range(4):   # 4 regions, sorted alphabetically
        wv += [1] * 8    # R64
        wv += [2] * 4    # R32
        wv += [4] * 2    # S16
        wv += [8] * 1    # E8
    wv += [16] * 2       # F4
    wv += [32] * 1       # NCG
    return wv            # 63 items total


# ── Load r64 odds ──────────────────────────────────────────────────────────────
def _load_r64_odds(path: Path) -> dict[frozenset[str], tuple[str, float]] | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    result: dict[frozenset[str], tuple[str, float]] = {}
    for row in frame.itertuples(index=False):
        key: frozenset[str] = frozenset({str(row.Favorite), str(row.Underdog)})
        result[key] = (str(row.Favorite), float(row.Probability))
    return result


# ── Load power ratings ─────────────────────────────────────────────────────────
def _load_power_ratings(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return dict(zip(df["Team"], df["Value"].astype(float)))


# ── Parse competitor brackets ──────────────────────────────────────────────────
def _parse_competitor_brackets(path: Path) -> list[tuple[str, list[str], str]]:
    """Return list of (label, picks_63, champion) for each competitor bracket."""
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    results = []
    for b_idx, block in enumerate(blocks, start=1):
        rows_raw = [line for line in block.splitlines() if line.strip()]
        # Skip header rows
        rows_raw = [r for r in rows_raw if not r.lower().startswith("round,")]
        picks_map: dict[tuple[str, str, int], str] = {}
        champion = ""
        for line in rows_raw:
            parts = line.split(",", 5)
            if len(parts) < 6:
                continue
            rnd, region, game_idx, _, _, pick = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
            picks_map[(rnd.strip(), region.strip(), int(game_idx.strip()))] = pick.strip()
            if rnd.strip() == "Champ":
                champion = pick.strip()
        picks = _extract_picks_ordered(picks_map)
        results.append((f"competitor_{b_idx}", picks, champion))
    return results


def _extract_picks_ordered(picks_map: dict[tuple[str, str, int], str]) -> list[str]:
    """Extract 63 picks in simulation order from a (Round, Region, GameIndex)->Pick map."""
    picks = []
    # Regions sorted alphabetically (matches sorted(region_games.keys()))
    region_rounds = [("R64", 8), ("R32", 4), ("S16", 2), ("E8", 1)]
    seen_regions = {reg for (rnd, reg, _) in picks_map if rnd in ("R64", "R32", "S16", "E8")}
    for region in sorted(seen_regions):
        for rnd, n_games in region_rounds:
            for g in range(1, n_games + 1):
                p = picks_map.get((rnd, region, g))
                if p is not None:
                    picks.append(p)
    # F4 — Region may be "Final Four" (competitor) or "FinalFour" (optimized)
    for g in (1, 2):
        p = picks_map.get(("F4", "Final Four", g)) or picks_map.get(("F4", "FinalFour", g))
        if p is not None:
            picks.append(p)
    # NCG — Region may be "Championship" (competitor) or "Final" (optimized)
    p = (picks_map.get(("NCG", "Championship", 1))
         or picks_map.get(("NCG", "Final", 1)))
    if p is not None:
        picks.append(p)
    return picks


# ── Parse optimized brackets ───────────────────────────────────────────────────
def _parse_optimized_brackets(path: Path) -> list[tuple[str, list[str], str, int]]:
    """Return list of (label, picks_63, champion, pool_size) for each entry rank."""
    df = pd.read_csv(path)
    results = []
    for entry_rank in sorted(df["EntryRank"].unique()):
        sub = df[df["EntryRank"] == entry_rank]
        champ_row = sub[sub["Round"] == "Champ"]
        champion = str(champ_row["Pick"].iloc[0]) if len(champ_row) > 0 else ""
        pool_size = int(sub["PoolSize"].iloc[0])
        strategy = str(sub["CandidateStrategy"].iloc[0])
        scored = sub[sub["Round"] != "Champ"].copy()
        picks_map: dict[tuple[str, str, int], str] = {}
        for _, row in scored.iterrows():
            picks_map[(str(row["Round"]), str(row["Region"]), int(row["GameIndex"]))] = str(row["Pick"])
        picks = _extract_picks_ordered(picks_map)
        results.append((f"optimized_{entry_rank}_{strategy}", picks, champion, pool_size))
    return results


# ── Build deterministic bracket ────────────────────────────────────────────────
def _deterministic_picks(
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
    label: str,
) -> tuple[str, list[str], str]:
    """Always pick the higher-rated team. Returns (label, picks_63, champion)."""
    def pick_higher(a: str, b: str) -> str:
        return a if ratings.get(a, 0.0) >= ratings.get(b, 0.0) else b

    picks: list[str] = []
    region_winners: dict[str, str] = {}

    for region in sorted(region_games.keys()):
        games = region_games[region]
        r64 = [pick_higher(a, b) for a, b in games]
        picks.extend(r64)
        r32 = [pick_higher(r64[i], r64[i + 1]) for i in range(0, 8, 2)]
        picks.extend(r32)
        s16 = [pick_higher(r32[i], r32[i + 1]) for i in range(0, 4, 2)]
        picks.extend(s16)
        e8 = pick_higher(s16[0], s16[1])
        picks.append(e8)
        region_winners[region] = e8

    champs = [region_winners[r] for r in _F4_REGION_ORDER if r in region_winners]
    f4_0 = pick_higher(champs[0], champs[1])
    f4_1 = pick_higher(champs[2], champs[3])
    picks.extend([f4_0, f4_1])
    ncg = pick_higher(f4_0, f4_1)
    picks.append(ncg)

    return label, picks, ncg


# ── Main evaluation ────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data…")
    teams_df   = pd.read_csv(TEAMS_PATH)
    espn_df    = pd.read_csv(ESPN_RATINGS_PATH)
    kenpom_df  = pd.read_csv(KENPOM_RATINGS_PATH)
    games_df   = pd.read_csv(GAMES_PATH)

    ratings  = dict(zip(teams_df["Team"], teams_df["Rating"].astype(float)))
    tempos   = _build_tempo_map(teams_df)
    seeds    = _build_seed_map(games_df)
    region_games = _sort_region_games(games_df)
    regions  = sorted(region_games.keys())

    espn_ratings   = dict(zip(espn_df["Team"],   espn_df["Rating"].astype(float)))
    kenpom_ratings = dict(zip(kenpom_df["Team"], kenpom_df["Rating"].astype(float)))

    r64_odds         = _load_r64_odds(R64_ODDS_PATH)
    power_ratings    = _load_power_ratings(POWER_RATINGS_PATH)

    weight_vector = _make_weight_vector()
    assert len(weight_vector) == 63, f"Expected 63 weights, got {len(weight_vector)}"

    opponent_strategy_names = ["safe", "safe_seeded", "balanced", "upset_heavy", "safe_plus"]
    opponent_weights = _normalize_strategy_mix(OPPONENT_MIX, opponent_strategy_names)

    # ── Collect all brackets ─────────────────────────────────────────────────
    all_brackets: list[tuple[str, str, list[str]]] = []  # (label, group, picks)

    opt_brackets = _parse_optimized_brackets(OPTIMIZED_PATH)
    for label, picks, champion, pool_size in opt_brackets:
        all_brackets.append((label, "optimized", picks))
        print(f"  optimized: {label} | champion={champion} | pool_size={pool_size} | picks={len(picks)}")

    comp_brackets = _parse_competitor_brackets(COMPETITOR_PATH)
    for label, picks, champion in comp_brackets:
        all_brackets.append((label, "competitor", picks))
        print(f"  competitor: {label} | champion={champion} | picks={len(picks)}")

    _, det_espn_picks, det_espn_champ = _deterministic_picks(region_games, espn_ratings, "det_espn_bpi")
    _, det_kenpom_picks, det_kenpom_champ = _deterministic_picks(region_games, kenpom_ratings, "det_kenpom")
    # Fixed opponent uses the same KenPom ratings as the simulation (teams.csv)
    _, det_sim_picks, det_sim_champ = _deterministic_picks(region_games, ratings, "det_sim")
    all_brackets.append(("det_espn_bpi", "deterministic", det_espn_picks))
    all_brackets.append(("det_kenpom",   "deterministic", det_kenpom_picks))
    print(f"  deterministic espn_bpi: champion={det_espn_champ} | picks={len(det_espn_picks)}")
    print(f"  deterministic kenpom:   champion={det_kenpom_champ} | picks={len(det_kenpom_picks)}")
    print(f"  deterministic sim (fixed opp): champion={det_sim_champ} | picks={len(det_sim_picks)}")

    n_brackets = len(all_brackets)
    print(f"\nTotal brackets to evaluate: {n_brackets}")
    print(f"Pool sizes: {POOL_SIZES}  |  N_outcomes: {N_OUTCOMES}  |  Seed: {SEED}\n")

    # ── Simulate ──────────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)

    # bracket_scores[b_idx, i] = score of bracket b in outcome i
    bracket_scores = np.zeros((n_brackets, N_OUTCOMES), dtype=np.int32)

    # opp_max[pool_idx, i] = max opponent score in outcome i for pool size POOL_SIZES[pool_idx]
    opp_max   = np.zeros((len(POOL_SIZES), N_OUTCOMES), dtype=np.int32)
    opp_ties  = np.zeros((len(POOL_SIZES), N_OUTCOMES), dtype=np.int32)

    def _sim_opponent(truth_picks: list[str]) -> int:
        opp_strategy = str(rng.choice(opponent_strategy_names, p=opponent_weights))
        if opp_strategy == "safe_seeded" and power_ratings is not None:
            opp_picks = simulate_forward_bracket(power_ratings, region_games, rng)
        elif opp_strategy == "safe_plus":
            _, opp_picks, _ = _safe_plus_bracket_rows(
                regions, region_games, ratings, tempos, seeds,
                SIGMA_70, SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng, None,
            )
        else:
            _, opp_picks, _ = _simulate_bracket_rows(
                regions, region_games, ratings, tempos, seeds,
                SIGMA_70, SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
                strategy=opp_strategy, strategy_label=opp_strategy,
                seed_popularity=None, r64_odds=None,
            )
        return _score_picks(opp_picks, truth_picks, weight_vector)

    for i in range(N_OUTCOMES):
        if (i + 1) % 250 == 0:
            print(f"  outcome {i + 1}/{N_OUTCOMES}…", flush=True)

        # Simulate true tournament outcome
        _, truth_picks, _ = _simulate_bracket_rows(
            regions, region_games, ratings, tempos, seeds,
            SIGMA_70, SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
            strategy=None, strategy_label="truth",
            seed_popularity=None, r64_odds=r64_odds,
        )

        # Score every bracket against this outcome
        for b_idx, (label, group, picks) in enumerate(all_brackets):
            bracket_scores[b_idx, i] = _score_picks(picks, truth_picks, weight_vector)

        # Simulate opponents for each pool size
        det_score = _score_picks(det_sim_picks, truth_picks, weight_vector) if INCLUDE_DET_OPPONENT else None
        for k, ps in enumerate(POOL_SIZES):
            n_random = (ps - 2) if INCLUDE_DET_OPPONENT else (ps - 1)
            opp_scores = [_sim_opponent(truth_picks) for _ in range(n_random)]
            if det_score is not None:
                opp_scores.append(det_score)
            opp_max[k, i] = max(opp_scores)
            opp_ties[k, i] = sum(1 for s in opp_scores if s == opp_max[k, i])

    # ── Compute IndividualEquity per (bracket, pool_size) ─────────────────────
    print("\nComputing equity metrics…")
    result_rows = []
    for b_idx, (label, group, picks) in enumerate(all_brackets):
        bscores = bracket_scores[b_idx]
        row: dict[str, object] = {"Label": label, "Group": group}

        for k, ps in enumerate(POOL_SIZES):
            wins  = int(np.sum(bscores > opp_max[k]))
            ties  = np.where(bscores == opp_max[k])[0]
            eq    = (wins + sum(1.0 / (opp_ties[k, t] + 1) for t in ties)) / N_OUTCOMES
            row[f"IndividualEquity_pool{ps}"] = round(eq, 6)

        result_rows.append(row)

    out_df = pd.DataFrame(result_rows)

    # Sort: optimized first (by entry rank), then competitors, then deterministic
    group_order = {"optimized": 0, "competitor": 1, "deterministic": 2}
    out_df["_sort"] = out_df["Group"].map(group_order)
    out_df = out_df.sort_values(["_sort", "Label"]).drop(columns="_sort").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(out_df)} rows to {OUT_PATH}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
