"""Characterize competitor brackets: upset rates, log-likelihood under each strategy,
champion/F4 seeds, pairwise agreement, and a mixture-model opponent mix recommendation.

Strategies evaluated: safe, safe_seeded, balanced, upset_heavy, chalk_plus.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize  # used by mixture model

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ncaa_tourney.simulation import (
    _build_tempo_map,
    _build_seed_map,
    _sort_region_games,
    _simulate_bracket_rows,
    _safe_plus_bracket_rows,
    win_probability,
    STRATEGY_RANDOMNESS,
    SPREAD_TO_Z_A,
    SPREAD_TO_Z_B,
)
from ncaa_tourney.power_model import load_params, simulate_forward_bracket

TEAMS_CSV  = REPO_ROOT / "data/processed/teams.csv"
GAMES_CSV  = REPO_ROOT / "data/processed/round1_games.csv"
COMP_CSV   = REPO_ROOT / "data/processed/competitor_picks_full_names.csv"
POWER_CSV  = REPO_ROOT / "output/power_ratings.csv"

ROUND_ORDER      = ["R64", "R32", "S16", "E8", "F4", "NCG"]
ROUND_IDX        = {r: i for i, r in enumerate(ROUND_ORDER)}
STRATEGIES       = ["safe", "safe_seeded", "balanced", "upset_heavy", "chalk_plus"]
CHALK_DETERM     = 0.6   # chalk_plus deterministic threshold
N_EXPECTED_SIMS  = 5_000
N_CAL_SIMS       = 2_000  # sims per bisection step during sharpening calibration

# Down-weight late rounds (small sample, high path-dependency) in calibration loss
CAL_ROUND_WEIGHTS = {"R64": 1.0, "R32": 0.8, "S16": 0.6, "E8": 0.4, "F4": 0.3, "NCG": 0.2}

SIMULATION_PY = REPO_ROOT / "ncaa_tourney/simulation.py"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_blocks(path: Path) -> list[pd.DataFrame]:
    text = path.read_text(encoding="utf-8")
    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    dfs = []
    for block in raw_blocks:
        lines = block.splitlines()
        if lines[0].lower().startswith("round,"):
            df = pd.DataFrame(
                [l.split(",") for l in lines[1:]],
                columns=lines[0].split(","),
            )
        else:
            df = pd.DataFrame(
                [l.split(",") for l in lines],
                columns=["Round", "Region", "GameIndex", "TeamA", "TeamB", "Pick"],
            )
        df = df[df["Round"].isin(ROUND_ORDER + ["Champ"])].copy()
        df["GameIndex"] = pd.to_numeric(df["GameIndex"], errors="coerce")
        dfs.append(df)
    return dfs


# ---------------------------------------------------------------------------
# Log-likelihood under each strategy
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def pick_log_likelihood(
    df: pd.DataFrame,
    ratings: dict[str, float],
    tempos: dict[str, float],
    strategy: str,
    theta_dict: dict[str, float] | None = None,
    alpha: np.ndarray | None = None,
) -> float:
    scored = df[df["Round"].isin(ROUND_ORDER)]
    log_prob = 0.0

    for _, row in scored.iterrows():
        ta, tb, pick = str(row["TeamA"]), str(row["TeamB"]), str(row["Pick"])
        rnd = str(row["Round"])

        if strategy == "safe_seeded":
            # Analytical power model: sigmoid(alpha[r] * sharpening[r] * (theta_A - theta_B))
            # Uses SAFE_SEEDED_SHARPENING so LL reflects the calibrated strategy.
            from ncaa_tourney.simulation import SAFE_SEEDED_SHARPENING
            r = ROUND_IDX.get(rnd, 0)
            a_val = float(alpha[r]) * float(SAFE_SEEDED_SHARPENING[r]) if alpha is not None else 1.0
            th_a = (theta_dict or {}).get(ta, 0.0)
            th_b = (theta_dict or {}).get(tb, 0.0)
            p_a = _sigmoid(a_val * (th_a - th_b))

        elif strategy == "chalk_plus":
            # determ_thresh fraction of games: deterministic (fav gets 1.0)
            # remaining: safe (base probability)
            base_p_a = win_probability(
                ratings.get(ta, 0.0), ratings.get(tb, 0.0),
                tempos.get(ta, 70.0), tempos.get(tb, 70.0),
                spread_a=SPREAD_TO_Z_A, spread_b=SPREAD_TO_Z_B,
            )
            fav_is_a = base_p_a >= 0.5
            if pick == ta:
                p_a = CHALK_DETERM * (1.0 if fav_is_a else 0.0) + (1.0 - CHALK_DETERM) * base_p_a
            else:
                p_a = CHALK_DETERM * (0.0 if fav_is_a else 1.0) + (1.0 - CHALK_DETERM) * base_p_a
            # p_a here is already P(pick), not P(TeamA wins)
            log_prob += math.log(max(p_a, 1e-9))
            continue

        else:
            # safe, balanced, upset_heavy
            base_p_a = win_probability(
                ratings.get(ta, 0.0), ratings.get(tb, 0.0),
                tempos.get(ta, 70.0), tempos.get(tb, 70.0),
                spread_a=SPREAD_TO_Z_A, spread_b=SPREAD_TO_Z_B,
            )
            randomness = STRATEGY_RANDOMNESS[strategy].get(rnd, 0.0)
            p_fav = max(base_p_a, 1.0 - base_p_a)
            p_fav_adj = (1.0 - randomness) * p_fav + randomness * 0.5
            p_a = p_fav_adj if base_p_a >= 0.5 else 1.0 - p_fav_adj
            p_a = float(np.clip(p_a, 0.01, 0.99))

        p_pick = p_a if pick == ta else 1.0 - p_a
        log_prob += math.log(max(p_pick, 1e-9))

    return log_prob


# ---------------------------------------------------------------------------
# Upset rates from observed brackets
# ---------------------------------------------------------------------------

def upset_rates(df: pd.DataFrame, ratings: dict[str, float]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for rnd in ROUND_ORDER:
        rows = df[df["Round"] == rnd]
        if rows.empty:
            rates[rnd] = float("nan")
            continue
        n_upset = sum(
            1 for _, row in rows.iterrows()
            if ratings.get(str(row["Pick"]), 0.0) < ratings.get(
                str(row["TeamB"]) if str(row["Pick"]) == str(row["TeamA"]) else str(row["TeamA"]), 0.0
            )
        )
        rates[rnd] = n_upset / len(rows)
    return rates


# ---------------------------------------------------------------------------
# Expected upset rates under each strategy via simulation
# ---------------------------------------------------------------------------

def expected_upset_rates(
    ratings: dict[str, float],
    tempos: dict[str, float],
    seeds: dict[str, int],
    region_games: dict[str, list[tuple[str, str]]],
    regions: list[str],
    power_params: dict[str, float],
    n_sims: int = N_EXPECTED_SIMS,
) -> dict[str, dict[str, float]]:
    """Simulate N brackets under each strategy; track upset rate by round."""
    rng = np.random.default_rng(0)
    results: dict[str, dict[str, list[int]]] = {
        s: {r: [] for r in ROUND_ORDER} for s in STRATEGIES
    }

    for _ in range(n_sims):
        for strategy in STRATEGIES:
            if strategy == "safe_seeded":
                picks_list = simulate_forward_bracket(power_params, region_games, rng,
                                                      round_sharpening=None)
                for rnd, flags in _picks_to_upset_flags(picks_list, region_games, ratings).items():
                    results[strategy][rnd].extend(flags)
            elif strategy == "chalk_plus":
                rows, _, _ = _simulate_bracket_rows(
                    regions, region_games, ratings, tempos, seeds,
                    SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
                    strategy=None, strategy_label="chalk_plus", determ_thresh=CHALK_DETERM,
                )
                for row in rows:
                    rnd = str(row["Round"])
                    if rnd not in ROUND_ORDER:
                        continue
                    ta, tb, pick = str(row["TeamA"]), str(row["TeamB"]), str(row["Pick"])
                    results[strategy][rnd].append(
                        int(ratings.get(pick, 0.0) < ratings.get(tb if pick == ta else ta, 0.0))
                    )
            else:
                rows, _, _ = _simulate_bracket_rows(
                    regions, region_games, ratings, tempos, seeds,
                    SPREAD_TO_Z_A, SPREAD_TO_Z_B, rng,
                    strategy=strategy, strategy_label=strategy,
                )
                for row in rows:
                    rnd = str(row["Round"])
                    if rnd not in ROUND_ORDER:
                        continue
                    ta, tb, pick = str(row["TeamA"]), str(row["TeamB"]), str(row["Pick"])
                    results[strategy][rnd].append(
                        int(ratings.get(pick, 0.0) < ratings.get(tb if pick == ta else ta, 0.0))
                    )

    return {
        s: {r: float(np.mean(v)) if v else float("nan") for r, v in rnd_map.items()}
        for s, rnd_map in results.items()
    }


# ---------------------------------------------------------------------------
# Calibrate safe_seeded sharpening to match observed upset rates
# ---------------------------------------------------------------------------

def _picks_to_upset_flags(
    picks: list[str],
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
) -> dict[str, list[int]]:
    """Given the 63-pick output of simulate_forward_bracket, return per-round upset flags.

    Reconstructs matchups directly from the pick order (no extra sims needed).
    Pick order (matches _play_bracket output):
      Per region (sorted alpha): R64×8, R32×4, S16×2, E8×1  → 15 picks each
      Then F4×2, NCG×1
    F4 pairings follow _F4_REGION_ORDER = [East, South, Midwest, West]:
      game 0: region_sorted[0] (East) vs region_sorted[2] (South)
      game 1: region_sorted[1] (Midwest) vs region_sorted[3] (West)
    """
    flags: dict[str, list[int]] = {r: [] for r in ROUND_ORDER}
    regions = sorted(region_games.keys())  # alphabetical: East, Midwest, South, West

    pos = 0
    e8_winners: list[str] = []

    for region in regions:
        games = region_games[region]
        r64 = picks[pos: pos + 8]; pos += 8
        r32 = picks[pos: pos + 4]; pos += 4
        s16 = picks[pos: pos + 2]; pos += 2
        e8  = picks[pos];          pos += 1

        for i, (ta, tb) in enumerate(games):
            fav = ta if ratings.get(ta, 0.0) >= ratings.get(tb, 0.0) else tb
            flags["R64"].append(int(r64[i] != fav))

        for i in range(4):
            ta, tb = r64[2 * i], r64[2 * i + 1]
            fav = ta if ratings.get(ta, 0.0) >= ratings.get(tb, 0.0) else tb
            flags["R32"].append(int(r32[i] != fav))

        for i in range(2):
            ta, tb = r32[2 * i], r32[2 * i + 1]
            fav = ta if ratings.get(ta, 0.0) >= ratings.get(tb, 0.0) else tb
            flags["S16"].append(int(s16[i] != fav))

        ta, tb = s16[0], s16[1]
        fav = ta if ratings.get(ta, 0.0) >= ratings.get(tb, 0.0) else tb
        flags["E8"].append(int(e8 != fav))
        e8_winners.append(e8)

    # F4: East(0) vs South(2), Midwest(1) vs West(3)  (index into e8_winners)
    f4_pairs = [(0, 2), (1, 3)]
    f4_winners: list[str] = []
    for i, (ia, ib) in enumerate(f4_pairs):
        ta, tb = e8_winners[ia], e8_winners[ib]
        winner = picks[pos + i]
        fav = ta if ratings.get(ta, 0.0) >= ratings.get(tb, 0.0) else tb
        flags["F4"].append(int(winner != fav))
        f4_winners.append(winner)
    pos += 2

    ta, tb = f4_winners[0], f4_winners[1]
    fav = ta if ratings.get(ta, 0.0) >= ratings.get(tb, 0.0) else tb
    flags["NCG"].append(int(picks[pos] != fav))

    return flags


def _upset_rates_sharpened(
    sharpening: np.ndarray,
    power_params: dict[str, float],
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
    rng: np.random.Generator,
    n_sims: int,
) -> dict[str, float]:
    """Expected upset rates under safe_seeded with given per-round sharpening."""
    counts: dict[str, list[float]] = {r: [] for r in ROUND_ORDER}
    for _ in range(n_sims):
        picks = simulate_forward_bracket(power_params, region_games, rng,
                                         round_sharpening=sharpening)
        for rnd, flags in _picks_to_upset_flags(picks, region_games, ratings).items():
            counts[rnd].extend(flags)
    return {r: float(np.mean(v)) if v else float("nan") for r, v in counts.items()}


def calibrate_sharpening(
    observed: dict[str, float],
    power_params: dict[str, float],
    region_games: dict[str, list[tuple[str, str]]],
    ratings: dict[str, float],
) -> np.ndarray:
    """Round-by-round bisection: find per-round sharpening matching observed upset rates.

    Calibrates rounds sequentially R64→NCG, using already-fitted earlier-round
    sharpenings when simulating later rounds (correct path dependency).
    10 bisection steps × N_CAL_SIMS sims = ~120k sims total, runs in ~30s.
    Sharpening > 1 → more chalk; < 1 → more upsets.
    """
    rng = np.random.default_rng(7)
    sharpening = np.ones(6)

    for r_idx, rnd in enumerate(ROUND_ORDER):
        target = observed.get(rnd, float("nan"))
        if np.isnan(target):
            continue

        def rate_at(s: float, r_idx: int = r_idx) -> float:
            test = sharpening.copy()
            test[r_idx] = s
            rates = _upset_rates_sharpened(test, power_params, region_games, ratings, rng, N_CAL_SIMS)
            return rates[rnd]

        # Higher sharpening → fewer upsets; lower → more upsets.
        # Search in [0.15, 6.0]; if uncalibrated already matches, bisect converges quickly.
        lo, hi = 0.15, 6.0
        for _ in range(10):
            mid = (lo + hi) / 2.0
            mid_rate = rate_at(mid)
            if mid_rate > target:
                lo = mid   # too many upsets → need more sharpening
            else:
                hi = mid   # too few upsets → need less sharpening

        sharpening[r_idx] = (lo + hi) / 2.0
        print(f"    {rnd}: sharpening={sharpening[r_idx]:.3f}  "
              f"(target={target:.1%}, achieved≈{rate_at(sharpening[r_idx]):.1%})")

    return sharpening


# ---------------------------------------------------------------------------
# Mixture model: find weights that best explain the 16 observed brackets
# ---------------------------------------------------------------------------

def fit_mixture(ll_matrix: np.ndarray) -> np.ndarray:
    """Max-likelihood mixture weights over strategies.
    ll_matrix: (n_brackets, n_strategies) log-likelihoods.
    Returns normalized weights summing to 1.
    """
    n_strats = ll_matrix.shape[1]

    def neg_ll(log_w: np.ndarray) -> float:
        log_weights = log_w - logsumexp(log_w)  # normalize in log space
        # log P(bracket i) = log sum_k w_k * exp(ll[i,k])
        #                   = logsumexp(log_weights + ll[i, :])
        total = sum(logsumexp(log_weights + ll_matrix[i]) for i in range(len(ll_matrix)))
        return -total

    result = minimize(neg_ll, np.zeros(n_strats), method="Nelder-Mead",
                      options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    log_w = result.x - logsumexp(result.x)
    return np.exp(log_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    teams_df = pd.read_csv(TEAMS_CSV)
    games_df = pd.read_csv(GAMES_CSV)
    power_df = pd.read_csv(POWER_CSV)

    ratings      = dict(zip(teams_df["Team"], teams_df["Rating"]))
    tempos       = _build_tempo_map(teams_df)
    seeds        = _build_seed_map(games_df)
    region_games = _sort_region_games(games_df)
    regions      = sorted(region_games.keys())

    power_params = {str(r.Team): float(r.Value) for r in power_df.itertuples(index=False)}
    theta_dict, alpha, _, _ = load_params(power_params)

    blocks = parse_blocks(COMP_CSV)
    n = len(blocks)
    print(f"Parsed {n} competitor brackets\n")

    # ── Per-bracket stats ────────────────────────────────────────────────────
    summary_rows = []
    ll_matrix = np.zeros((n, len(STRATEGIES)))

    for idx, df in enumerate(blocks, start=1):
        champ_row = df[df["Round"] == "Champ"]
        champ = str(champ_row["Pick"].iloc[0]) if not champ_row.empty else "?"
        champ_seed = seeds.get(champ, "?")

        f4_rows = df[df["Round"] == "F4"]
        f4_teams: list[str] = []
        for _, r in f4_rows.iterrows():
            f4_teams += [str(r["TeamA"]), str(r["TeamB"])]
        f4_seeds = sorted(seeds.get(t, 99) for t in f4_teams)

        lls = {}
        for s_idx, s in enumerate(STRATEGIES):
            ll = pick_log_likelihood(df, ratings, tempos, s, theta_dict, alpha)
            lls[s] = ll
            ll_matrix[idx - 1, s_idx] = ll

        best = max(lls, key=lls.__getitem__)
        rates = upset_rates(df, ratings)
        overall = sum(
            sum(
                1 for _, r in df[df["Round"] == rnd].iterrows()
                if ratings.get(str(r["Pick"]), 0.0) < ratings.get(
                    str(r["TeamB"]) if str(r["Pick"]) == str(r["TeamA"]) else str(r["TeamA"]), 0.0
                )
            )
            for rnd in ROUND_ORDER
        ) / max(len(df[df["Round"].isin(ROUND_ORDER)]), 1)

        summary_rows.append({
            "Bracket": idx,
            "Champion": champ,
            "Seed": champ_seed,
            "F4Seeds": str(f4_seeds),
            "BestStrategy": best,
            **{f"LL_{s}": round(lls[s], 2) for s in STRATEGIES},
            **{f"UR_{r}": round(rates[r], 3) if not pd.isna(rates[r]) else float("nan") for r in ROUND_ORDER},
            "OverallUR": round(overall, 3),
        })

    summary = pd.DataFrame(summary_rows)

    # ── Print bracket table ──────────────────────────────────────────────────
    print("=" * 100)
    print(f"{'#':>2}  {'Champion':<28} {'Sd':>2}  {'F4 Seeds':<15} {'Best':>12}  {'OverallUR':>9}")
    print("-" * 100)
    for _, r in summary.iterrows():
        print(f"{int(r['Bracket']):>2}  {r['Champion']:<28} {str(r['Seed']):>2}  "
              f"{r['F4Seeds']:<15} {r['BestStrategy']:>12}  {r['OverallUR']:>9.1%}")

    print("\n--- Log-likelihoods (higher = better fit) ---")
    header = f"{'#':>2}  " + "  ".join(f"{s:>12}" for s in STRATEGIES)
    print(header)
    print("-" * (len(header) + 5))
    for _, r in summary.iterrows():
        vals = "  ".join(f"{r[f'LL_{s}']:>12.1f}" for s in STRATEGIES)
        print(f"{int(r['Bracket']):>2}  {vals}  ← {r['BestStrategy']}")

    # ── Mixture model ────────────────────────────────────────────────────────
    print("\n--- Mixture model: best-fit opponent weights ---")
    weights = fit_mixture(ll_matrix)
    for s, w in zip(STRATEGIES, weights):
        print(f"  {s:<14}: {w:.1%}")

    # Suggested opponent-mix string for CLI
    mix_str = ",".join(f"{w:.2f}" for w in weights)
    print(f"\n  → --opponent-mix {mix_str}")
    print(f"    (order: safe, safe_seeded, balanced, upset_heavy, safe_plus, chalk_plus)")
    print(f"    Note: safe_plus not modeled here; distribute its share between safe and safe_seeded")

    # ── Expected upset rates by strategy vs observed ─────────────────────────
    print(f"\nSimulating expected upset rates ({N_EXPECTED_SIMS:,} brackets per strategy)…")
    expected = expected_upset_rates(
        ratings, tempos, seeds, region_games, regions, power_params
    )

    observed_mean = {r: float(summary[f"UR_{r}"].mean()) for r in ROUND_ORDER}

    print("\n--- Upset rate by round: observed vs expected under each strategy ---")
    col_w = 9
    header2 = f"{'Round':<6}  {'Observed':>{col_w}}" + "".join(f"  {s:>{col_w}}" for s in STRATEGIES)
    print(header2)
    print("-" * len(header2))
    for rnd in ROUND_ORDER:
        obs = observed_mean[rnd]
        exp_vals = "".join(f"  {expected[s][rnd]:>{col_w}.1%}" for s in STRATEGIES)
        print(f"{rnd:<6}  {obs:>{col_w}.1%}{exp_vals}")

    # ── Calibrate safe_seeded sharpening ────────────────────────────────────
    print(f"\nCalibrating safe_seeded sharpening ({N_CAL_SIMS:,} sims/eval, Nelder-Mead)…")
    sharpening = calibrate_sharpening(observed_mean, power_params, region_games, ratings)
    sharp_str = ", ".join(f"{v:.4f}" for v in sharpening)
    print(f"  Fitted sharpening: [{sharp_str}]")
    print(f"  (R64, R32, S16, E8, F4, NCG — >1 = more chalk, <1 = more upsets)")

    # Verify: expected upset rates under calibrated model
    rng_cal = np.random.default_rng(99)
    cal_rates = _upset_rates_sharpened(sharpening, power_params, region_games, ratings, rng_cal, 2_000)
    print("\n--- Upset rate by round: observed vs uncalibrated vs calibrated safe_seeded ---")
    print(f"{'Round':<6}  {'Observed':>9}  {'Uncal':>9}  {'Calibrated':>10}  {'Delta':>7}")
    print("-" * 50)
    for rnd in ROUND_ORDER:
        obs = observed_mean[rnd]
        uncal = expected["safe_seeded"][rnd]
        cal = cal_rates[rnd]
        print(f"{rnd:<6}  {obs:>9.1%}  {uncal:>9.1%}  {cal:>10.1%}  {cal-obs:>+7.1%}")

    print(f"  (Raw fits reported above for reference — SAFE_SEEDED_SHARPENING in simulation.py")
    print(f"   is hand-tuned; this script no longer auto-updates it.)")

    # ── Hard-assignment summary ──────────────────────────────────────────────
    print("\n--- Best-fit strategy counts ---")
    for s in STRATEGIES:
        count = (summary["BestStrategy"] == s).sum()
        print(f"  {s:<14}: {count}/{n}")

    # ── Pairwise agreement ───────────────────────────────────────────────────
    print("\n--- Pairwise pick agreement ---")
    pick_lists = []
    for df in blocks:
        scored = df[df["Round"].isin(ROUND_ORDER)]
        pick_lists.append(list(scored.sort_values(["Round", "Region", "GameIndex"])["Pick"]))

    agree = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if len(pick_lists[i]) == len(pick_lists[j]):
                agree[i, j] = np.mean([a == b for a, b in zip(pick_lists[i], pick_lists[j])])

    header3 = "    " + "".join(f"{i+1:>6}" for i in range(n))
    print(header3)
    for i in range(n):
        row_str = f"{i+1:>3} " + "".join(
            f"{agree[i,j]:>6.1%}" if j > i else "      " for j in range(n)
        )
        print(row_str)

    off_diag = agree[np.triu_indices(n, k=1)]
    print(f"\n  Mean pairwise agreement: {off_diag.mean():.1%}  "
          f"(range {off_diag.min():.1%}–{off_diag.max():.1%})")


if __name__ == "__main__":
    main()
