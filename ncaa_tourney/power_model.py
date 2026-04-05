"""Power ratings model for NCAA bracket simulation.

Model:  P(i beats j in game round r) = sigmoid(alpha[r] * (eff_i - eff_j))

where  eff_i = theta_i + gamma  if team i has previously beaten a higher-rated
opponent in the current bracket path (upset flag), else  eff_i = theta_i.

- 64 team ratings (theta), one fixed at 0 for identifiability
- 6 round-scaling parameters (alpha), one per game round (R64 through NCG)
  A small L2 prior on (alpha - 1) keeps them near 1 absent strong evidence.
- 1 upset-boost scalar (gamma): effective rating bonus carried forward by teams
  that have pulled at least one upset.  Fitted jointly with theta and alpha via
  simulation-based marginals.

Yahoo round-name → game round mapping (critical):
  Yahoo "R32"   = P(advances to R32)   = P(wins R64 game)   → game round 0, alpha[0]
  Yahoo "S16"   = P(advances to S16)   = P(wins R64 + R32)  → game round 1, alpha[1]
  Yahoo "E8"    = P(advances to E8)    = P(wins through S16) → game round 2, alpha[2]
  Yahoo "F4"    = P(advances to F4)    = P(wins through E8)  → game round 3, alpha[3]
  Yahoo "NCG"   = P(advances to NCG)   = P(wins through F4)  → game round 4, alpha[4]
  Yahoo "Champ" = P(wins championship) = P(wins through NCG) → game round 5, alpha[5]

Key functions
-------------
build_bracket_structure(games_df)
    Parse round1_games.csv into opponent pools for each game round.

compute_marginals(theta, structure, alpha=None)
    Analytical DP — valid only when gamma=0.

compute_marginals_sim(params, region_games, rng, n_sims, gamma=None)
    Monte Carlo marginals — handles path-dependent gamma effect.

fit_power_ratings(yahoo_df, games_df, alpha_prior_weight, n_fit_sims)
    Fit theta + alpha + gamma jointly via simulation-based L-BFGS-B.

load_params(params_dict)
    Split a combined params dict into (theta_dict, alpha_array, gamma).

simulate_forward_bracket(params_dict, region_games, rng)
    Forward-simulate a complete bracket; returns picks list (63 entries).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# F4 matchup order — East vs South, Midwest vs West
_F4_REGION_ORDER = ["East", "South", "Midwest", "West"]

# Game rounds in order (column index in adv matrix = index into this list)
_GAME_ROUNDS = ["R64", "R32", "S16", "E8", "F4", "NCG"]

# Yahoo round label → adv column index (= game round index)
# Yahoo "R32" means P(advances to R32) = P(wins R64 game) = column 0
_YAHOO_ROUND_MAP: dict[str, int] = {
    "R32":   0,   # wins R64 game
    "S16":   1,   # wins R32 game
    "E8":    2,   # wins S16 game
    "F4":    3,   # wins E8 game
    "NCG":   4,   # wins F4 game
    "Champ": 5,   # wins NCG game
}

# Sentinel prefix for alpha values stored in the params CSV
_ALPHA_PREFIX = "__alpha_"

# Sentinel key for gamma stored in the params CSV
_GAMMA_KEY = "__gamma__"

# Sentinel key for sigma stored in the params CSV
_SIGMA_KEY = "__sigma__"


def _alpha_key(game_round: str) -> str:
    return f"{_ALPHA_PREFIX}{game_round}__"


@dataclass
class BracketStructure:
    """Precomputed opponent pools for each team at each game round.

    teams[i] is the name of team i.  All pool attributes are parallel lists
    indexed by team index i.

    Pool semantics (teams team i might face in that game round):
      r64_opp  — 1 team  (fixed R64 game opponent)
      r32_pool — 2 teams (winner of partner R64 game)
      s16_pool — 4 teams (winners from partner quad = 2 R64 games)
      e8_pool  — 8 teams (opposite region half = 4 R64 games)
      f4_pool  — 16 teams (paired region per F4 bracket)
      ncg_pool — 32 teams (opposite bracket half)
    """

    teams: list[str]
    r64_opp:  list[int]
    r32_pool: list[list[int]]
    s16_pool: list[list[int]]
    e8_pool:  list[list[int]]
    f4_pool:  list[list[int]]
    ncg_pool: list[list[int]]


def build_bracket_structure(games_df: pd.DataFrame) -> BracketStructure:
    """Build opponent pools from round1_games.csv.

    Region game numbering (0-indexed within each region, sorted by GameId):
      Partner game:  g ^ 1     (0↔1, 2↔3, 4↔5, 6↔7)
      Quad:          g // 2    (0,1→quad 0; 2,3→quad 1; 4,5→quad 2; 6,7→quad 3)
      Partner quad:  q ^ 1     (0↔1, 2↔3)
      Region half:   g // 4    (0-3 → top; 4-7 → bottom)
    F4 pairings: East↔South, Midwest↔West (indices 0,1 and 2,3 in _F4_REGION_ORDER).
    """
    region_games: dict[str, list[tuple[str, str]]] = {}
    for region, grp in games_df.groupby("Region"):
        ordered = grp.sort_values("GameId")
        region_games[str(region)] = [(str(r.TeamA), str(r.TeamB)) for r in ordered.itertuples(index=False)]

    teams: list[str] = []
    team_idx: dict[str, int] = {}
    region_team_indices: dict[str, list[int]] = {}

    for region in sorted(region_games.keys()):
        seen: set[int] = set()
        region_team_indices[region] = []
        for a, b in region_games[region]:
            for t in (a, b):
                if t not in team_idx:
                    team_idx[t] = len(teams)
                    teams.append(t)
                idx = team_idx[t]
                if idx not in seen:
                    region_team_indices[region].append(idx)
                    seen.add(idx)

    n = len(teams)
    r64_opp:  list[int]       = [0] * n
    r32_pool: list[list[int]] = [[] for _ in range(n)]
    s16_pool: list[list[int]] = [[] for _ in range(n)]
    e8_pool:  list[list[int]] = [[] for _ in range(n)]
    f4_pool:  list[list[int]] = [[] for _ in range(n)]
    ncg_pool: list[list[int]] = [[] for _ in range(n)]

    ncg_half: dict[str, int] = {}
    f4_partner: dict[str, str] = {}
    for k in range(0, len(_F4_REGION_ORDER), 2):
        r0, r1 = _F4_REGION_ORDER[k], _F4_REGION_ORDER[k + 1]
        f4_partner[r0] = r1
        f4_partner[r1] = r0
        half_id = k // 2
        ncg_half[r0] = half_id
        ncg_half[r1] = half_id

    teams_by_region: dict[str, list[int]] = region_team_indices

    for region in sorted(region_games.keys()):
        games = region_games[region]

        for g, (a, b) in enumerate(games):
            for t, opp in ((a, b), (b, a)):
                i = team_idx[t]
                j = team_idx[opp]

                r64_opp[i] = j

                # R32 game pool: 2 teams from partner R64 game
                pg = g ^ 1
                r32_pool[i] = [team_idx[x] for x in games[pg]]

                # S16 game pool: 4 teams from partner quad
                quad = g // 2
                partner_quad = quad ^ 1
                pq_start = partner_quad * 2
                s16_pool[i] = [
                    team_idx[x]
                    for gg in range(pq_start, pq_start + 2)
                    for x in games[gg]
                ]

                # E8 game pool: 8 teams from opposite region half
                half = g // 4
                opp_half_start = (1 - half) * 4
                e8_pool[i] = [
                    team_idx[x]
                    for gg in range(opp_half_start, opp_half_start + 4)
                    for x in games[gg]
                ]

                # F4 game pool: 16 teams from paired region
                paired_region = f4_partner.get(region, "")
                f4_pool[i] = list(teams_by_region.get(paired_region, []))

                # NCG game pool: 32 teams from opposite bracket half
                my_half = ncg_half.get(region, 0)
                ncg_pool[i] = [
                    idx
                    for reg, idxs in teams_by_region.items()
                    if ncg_half.get(reg, -1) != my_half
                    for idx in idxs
                ]

    return BracketStructure(
        teams=teams,
        r64_opp=r64_opp,
        r32_pool=r32_pool,
        s16_pool=s16_pool,
        e8_pool=e8_pool,
        f4_pool=f4_pool,
        ncg_pool=ncg_pool,
    )


def compute_marginals(
    theta: np.ndarray,
    structure: BracketStructure,
    alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Compute marginal advancement probabilities via bracket DP.

    P(i beats j in game round r) = sigmoid(alpha[r] * (theta_i - theta_j))

    Parameters
    ----------
    theta : (n,) team ratings
    structure : BracketStructure
    alpha : (6,) round scaling parameters, default all ones

    Returns
    -------
    adv : (n, 6) array
        adv[i, r] = P(team i wins through game round r)
        r=0: R64 game  (= Yahoo "R32" probability)
        r=1: R32 game  (= Yahoo "S16" probability)
        r=2: S16 game  (= Yahoo "E8"  probability)
        r=3: E8 game   (= Yahoo "F4"  probability)
        r=4: F4 game   (= Yahoo "NCG" probability)
        r=5: NCG game  (= Yahoo "Champ" probability)
    """
    if alpha is None:
        alpha = np.ones(6)

    n = len(structure.teams)
    diffs = theta[:, None] - theta[None, :]  # (n, n)
    adv = np.zeros((n, 6))

    # R64 game (column 0): fixed opponent, scaled by alpha[0]
    P0 = 1.0 / (1.0 + np.exp(-alpha[0] * diffs))
    for i in range(n):
        adv[i, 0] = P0[i, structure.r64_opp[i]]

    # Remaining 5 game rounds (columns 1–5)
    pools = [
        structure.r32_pool,   # column 1 (R32 game): pool = R64 partner game
        structure.s16_pool,   # column 2 (S16 game): pool = partner quad
        structure.e8_pool,    # column 3 (E8 game):  pool = opposite region half
        structure.f4_pool,    # column 4 (F4 game):  pool = paired region
        structure.ncg_pool,   # column 5 (NCG game): pool = opposite bracket half
    ]
    for r, pool in enumerate(pools, start=1):
        Pr = 1.0 / (1.0 + np.exp(-alpha[r] * diffs))
        for i in range(n):
            p_win = sum(adv[j, r - 1] * Pr[i, j] for j in pool[i])
            adv[i, r] = adv[i, r - 1] * p_win

    return adv


def fit_power_ratings(
    yahoo_df: pd.DataFrame,
    games_df: pd.DataFrame,
    alpha_prior_weight: float = 0.001,
    n_fit_sims: int = 20000,
    n_fit_sims_theta: int = 2000,
    max_outer_iter: int = 5,
    gamma_tol: float = 0.005,
    sigma: float = 0.0,
) -> dict[str, float]:
    """Fit team ratings (theta), round scalings (alpha), and upset boost (gamma).

    sigma is NOT fit from data — it is a posthoc parameter set by the caller
    based on cross-system rating disagreement (e.g. std of ESPN BPI vs KenPom
    disagreement ≈ 1.1 rating points → sigma ≈ 1.0–1.5).  It is stored in the
    output params dict so that simulate_forward_bracket uses it automatically.

    Iterative coordinate descent:

      Iteration 1 (bootstrap):
        Stage 1 — analytical DP, gamma fixed at 0 (fast exact solution).
        Stage 2 — Brent search over gamma with theta/alpha fixed (n_fit_sims sims).

      Iterations 2+ (refinement, until |Δgamma| < gamma_tol):
        Stage 1 — L-BFGS-B on theta/alpha with gamma fixed, using simulation
                   (n_fit_sims_theta sims, fixed seed, warm-started).
        Stage 2 — Brent search over gamma (n_fit_sims sims).

    Parameters
    ----------
    yahoo_df : DataFrame with columns Team, Round, Probability
    games_df : round1_games.csv DataFrame
    alpha_prior_weight : L2 penalty pulling alpha toward 1.0
    n_fit_sims : sims per stage-2 (gamma) Brent evaluation
    n_fit_sims_theta : sims per stage-1 (theta/alpha) evaluation in iter 2+
    max_outer_iter : maximum coordinate-descent iterations
    gamma_tol : convergence threshold on |Δgamma|
    sigma : posthoc rating noise std (default 0.0 = no noise); stored in output

    Returns
    -------
    params : dict  {team_name: theta, ..., "__alpha_R64__": alpha[0], ...,
                    "__gamma__": gamma, "__sigma__": sigma}
    """
    from scipy.optimize import minimize_scalar
    from ncaa_tourney.rankings import _canonical_team_key, _resolve_alias

    # ------------------------------------------------------------------ #
    # Setup: shared indexing used by both DP and simulation stages         #
    # ------------------------------------------------------------------ #
    structure = build_bracket_structure(games_df)
    n = len(structure.teams)   # structure.teams and sim_teams share traversal order

    region_games: dict[str, list[tuple[str, str]]] = {}
    for region, grp in games_df.groupby("Region"):
        ordered = grp.sort_values("GameId")
        region_games[str(region)] = [(str(r.TeamA), str(r.TeamB)) for r in ordered.itertuples(index=False)]

    # Team index (same order as structure.teams — both use sorted regions)
    team_idx: dict[str, int] = {t: i for i, t in enumerate(structure.teams)}

    canon_to_idx: dict[str, int] = {
        _resolve_alias(_canonical_team_key(t)): i
        for i, t in enumerate(structure.teams)
    }

    obs_ti_list, obs_ri_list, obs_pr_list = [], [], []
    for row in yahoo_df.itertuples(index=False):
        r_idx = _YAHOO_ROUND_MAP.get(str(row.Round))
        if r_idx is None:
            continue
        t_idx = canon_to_idx.get(_resolve_alias(_canonical_team_key(str(row.Team))))
        if t_idx is None:
            continue
        obs_ti_list.append(t_idx)
        obs_ri_list.append(r_idx)
        obs_pr_list.append(float(row.Probability))  # type: ignore[arg-type]

    if not obs_ti_list:
        raise ValueError("No matching observations — check team names and round codes in yahoo_df")

    obs_ti = np.array(obs_ti_list, dtype=np.int32)
    obs_ri = np.array(obs_ri_list, dtype=np.int32)
    obs_pr = np.array(obs_pr_list, dtype=np.float64)

    n_theta_free = n - 1
    theta_free = list(range(1, n))
    bounds = [(None, None)] * n_theta_free + [(0.01, None)] * 6

    def _sim_adv(theta_dict: dict[str, float], alpha: np.ndarray,
                 gamma: float, sigma: float, n_sims: int) -> np.ndarray:
        """Fixed-seed Monte Carlo marginals, shape (n, 6)."""
        counts = np.zeros((n, 6), dtype=np.float64)
        rng = np.random.default_rng(0)
        for _ in range(n_sims):
            track: dict[str, int] = {}
            _play_bracket(theta_dict, alpha, gamma, sigma, region_games, rng, track)
            for team, rnd in track.items():
                if rnd >= 0:
                    counts[team_idx[team], : rnd + 1] += 1.0
        return counts / n_sims

    def _gamma_search(theta_dict: dict[str, float], alpha: np.ndarray,
                      sigma: float, label: str) -> float:
        """Brent search over gamma with theta/alpha/sigma fixed."""
        eval_count = [0]

        def obj(g: float) -> float:
            adv = _sim_adv(theta_dict, alpha, g, sigma, n_fit_sims)
            mse = float(np.mean((adv[obs_ti, obs_ri] - obs_pr) ** 2))
            eval_count[0] += 1
            print(f"    {label} gamma eval {eval_count[0]:2d}  gamma={g:.4f}  MSE={mse:.6f}", flush=True)
            return mse

        res = minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded",
                              options={"xatol": gamma_tol})
        return float(res.x)

    # ------------------------------------------------------------------ #
    # Iteration 1: analytical DP → theta/alpha; Brent → gamma            #
    # ------------------------------------------------------------------ #
    def dp_obj(x: np.ndarray) -> float:
        theta = np.zeros(n)
        theta[theta_free] = x[:n_theta_free]
        alpha = x[n_theta_free:]
        adv = compute_marginals(theta, structure, alpha)
        mse = float(np.mean((adv[obs_ti, obs_ri] - obs_pr) ** 2))
        prior = float(alpha_prior_weight * np.sum((alpha - 1.0) ** 2))
        return mse + prior

    x0 = np.concatenate([np.zeros(n_theta_free), np.ones(6)])
    res_dp = minimize(dp_obj, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 4000, "ftol": 1e-14})

    theta_fit = np.zeros(n)
    theta_fit[theta_free] = res_dp.x[:n_theta_free]
    alpha_fit = res_dp.x[n_theta_free:]
    theta_dict_fit = {structure.teams[i]: float(theta_fit[i]) for i in range(n)}
    x0 = res_dp.x  # warm-start for sim-based stage 1 in subsequent iters

    print(f"  Iter 1 stage 1 (DP)  loss={res_dp.fun:.6f}", flush=True)

    gamma = _gamma_search(theta_dict_fit, alpha_fit, 0.0, "iter 1")
    print(f"  Iter 1 stage 2  gamma={gamma:.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # Iterations 2+: simulation-based theta/alpha; Brent → gamma         #
    # ------------------------------------------------------------------ #
    for outer_iter in range(2, max_outer_iter + 1):
        gamma_prev = gamma

        gamma_fixed = gamma  # capture for closure

        def sim_theta_obj(x: np.ndarray) -> float:
            theta = np.zeros(n)
            theta[theta_free] = x[:n_theta_free]
            alpha = x[n_theta_free:]
            td = {structure.teams[i]: float(theta[i]) for i in range(n)}
            adv = _sim_adv(td, alpha, gamma_fixed, 0.0, n_fit_sims_theta)
            mse = float(np.mean((adv[obs_ti, obs_ri] - obs_pr) ** 2))
            prior = float(alpha_prior_weight * np.sum((alpha - 1.0) ** 2))
            return mse + prior

        res_sim = minimize(sim_theta_obj, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 50, "ftol": 1e-10, "eps": 0.02})
        x0 = res_sim.x
        theta_fit = np.zeros(n)
        theta_fit[theta_free] = res_sim.x[:n_theta_free]
        alpha_fit = res_sim.x[n_theta_free:]
        theta_dict_fit = {structure.teams[i]: float(theta_fit[i]) for i in range(n)}
        print(f"  Iter {outer_iter} stage 1 (sim)  loss={res_sim.fun:.6f}  gamma_fixed={gamma:.4f}", flush=True)

        gamma = _gamma_search(theta_dict_fit, alpha_fit, 0.0, f"iter {outer_iter}")
        print(f"  Iter {outer_iter} stage 2  gamma={gamma_prev:.4f} → {gamma:.4f}", flush=True)

        if abs(gamma - gamma_prev) < gamma_tol:
            print(f"  Converged at iter {outer_iter}", flush=True)
            break

    params: dict[str, float] = {structure.teams[i]: float(theta_fit[i]) for i in range(n)}
    for r, game_round in enumerate(_GAME_ROUNDS):
        params[_alpha_key(game_round)] = float(alpha_fit[r])
    params[_GAMMA_KEY] = gamma
    params[_SIGMA_KEY] = sigma

    return params


def load_params(params: dict[str, float]) -> tuple[dict[str, float], np.ndarray, float, float]:
    """Split a combined params dict into (theta_dict, alpha_array, gamma, sigma).

    Parameters
    ----------
    params : dict returned by fit_power_ratings or loaded from CSV

    Returns
    -------
    theta_dict : {team_name: theta}
    alpha : (6,) array indexed by game round (R64=0 ... NCG=5)
    gamma : upset boost scalar (0.0 if not present in params)
    sigma : rating noise std (0.0 if not present in params)
    """
    theta_dict = {k: v for k, v in params.items() if not k.startswith("__")}
    alpha = np.array([params.get(_alpha_key(r), 1.0) for r in _GAME_ROUNDS])
    gamma = float(params.get(_GAMMA_KEY, 0.0))
    sigma = float(params.get(_SIGMA_KEY, 0.0))
    return theta_dict, alpha, gamma, sigma


def _play_bracket(
    theta_dict: dict[str, float],
    alpha: np.ndarray,
    gamma: float,
    sigma: float,
    region_games: dict[str, list[tuple[str, str]]],
    rng: np.random.Generator,
    track_adv: dict[str, int] | None = None,
    round_sharpening: np.ndarray | None = None,
) -> list[str]:
    """Inner bracket simulation used by both simulate_forward_bracket and compute_marginals_sim.

    Upset rule: a team earns the upset flag when it beats a team whose current
    effective theta is higher.  The flag (and its gamma boost) carries forward
    for every subsequent game that team plays.

    Parameters
    ----------
    track_adv : if provided, maps team_name -> highest round won (0-5).
                Updated in-place.

    Returns
    -------
    picks : 63-element list in the same order as _simulate_bracket_rows.
    """
    upset_flags: dict[str, bool] = {}

    # Per-simulation independent rating perturbations
    if sigma > 0.0:
        noise: dict[str, float] = {t: float(rng.normal(0.0, sigma)) for t in theta_dict}
    else:
        noise = {}

    def eff(t: str) -> float:
        return (theta_dict.get(t, 0.0)
                + noise.get(t, 0.0)
                + (gamma if upset_flags.get(t, False) else 0.0))

    def play(a: str, b: str, r: int) -> str:
        sharp = float(round_sharpening[r]) if round_sharpening is not None else 1.0
        p = 1.0 / (1.0 + math.exp(-alpha[r] * sharp * (eff(a) - eff(b))))
        winner, loser = (a, b) if rng.random() < p else (b, a)
        if gamma != 0.0 and eff(winner) < eff(loser):
            upset_flags[winner] = True
        return winner

    picks: list[str] = []
    region_winners: dict[str, str] = {}

    for region in sorted(region_games.keys()):
        games = region_games[region]

        r64 = [play(a, b, 0) for a, b in games]
        picks.extend(r64)
        if track_adv is not None:
            for w in r64:
                track_adv[w] = max(track_adv.get(w, -1), 0)

        r32 = [play(r64[i], r64[i + 1], 1) for i in range(0, 8, 2)]
        picks.extend(r32)
        if track_adv is not None:
            for w in r32:
                track_adv[w] = max(track_adv.get(w, -1), 1)

        s16 = [play(r32[i], r32[i + 1], 2) for i in range(0, 4, 2)]
        picks.extend(s16)
        if track_adv is not None:
            for w in s16:
                track_adv[w] = max(track_adv.get(w, -1), 2)

        e8_winner = play(s16[0], s16[1], 3)
        picks.append(e8_winner)
        if track_adv is not None:
            track_adv[e8_winner] = max(track_adv.get(e8_winner, -1), 3)
        region_winners[region] = e8_winner

    champs = [region_winners[r] for r in _F4_REGION_ORDER if r in region_winners]
    f4_winner_0 = play(champs[0], champs[1], 4)
    f4_winner_1 = play(champs[2], champs[3], 4)
    picks.extend([f4_winner_0, f4_winner_1])
    if track_adv is not None:
        track_adv[f4_winner_0] = max(track_adv.get(f4_winner_0, -1), 4)
        track_adv[f4_winner_1] = max(track_adv.get(f4_winner_1, -1), 4)

    ncg_winner = play(f4_winner_0, f4_winner_1, 5)
    picks.append(ncg_winner)
    if track_adv is not None:
        track_adv[ncg_winner] = max(track_adv.get(ncg_winner, -1), 5)

    return picks


def simulate_forward_bracket(
    params: dict[str, float],
    region_games: dict[str, list[tuple[str, str]]],
    rng: np.random.Generator,
    round_sharpening: np.ndarray | None = None,
) -> list[str]:
    """Forward-simulate a complete bracket using power ratings.

    P(A wins game round r) = sigmoid(alpha[r] * (eff_A - eff_B))
    where eff_i = theta_i + gamma  if team i has previously beaten a
    higher-rated opponent, else theta_i.  gamma is read from params.

    Parameters
    ----------
    params : combined dict from fit_power_ratings (theta + alpha + gamma)
    region_games : {region: [(teamA, teamB), ...]} from _sort_region_games
    rng : numpy Generator

    Returns
    -------
    picks : list[str] of 63 team names, ordered identically to
        _simulate_bracket_rows picks output.
    """
    theta_dict, alpha, gamma, sigma = load_params(params)
    return _play_bracket(theta_dict, alpha, gamma, sigma, region_games, rng,
                         round_sharpening=round_sharpening)


def compute_marginals_sim(
    params: dict[str, float],
    region_games: dict[str, list[tuple[str, str]]],
    rng: np.random.Generator,
    n_sims: int = 20000,
    gamma: float | None = None,
    sigma: float | None = None,
) -> tuple[list[str], np.ndarray]:  # type: ignore[type-arg]
    """Estimate marginal advancement probabilities via Monte Carlo simulation.

    Handles path-dependent effects (e.g. gamma upset boost) that the
    analytical DP in compute_marginals cannot capture.

    Parameters
    ----------
    params : combined params dict (theta + alpha + gamma sentinels)
    region_games : {region: [(teamA, teamB), ...]}
    rng : numpy Generator
    n_sims : number of bracket simulations
    gamma : upset boost override; if None, reads gamma from params (default)

    Returns
    -------
    teams : list[str] — team names in bracket order
    adv   : (n_teams, 6) array — fraction of sims each team won through each round
            column 0 = R64 game (Yahoo "R32"), ..., column 5 = NCG game (Yahoo "Champ")
    """
    theta_dict, alpha, gamma_from_params, sigma_from_params = load_params(params)
    gamma_eff: float = gamma_from_params if gamma is None else gamma
    sigma_eff: float = sigma_from_params if sigma is None else sigma

    teams: list[str] = []
    team_idx: dict[str, int] = {}
    for region in sorted(region_games.keys()):
        for a, b in region_games[region]:
            for t in (a, b):
                if t not in team_idx:
                    team_idx[t] = len(teams)
                    teams.append(t)

    n = len(teams)
    adv_counts = np.zeros((n, 6), dtype=np.float64)

    for _ in range(n_sims):
        track_adv: dict[str, int] = {}
        _play_bracket(theta_dict, alpha, gamma_eff, sigma_eff, region_games, rng, track_adv)
        for team, round_won in track_adv.items():
            if round_won >= 0:
                adv_counts[team_idx[team], : round_won + 1] += 1.0

    return teams, adv_counts / n_sims
