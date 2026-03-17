from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path
from typing import cast

import pandas as pd

from ncaa_tourney.bracket import (
    create_bracket_template,
    load_bracket_from_public_table,
    load_bracket_manual_csv,
)
from ncaa_tourney.dataset import build_team_and_game_tables
from ncaa_tourney.io_utils import ensure_parent
from ncaa_tourney.rankings import (
    create_rankings_template,
    create_tempo_template,
    load_rankings_espn_bpi,
    load_rankings_manual_csv,
    load_rankings_kenpom_html,
    load_rankings_kenpom_public,
    load_tempo_kenpom_html,
    load_tempo_kenpom_public,
    load_tempo_manual_csv,
    merge_rankings_with_tempo,
    overlay_kenpom_ratings,
)
from ncaa_tourney.simulation import simulate_tournament
from ncaa_tourney.simulation import generate_strategy_brackets
from ncaa_tourney.simulation import optimize_pool_bracket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NCAA tournament predictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rank = sub.add_parser("init-rankings-template")
    p_rank.add_argument("--out", required=True)

    p_tempo = sub.add_parser("init-tempo-template")
    p_tempo.add_argument("--out", required=True)

    p_bracket = sub.add_parser("init-bracket-template")
    p_bracket.add_argument("--out", required=True)

    p_build = sub.add_parser("build-dataset")
    p_build.add_argument("--rankings-source", choices=["manual_csv", "espn_bpi", "merged_report", "kenpom_html", "kenpom_public"], required=True)
    p_build.add_argument("--rankings-path", default="")
    p_build.add_argument(
        "--tempo-source",
        choices=["none", "manual_csv", "kenpom_public", "kenpom_html"],
        default="none",
    )
    p_build.add_argument("--tempo-path", default="")
    p_build.add_argument("--tempo-url", default="https://kenpom.com/")
    p_build.add_argument("--team-match-threshold", type=float, default=0.86)
    p_build.add_argument("--default-tempo", type=float, default=70.0)
    p_build.add_argument("--bracket-source", choices=["manual_csv", "public_table"], required=True)
    p_build.add_argument("--bracket-path", default="")
    p_build.add_argument("--bracket-url", default="")
    p_build.add_argument("--out-teams", required=True)
    p_build.add_argument("--out-games", required=True)

    p_check = sub.add_parser("check-sources")
    p_check.add_argument("--rankings-source", choices=["manual_csv", "espn_bpi", "kenpom_html", "kenpom_public"], required=True)
    p_check.add_argument("--rankings-path", default="")
    p_check.add_argument(
        "--tempo-source",
        choices=["manual_csv", "kenpom_public", "kenpom_html"],
        required=True,
    )
    p_check.add_argument("--tempo-path", default="")
    p_check.add_argument("--tempo-url", default="https://kenpom.com/")
    p_check.add_argument("--team-match-threshold", type=float, default=0.86)
    p_check.add_argument("--default-tempo", type=float, default=68.0)
    p_check.add_argument("--out-report", required=True)
    p_check.add_argument("--out-unmatched", default="")
    p_check.add_argument("--out-alias-suggestions", default="")

    p_sim = sub.add_parser("simulate")
    p_sim.add_argument("--teams", required=True)
    p_sim.add_argument("--games", required=True)
    p_sim.add_argument("--n-sims", type=int, default=20000)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--spread-a", type=float, default=-0.78)
    p_sim.add_argument("--spread-b", type=float, default=12.99)
    p_sim.add_argument("--out-summary", required=True)
    p_sim.add_argument("--out-brackets", required=True)
    p_sim.add_argument("--r64-odds", default="")

    p_picks = sub.add_parser("make-picks")
    p_picks.add_argument("--teams", required=True)
    p_picks.add_argument("--games", required=True)
    p_picks.add_argument("--seed", type=int, default=42)
    p_picks.add_argument("--spread-a", type=float, default=-0.78)
    p_picks.add_argument("--spread-b", type=float, default=12.99)
    p_picks.add_argument("--r64-odds", default="")
    p_picks.add_argument("--out", required=True)

    p_opt = sub.add_parser("optimize-picks")
    p_opt.add_argument("--teams", required=True)
    p_opt.add_argument("--games", required=True)
    p_opt.add_argument("--pool-size", type=str, default="50")
    p_opt.add_argument("--n-candidates", type=int, default=300)
    p_opt.add_argument("--n-outcomes", type=int, default=2000)
    p_opt.add_argument("--seed", type=int, default=42)
    p_opt.add_argument("--spread-a", type=float, default=-0.78)
    p_opt.add_argument("--spread-b", type=float, default=12.99)
    p_opt.add_argument("--round-points", default="1,2,4,8,16,32")
    p_opt.add_argument("--candidate-mix", default="0.34,0.33,0.33")
    p_opt.add_argument("--opponent-mix", default="0.5,0.35,0.15")
    p_opt.add_argument("--opponent-safe-seed-chalk-share", type=float, default=0.0)
    p_opt.add_argument("--opponent-seed-popularity", default="")
    p_opt.add_argument("--r64-odds", default="")
    p_opt.add_argument("--out", required=True)
    p_opt.add_argument("--out-summary", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-rankings-template":
        ensure_parent(args.out)
        create_rankings_template(args.out)
        print(f"Wrote rankings template to {args.out}")
        return

    if args.command == "init-bracket-template":
        ensure_parent(args.out)
        create_bracket_template(args.out)
        print(f"Wrote bracket template to {args.out}")
        return

    if args.command == "init-tempo-template":
        ensure_parent(args.out)
        create_tempo_template(args.out)
        print(f"Wrote tempo template to {args.out}")
        return

    if args.command == "build-dataset":
        # Support using a previously-generated merged source report directly.
        if args.rankings_source == "merged_report":
            if not args.rankings_path:
                raise ValueError("--rankings-path required for merged_report")
            merged = pd.read_csv(args.rankings_path)
            required = {"Team", "Rating", "Source", "Tempo"}
            missing = required - set(merged.columns)
            if missing:
                missing_cols = ", ".join(sorted(missing))
                raise ValueError(f"Merged report is missing required columns: {missing_cols}")
            rankings = merged.copy()
        else:
            rankings = _load_rankings(args.rankings_source, args.rankings_path)

            if args.tempo_source != "none":
                tempo = _load_tempo(args.tempo_source, args.tempo_path, args.tempo_url)
                rankings = merge_rankings_with_tempo(
                    rankings,
                    tempo,
                    min_similarity=args.team_match_threshold,
                    default_tempo=args.default_tempo,
                )
            else:
                rankings = _apply_default_tempo(rankings, args.default_tempo)

        if args.bracket_source == "manual_csv":
            if not args.bracket_path:
                raise ValueError("--bracket-path required for manual_csv")
            bracket = load_bracket_manual_csv(args.bracket_path)
        else:
            if not args.bracket_url:
                raise ValueError("--bracket-url required for public_table")
            bracket = load_bracket_from_public_table(args.bracket_url)

        teams, games = build_team_and_game_tables(rankings, bracket)

        ensure_parent(args.out_teams)
        ensure_parent(args.out_games)
        teams.to_csv(args.out_teams, index=False)
        games.to_csv(args.out_games, index=False)
        print(f"Wrote teams to {args.out_teams}")
        print(f"Wrote games to {args.out_games}")
        return

    if args.command == "simulate":
        teams = pd.read_csv(args.teams)
        games = pd.read_csv(args.games)
        summary, top = simulate_tournament(
            teams,
            games,
            n_sims=args.n_sims,
            seed=args.seed,
            spread_a=args.spread_a,
            spread_b=args.spread_b,
            r64_odds=_load_r64_odds(args.r64_odds),
        )

        ensure_parent(args.out_summary)
        ensure_parent(args.out_brackets)
        summary.to_csv(args.out_summary, index=False)
        top.to_csv(args.out_brackets, index=False)
        print(f"Wrote simulation summary to {args.out_summary}")
        print(f"Wrote top bracket paths to {args.out_brackets}")
        return

    if args.command == "make-picks":
        teams = pd.read_csv(args.teams)
        games = pd.read_csv(args.games)
        picks = generate_strategy_brackets(
            teams,
            games,
            seed=args.seed,
            spread_a=args.spread_a,
            spread_b=args.spread_b,
            r64_odds=_load_r64_odds(args.r64_odds),
        )

        ensure_parent(args.out)
        picks.to_csv(args.out, index=False)
        print(f"Wrote strategy bracket picks to {args.out}")
        return

    if args.command == "optimize-picks":
        teams = pd.read_csv(args.teams)
        games = pd.read_csv(args.games)
        round_points = _parse_round_points(args.round_points)
        candidate_mix = _parse_strategy_mix(args.candidate_mix)
        opponent_mix = _parse_strategy_mix(args.opponent_mix)
        opponent_seed_popularity = _load_seed_popularity(args.opponent_seed_popularity)

        pool_sizes = _parse_pool_sizes(args.pool_size)
        r64_odds = _load_r64_odds(args.r64_odds)

        picks, summary = optimize_pool_bracket(
            teams,
            games,
            pool_sizes=pool_sizes,
            n_candidates=args.n_candidates,
            n_outcomes=args.n_outcomes,
            seed=args.seed,
            spread_a=args.spread_a,
            spread_b=args.spread_b,
            round_points=round_points,
            candidate_mix=candidate_mix,
            opponent_mix=opponent_mix,
            opponent_safe_seed_chalk_share=args.opponent_safe_seed_chalk_share,
            opponent_seed_popularity=opponent_seed_popularity,
            r64_odds=r64_odds,
        )

        ensure_parent(args.out)
        ensure_parent(args.out_summary)
        picks.to_csv(args.out, index=False)
        summary.to_csv(args.out_summary, index=False)
        print(f"Wrote optimized picks to {args.out}")
        print(f"Wrote optimizer summary to {args.out_summary}")
        return

    if args.command == "check-sources":
        rankings = _load_rankings(args.rankings_source, args.rankings_path)
        tempo = _load_tempo(args.tempo_source, args.tempo_path, args.tempo_url)

        merged = merge_rankings_with_tempo(
            rankings,
            tempo,
            min_similarity=args.team_match_threshold,
            default_tempo=args.default_tempo,
        )
        ensure_parent(args.out_report)
        merged.to_csv(args.out_report, index=False)

        out_unmatched = args.out_unmatched or _derive_output_path(args.out_report, "_unmatched")
        out_alias = args.out_alias_suggestions or _derive_output_path(args.out_report, "_alias_suggestions")

        unmatched = cast(pd.DataFrame, merged[merged["TempoMatchType"] == "default"].copy())
        ensure_parent(out_unmatched)
        unmatched.to_csv(out_unmatched, index=False)

        alias_suggestions = _build_alias_suggestions(unmatched, tempo)
        ensure_parent(out_alias)
        alias_suggestions.to_csv(out_alias, index=False)

        counts = merged["TempoMatchType"].value_counts()
        print(f"Wrote source-link report to {args.out_report}")
        print(f"Wrote unmatched teams to {out_unmatched}")
        print(f"Wrote alias suggestions to {out_alias}")
        print(f"Rankings teams: {len(rankings)}")
        print(f"Tempo teams: {len(tempo)}")
        print(f"Unmatched teams: {len(unmatched)}")
        for label, count in counts.items():
            print(f"{label}: {count}")
        return

    raise RuntimeError(f"Unknown command: {args.command}")



def _load_rankings(source: str, path: str) -> pd.DataFrame:
    if source == "manual_csv":
        if not path:
            raise ValueError("--rankings-path required for manual_csv")
        return load_rankings_manual_csv(path)
    if source == "kenpom_public":
        espn_base = load_rankings_espn_bpi()
        kenpom = load_rankings_kenpom_public()
        return overlay_kenpom_ratings(espn_base, kenpom)
    if source == "kenpom_html":
        if not path:
            raise ValueError("--rankings-path required for kenpom_html")
        espn_base = load_rankings_espn_bpi()
        kenpom = load_rankings_kenpom_html(path)
        return overlay_kenpom_ratings(espn_base, kenpom)
    return load_rankings_espn_bpi()


def _load_tempo(source: str, path: str, url: str) -> pd.DataFrame:
    if source == "manual_csv":
        if not path:
            raise ValueError("--tempo-path required for manual_csv tempo source")
        return load_tempo_manual_csv(path)
    if source == "kenpom_public":
        return load_tempo_kenpom_public(url)
    if source == "kenpom_html":
        if not path:
            raise ValueError("--tempo-path required for kenpom_html tempo source")
        return load_tempo_kenpom_html(path)
    raise ValueError(f"Unsupported tempo source: {source}")


def _apply_default_tempo(rankings: pd.DataFrame, default_tempo: float) -> pd.DataFrame:
    out = rankings.copy()
    out["Tempo"] = default_tempo
    out["TempoSource"] = "tempo_default"
    out["TempoMatchType"] = "default"
    out["TempoMatchScore"] = 0.0
    return out


def _derive_output_path(base_path: str, suffix: str) -> str:
    path = Path(base_path)
    stem = path.stem or "report"
    filename = f"{stem}{suffix}.csv"
    return str(path.with_name(filename))


def _parse_pool_sizes(value: str) -> list[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    sizes = [int(token) for token in tokens]
    for s in sizes:
        if s < 2:
            raise ValueError("Each pool size must be at least 2")
    return sizes


def _parse_round_points(value: str) -> dict[str, int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != 6:
        raise ValueError("--round-points must contain exactly 6 comma-separated integers")

    points = [int(token) for token in tokens]
    rounds = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    return dict(zip(rounds, points))


def _parse_strategy_mix(value: str) -> dict[str, float]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if len(tokens) != 3:
        raise ValueError("Strategy mix must contain exactly 3 comma-separated numbers: safe,balanced,upset_heavy")

    safe, balanced, upset_heavy = [float(token) for token in tokens]
    return {
        "safe": safe,
        "balanced": balanced,
        "upset_heavy": upset_heavy,
    }


def _load_r64_odds(path: str) -> dict[frozenset[str], tuple[str, float]] | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    required = {"Favorite", "Underdog", "Probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"--r64-odds CSV missing columns: {', '.join(sorted(missing))}")
    result: dict[frozenset[str], tuple[str, float]] = {}
    for row in frame.itertuples(index=False):
        key: frozenset[str] = frozenset({str(row.Favorite), str(row.Underdog)})
        result[key] = (str(row.Favorite), float(row.Probability))
    return result


def _load_seed_popularity(path: str) -> dict[str, dict[tuple[int, int], float]] | None:
    if not path:
        return None

    frame = pd.read_csv(path)
    required = {"Round", "SeedFavorite", "SeedUnderdog", "UnderdogPickRate"}
    missing = required - set(frame.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Popularity file is missing columns: {missing_cols}")

    table: dict[str, dict[tuple[int, int], float]] = {}
    for row in frame.itertuples(index=False):
        round_name = str(row.Round).strip()
        seed_favorite_raw = pd.to_numeric(getattr(row, "SeedFavorite"), errors="coerce")
        seed_underdog_raw = pd.to_numeric(getattr(row, "SeedUnderdog"), errors="coerce")
        raw_rate_value = pd.to_numeric(getattr(row, "UnderdogPickRate"), errors="coerce")
        if pd.isna(seed_favorite_raw) or pd.isna(seed_underdog_raw) or pd.isna(raw_rate_value):
            continue

        seed_favorite = int(float(seed_favorite_raw))
        seed_underdog = int(float(seed_underdog_raw))
        raw_rate = float(raw_rate_value)
        rate = raw_rate / 100.0 if raw_rate > 1.0 else raw_rate
        rate = min(max(rate, 0.0), 1.0)

        if round_name not in table:
            table[round_name] = {}
        table[round_name][(min(seed_favorite, seed_underdog), max(seed_favorite, seed_underdog))] = rate

    return table


def _normalize_for_suggestion(name: str) -> str:
    mascot_words = {
        "wildcats",
        "bulldogs",
        "huskies",
        "cougars",
        "tigers",
        "eagles",
        "hawks",
        "knights",
        "lions",
        "bears",
        "boilermakers",
        "blue",
        "devils",
        "crimson",
        "tide",
        "fighting",
        "illini",
        "university",
    }
    replacements = {
        "state": "st",
        "saint": "st",
        "university": "",
    }
    value = str(name).lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    tokens = []
    for token in value.split():
        mapped = replacements.get(token, token)
        if mapped and mapped not in mascot_words:
            tokens.append(mapped)
    return " ".join(tokens).strip()


def _build_alias_suggestions(unmatched: pd.DataFrame, tempo: pd.DataFrame) -> pd.DataFrame:
    if unmatched.empty:
        return pd.DataFrame(columns=["Team", "SuggestedTempoTeam", "SuggestedScore"])

    tempo_names = [str(name) for name in tempo["Team"].tolist()]
    tempo_keys = {name: _normalize_for_suggestion(name) for name in tempo_names}

    rows = []
    for team_name in [str(name) for name in unmatched["Team"].tolist()]:
        team_key = _normalize_for_suggestion(team_name)
        best_name = ""
        best_score = 0.0

        for tempo_name, tempo_key in tempo_keys.items():
            score = difflib.SequenceMatcher(None, team_key, tempo_key).ratio()
            if score > best_score:
                best_score = score
                best_name = tempo_name

        rows.append(
            {
                "Team": team_name,
                "SuggestedTempoTeam": best_name if best_score >= 0.72 else "",
                "SuggestedScore": round(best_score, 4) if best_score >= 0.72 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("SuggestedScore", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    main()
