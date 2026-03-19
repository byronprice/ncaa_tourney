from __future__ import annotations

import difflib
import os
import re
from io import StringIO

import pandas as pd
import requests


REQUIRED_COLUMNS = ["Team", "Rating", "Source"]
REQUIRED_TEMPO_COLUMNS = ["Team", "AdjT"]

DEFAULT_TEMPO = 68.0

TEAM_ALIASES = {
    "uconn": "connecticut",
    "uconn huskies": "connecticut",
    "byu": "brigham young",
    "byu cougars": "brigham young",
    "duke blue devils": "duke",
    "houston cougars": "houston",
    "ole miss": "mississippi",
    "unc": "north carolina",
    "lsu": "louisiana state",
    "miami fl": "miami",
    "saint marys": "st marys",
    "st marys ca": "st marys",
    "uc santa barbara": "ucsb",
    "florida atlantic": "fau",
    "texas am": "texas a m",
    "siu edwardsville": "siue",
    "ball state": "ball st",
    "uic": "illinois chicago",
    "app state": "appalachian st",
    "app st": "appalachian st",
    "long island university": "liu",
    "long island": "liu",
    "ualbany": "albany",
    "ualbany great": "albany",
    "cal state northridge": "csun",
    "cal st northridge": "csun",
    "ut martin": "tennessee martin",
    "iu indianapolis": "iu indy",
    "omaha": "nebraska omaha",
    "pennsylvania": "penn",
    "hawai i": "hawai i",
    "hawaii": "hawai i",
    "uc santa barbara": "uc santa barbara",
    "ul monroe": "louisiana monroe",
    "ul monroe warhawks": "louisiana monroe",
    "louisiana monroe": "louisiana monroe",
}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

MASCOT_SUFFIXES = {
    "wildcats",
    "bulldogs",
    "huskies",
    "cougars",
    "tigers",
    "eagles",
    "hawks",
    "raiders",
    "knights",
    "lions",
    "bears",
    "boilermakers",
    "tarheels",
    "tar heels",
    "bluejays",
    "blue jays",
    "blue devils",
    "fighting illini",
    "wolfpack",
    "wolf pack",
    "crimson tide",
    "spartans",
    "buckeyes",
    "cavaliers",
    "badgers",
    "volunteers",
    "jayhawks",
    "terrapins",
    "cyclones",
    "longhorns",
    "razorbacks",
    "hoosiers",
    "aggies",
    "orange",
    "friars",
    "lobos",
    "gonzaga bulldogs",
    "rebels",
    "cardinals",
    "flames",
    "mountaineers",
    "sharks",
    "danes",
    "matadors",
    "skyhawks",
    "jaguars",
    "mavericks",
    "quakers",
    "great danes",
    # additional mascots
    "commodores",
    "cornhuskers",
    "red storm",
    "hawkeyes",
    "wolverines",
    "hurricanes",
    "bruins",
    "rams",
    "trojans",
    "gators",
    "saints",
    "bulls",
    "broncos",
    "panthers",
    "cowboys",
    "paladins",
    "vandals",
    "pride",
    "owls",
    "redhawks",
    "bison",
    "billikens",
    "gaels",
    "royals",
    "zips",
    "lancers",
    "warriors",
    "rainbow warriors",
    "horned frogs",
    "red raiders",
}

TOKEN_REPLACEMENTS = {
    "state": "st",
    "saint": "st",
    "university": "",
}


def load_rankings_manual_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_COLUMNS)
    df = df.copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    if df["Rating"].isna().any():
        bad = df[df["Rating"].isna()]["Team"].tolist()
        raise ValueError(f"Invalid numeric Rating for teams: {bad}")
    if "Tempo" in df.columns:
        df["Tempo"] = pd.to_numeric(df["Tempo"], errors="coerce")
    return _normalize_rankings(df)


def load_rankings_espn_bpi(
    url: str = "https://www.espn.com/mens-college-basketball/bpi",
    max_pages: int = 20,
) -> pd.DataFrame:
    all_pages = []
    seen_teams: set[str] = set()

    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}?page={page}"
        page_df = _load_rankings_espn_bpi_page(page_url)
        if page_df.empty:
            break

        teams = set(page_df["Team"].tolist())
        if teams and teams.issubset(seen_teams):
            break

        all_pages.append(page_df)
        seen_teams.update(teams)

    if not all_pages:
        raise RuntimeError("No rankings parsed from ESPN BPI pages")

    out = pd.concat(all_pages, ignore_index=True)
    return _normalize_rankings(out)


def _load_rankings_espn_bpi_page(url: str) -> pd.DataFrame:
    response = _fetch_webpage(url)
    response.raise_for_status()

    try:
        tables = pd.read_html(StringIO(response.text), flavor="lxml")
    except ValueError:
        return pd.DataFrame(columns=["Team", "Rating", "Source"])
    if not tables:
        return pd.DataFrame(columns=["Team", "Rating", "Source"])

    team_table = None
    metric_table = None
    for table in tables:
        columns = [str(c) for c in table.columns]
        if team_table is None and any("Team" in c for c in columns):
            team_table = table
        if metric_table is None and any("BPI" in c for c in columns):
            metric_table = table

    if team_table is None or metric_table is None:
        return pd.DataFrame(columns=["Team", "Rating", "Source"])

    team_col = _first_matching_column(team_table, ["Team"])
    bpi_col = _first_matching_column(metric_table, ["BPI"])
    if team_col is None or bpi_col is None:
        return pd.DataFrame(columns=["Team", "Rating", "Source"])

    n_rows = min(len(team_table), len(metric_table))
    out = pd.DataFrame(
        {
            "Team": team_table[team_col].iloc[:n_rows].astype(str).map(_clean_team_name),
            "Rating": pd.to_numeric(metric_table[bpi_col].iloc[:n_rows], errors="coerce"),
            "Source": "espn_bpi",
        }
    )
    return out.dropna(subset=["Team", "Rating"])


def load_tempo_manual_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate_columns(df, REQUIRED_TEMPO_COLUMNS)
    out = pd.DataFrame(
        {
            "Team": df["Team"].astype(str).map(_clean_team_name),
            "Tempo": pd.to_numeric(df["AdjT"], errors="coerce"),
            "TempoSource": "manual_tempo",
        }
    )
    out = out.dropna(subset=["Team", "Tempo"])
    return _normalize_tempo(out)


def load_tempo_kenpom_public(url: str = "https://kenpom.com/") -> pd.DataFrame:
    response = _fetch_webpage(url, cookie_env_var="KENPOM_COOKIE")
    response.raise_for_status()
    return _parse_kenpom_tempo_tables(response.text, source="kenpom_public")


def load_tempo_kenpom_html(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as file:
        html = file.read()
    return _parse_kenpom_tempo_tables(html, source="kenpom_html")


def load_rankings_kenpom_public(url: str = "https://kenpom.com/") -> pd.DataFrame:
    response = _fetch_webpage(url, cookie_env_var="KENPOM_COOKIE")
    response.raise_for_status()
    return _parse_kenpom_ratings_tables(response.text, source="kenpom_public")


def load_rankings_kenpom_html(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as file:
        html = file.read()
    return _parse_kenpom_ratings_tables(html, source="kenpom_html")


def _parse_kenpom_tempo_tables(html: str, source: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), flavor="lxml")

    if not tables:
        raise RuntimeError("No tables found on KenPom page")

    tempo_table = None
    for table in tables:
        columns = [str(c) for c in table.columns]
        if any("Team" in c for c in columns) and any("AdjT" in c for c in columns):
            tempo_table = table
            break

    if tempo_table is None:
        raise RuntimeError("Could not detect Team/AdjT columns from KenPom table")

    team_col = _first_matching_column(tempo_table, ["Team"])
    tempo_col = _first_matching_column(tempo_table, ["AdjT"])
    if team_col is None or tempo_col is None:
        raise RuntimeError("Could not detect Team/AdjT columns from KenPom table")

    out = pd.DataFrame(
        {
            "Team": tempo_table[team_col].astype(str).map(_clean_team_name),
            "Tempo": pd.to_numeric(tempo_table[tempo_col], errors="coerce"),
            "TempoSource": source,
        }
    )
    out = out.dropna(subset=["Team", "Tempo"])
    return _normalize_tempo(out)


def _parse_kenpom_ratings_tables(html: str, source: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html), flavor="lxml")

    if not tables:
        raise RuntimeError("No tables found on KenPom page")

    ratings_table = None
    for table in tables:
        columns = [str(c) for c in table.columns]
        if any("Team" in c for c in columns) and any("NetRtg" in c or "Net Rtg" in c for c in columns):
            ratings_table = table
            break

    if ratings_table is None:
        raise RuntimeError("Could not detect Team/NetRtg columns from KenPom table")

    team_col = _first_matching_column(ratings_table, ["Team"])
    # NetRtg header may appear as 'NetRtg' or 'Net Rtg' or 'AdjEM' depending on export
    net_col = None
    candidates: list[tuple[object, str]] = []
    for col in ratings_table.columns:
        name = str(col)
        lname = name.lower()
        compact = lname.replace(" ", "")
        if "netrtg" in compact or "net rtg" in lname or "adjem" in lname:
            candidates.append((col, lname))

    def _looks_like_sos(header: str) -> bool:
        return (
            "sos" in header
            or "strength" in header
            or "strengthofschedule" in header
            or "strength of schedule" in header
        )

    if candidates:
        non_sos = [c for c in candidates if not _looks_like_sos(c[1])]
        net_col = non_sos[0][0] if non_sos else candidates[0][0]

    if team_col is None or net_col is None:
        raise RuntimeError("Could not detect Team/NetRtg columns from KenPom table")

    out = pd.DataFrame(
        {
            "Team": ratings_table[team_col].astype(str).map(_clean_team_name),
            "Rating": pd.to_numeric(ratings_table[net_col], errors="coerce") * 0.7,
            "Source": source,
        }
    )
    out = out.dropna(subset=["Team", "Rating"])
    return _normalize_rankings(out)


def load_kenpom_efficiencies(path: str) -> dict[str, tuple[float, float]]:
    """Parse a KenPom HTML file and return {canonical_key: (AdjO, AdjD)} per 100 possessions.

    Keys are canonical (same as used in overlay_kenpom_ratings), so lookups should
    also use _resolve_alias(_canonical_team_key(team_name)).
    """
    from bs4 import BeautifulSoup

    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    table = soup.find("table", id="ratings-table")
    if table is None:
        raise RuntimeError("Could not find #ratings-table in KenPom HTML")

    result: dict[str, tuple[float, float]] = {}
    for row in table.find("tbody").find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 9:
            continue
        team = _clean_team_name(cells[1])
        key = _resolve_alias(_canonical_team_key(team))
        try:
            adj_o = float(cells[5])
            adj_d = float(cells[7])
        except (ValueError, IndexError):
            continue
        result[key] = (adj_o, adj_d)

    return result


def overlay_kenpom_ratings(
    base_df: pd.DataFrame,
    kenpom_df: pd.DataFrame,
    min_similarity: float = 0.86,
) -> pd.DataFrame:
    """Return base_df (ESPN team names) with ratings replaced by KenPom values.

    For each ESPN team, the best-matching KenPom team's rating is substituted.
    Teams with no KenPom match keep their original ESPN rating.  Because we
    iterate over ESPN teams exactly once, there are no duplicate team names.
    """
    kenpom_keys: dict[str, float] = {}
    for row in kenpom_df.itertuples(index=False):
        key = _resolve_alias(_canonical_team_key(str(row.Team)))
        kenpom_keys[key] = float(row.Rating)
    kenpom_key_list = list(kenpom_keys.keys())

    result = base_df.copy()
    for idx, row in base_df.iterrows():
        team_name = str(row["Team"])
        key = _resolve_alias(_canonical_team_key(team_name))

        rating: float | None = None

        if key in kenpom_keys:
            rating = kenpom_keys[key]
        else:
            subset_key = _find_subset_match(key, kenpom_key_list)
            if subset_key is not None and _token_overlap_score(key, subset_key) >= 0.75:
                rating = kenpom_keys[subset_key]
            else:
                close = difflib.get_close_matches(key, kenpom_key_list, n=1, cutoff=min_similarity)
                if close:
                    rating = kenpom_keys[close[0]]

        if rating is not None:
            result.at[idx, "Rating"] = rating
            result.at[idx, "Source"] = "kenpom_html"

    return result


def merge_rankings_with_tempo(
    rankings_df: pd.DataFrame,
    tempo_df: pd.DataFrame,
    min_similarity: float = 0.86,
    default_tempo: float = DEFAULT_TEMPO,
) -> pd.DataFrame:
    rankings = rankings_df.copy()
    tempo = _normalize_tempo(tempo_df)

    tempo_by_key = {}
    for row in tempo.itertuples(index=False):
        row_team = str(row.Team)
        row_tempo = float(pd.to_numeric(row.Tempo, errors="coerce"))
        row_source = str(row.TempoSource)
        tempo_by_key[_resolve_alias(_canonical_team_key(row_team))] = {
            "Tempo": row_tempo,
            "TempoSource": row_source,
        }
    tempo_keys = list(tempo_by_key.keys())

    merged_rows = []
    misses = 0
    for row in rankings.itertuples(index=False):
        team_name = str(row.Team)
        key = _resolve_alias(_canonical_team_key(team_name))

        match = tempo_by_key.get(key)
        match_type = "exact"
        match_score = 1.0

        if match is None:
            subset_key = _find_subset_match(key, tempo_keys)
            if subset_key is not None:
                match = tempo_by_key[subset_key]
                match_type = "subset"
                match_score = _token_overlap_score(key, subset_key)

        if match is None:
            close = difflib.get_close_matches(key, tempo_keys, n=1, cutoff=min_similarity)
            if close:
                match = tempo_by_key[close[0]]
                match_type = "fuzzy"
                match_score = difflib.SequenceMatcher(None, key, close[0]).ratio()

        if match is None:
            match = {"Tempo": default_tempo, "TempoSource": "tempo_default"}
            match_type = "default"
            match_score = 0.0
            misses += 1

        merged_rows.append(
            {
                "Team": team_name,
                "Rating": float(pd.to_numeric(row.Rating, errors="coerce")),
                "Source": str(row.Source),
                "Tempo": float(pd.to_numeric(match["Tempo"], errors="coerce")),
                "TempoSource": str(match["TempoSource"]),
                "TempoMatchType": match_type,
                "TempoMatchScore": round(float(match_score), 4),
            }
        )

    if misses:
        print(f"Warning: default tempo applied for {misses} teams")

    return pd.DataFrame(merged_rows).sort_values("Rating", ascending=False).reset_index(drop=True)


def remap_team_names(
    df: pd.DataFrame,
    canonical_names: list[str],
    min_similarity: float = 0.86,
) -> pd.DataFrame:
    """Replace the Team column with the closest match from canonical_names.

    Uses the same matching pipeline as merge_rankings_with_tempo (exact key,
    subset, fuzzy).  Each canonical name is claimed by at most one source team.
    When multiple source teams propose the same canonical name, the winner is
    chosen by direct string similarity between the original source name and the
    original canonical display name — not by rating.  Losers are left unchanged.
    """
    canon_keys = {_resolve_alias(_canonical_team_key(n)): n for n in canonical_names}
    canon_key_list = list(canon_keys.keys())

    source_names = [str(n) for n in df["Team"]]

    # Pass 1: for every source row compute its best candidate canonical key.
    proposals: list[tuple[int, str | None]] = []  # (row_idx, canon_key or None)
    for idx, team_name in enumerate(source_names):
        key = _resolve_alias(_canonical_team_key(team_name))
        if key in canon_keys:
            proposals.append((idx, key))
        else:
            subset_key = _find_subset_match(key, canon_key_list)
            if subset_key is not None:
                proposals.append((idx, subset_key))
            else:
                close = difflib.get_close_matches(key, canon_key_list, n=1, cutoff=min_similarity)
                proposals.append((idx, close[0] if close else None))

    # Pass 2: for each canonical key that multiple rows propose, pick the winner
    # by comparing the original source name against the canonical display name.
    from collections import defaultdict
    claimants: dict[str, list[int]] = defaultdict(list)
    for row_idx, ck in proposals:
        if ck is not None:
            claimants[ck].append(row_idx)

    winners: dict[int, str] = {}
    for ck, row_indices in claimants.items():
        display_name = canon_keys[ck]
        if len(row_indices) == 1:
            winners[row_indices[0]] = display_name
        else:
            # Pick the source name most similar to the canonical display name
            def name_similarity(row_idx: int) -> float:
                return difflib.SequenceMatcher(
                    None,
                    _clean_team_name(source_names[row_idx]).lower(),
                    display_name.lower(),
                ).ratio()
            best = max(row_indices, key=name_similarity)
            winners[best] = display_name

    result = df.copy()
    for row_idx, canon_name in winners.items():
        result.at[row_idx, "Team"] = canon_name

    return result


def _first_matching_column(df: pd.DataFrame, keys: list[str]) -> str | None:
    for column in df.columns:
        name = str(column)
        if all(k.lower() in name.lower() for k in keys):
            return column
    return None


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_rankings(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["Team"] = ranked["Team"].astype(str).map(_clean_team_name)
    ranked["Source"] = ranked["Source"].astype(str).str.strip().replace("", "unknown")
    ranked = ranked.drop_duplicates(subset=["Team"], keep="first")
    return ranked.sort_values("Rating", ascending=False).reset_index(drop=True)


def _normalize_tempo(df: pd.DataFrame) -> pd.DataFrame:
    tempo = df.copy()
    tempo["Team"] = tempo["Team"].astype(str).map(_clean_team_name)
    tempo["Tempo"] = pd.to_numeric(tempo["Tempo"], errors="coerce")
    if "TempoSource" not in tempo.columns:
        tempo["TempoSource"] = "unknown"
    tempo["TempoSource"] = tempo["TempoSource"].astype(str).str.strip().replace("", "unknown")
    tempo = tempo.dropna(subset=["Team", "Tempo"])
    tempo = tempo.drop_duplicates(subset=["Team"], keep="first")
    return tempo.reset_index(drop=True)


def _clean_team_name(name: str) -> str:
    cleaned = str(name)
    cleaned = re.sub(r"^\d+\s+", "", cleaned)
    cleaned = re.sub(r"\s*\d+$", "", cleaned)
    cleaned = re.sub(r"\s+\(\d+-\d+\)$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _fetch_webpage(url: str, cookie_env_var: str = "") -> requests.Response:
    headers = dict(BROWSER_HEADERS)
    cookie_value = os.environ.get(cookie_env_var, "") if cookie_env_var else ""
    if cookie_value:
        headers["Cookie"] = cookie_value
    return requests.get(url, timeout=30, headers=headers)


def _canonical_team_key(name: str) -> str:
    key = _clean_team_name(name).lower()
    key = _strip_mascot_suffix(key)
    key = key.replace("&", " and ")
    key = key.replace("st.", "st")
    key = re.sub(r"[^a-z0-9 ]", " ", key)
    tokens = []
    for token in key.split():
        mapped = TOKEN_REPLACEMENTS.get(token, token)
        if mapped:
            tokens.append(mapped)
    key = " ".join(tokens)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _strip_mascot_suffix(name: str) -> str:
    tokens = name.split()
    if len(tokens) < 2:
        return name

    joined = " ".join(tokens)
    for length in [3, 2, 1]:
        if len(tokens) <= length:
            continue
        suffix = " ".join(tokens[-length:])
        if suffix in MASCOT_SUFFIXES:
            return " ".join(tokens[:-length])

    if joined in MASCOT_SUFFIXES:
        return name
    return name


def _find_subset_match(key: str, candidates: list[str]) -> str | None:
    key_tokens = set(key.split())
    if not key_tokens:
        return None

    best: tuple[float, str] | None = None
    for candidate in candidates:
        cand_tokens = set(candidate.split())
        if not cand_tokens:
            continue

        if cand_tokens.issubset(key_tokens) or key_tokens.issubset(cand_tokens):
            score = _token_overlap_score(key, candidate)
            if best is None or score > best[0]:
                best = (score, candidate)

    return best[1] if best else None


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens.union(right_tokens))


def _resolve_alias(key: str) -> str:
    return TEAM_ALIASES.get(key, key)


def create_rankings_template(path: str) -> None:
    template = pd.DataFrame(
        [
            {"Team": "Purdue", "Rating": 23.4, "Source": "manual"},
            {"Team": "Connecticut", "Rating": 22.9, "Source": "manual"},
        ]
    )
    template.to_csv(path, index=False)


def create_tempo_template(path: str) -> None:
    template = pd.DataFrame(
        [
            {"Team": "Purdue", "AdjT": 67.5},
            {"Team": "Connecticut", "AdjT": 66.8},
        ]
    )
    template.to_csv(path, index=False)
