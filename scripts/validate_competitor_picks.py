"""
Validate competitor_picks.csv for logical consistency.

Checks:
  1. Pick is one of TeamA or TeamB in each row
  2. Path consistency: teams appearing in later rounds must be prior-round winners
  3. F4 pairings: East vs South (game 1), West vs Midwest (game 2)
  4. NCG teams = F4 game 1 pick vs F4 game 2 pick
  5. Champ TeamA = NCG pick
  6. R64 matchups match the official bracket
  7. All team names are valid tournament participants
"""

import sys
import csv
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PICKS_PATH   = ROOT / "data/processed/competitor_picks.csv"
BRACKET_PATH = ROOT / "data/processed/round1_games.csv"

# ── Official bracket: short name mapping ──────────────────────────────────────
# Maps full official names (round1_games.csv) → short names used in competitor_picks.csv
OFFICIAL_TO_SHORT = {
    "Duke Blue Devils": "Duke",
    "Ohio State Buckeyes": "Ohio State",
    "St. John's Red Storm": "St. John's",
    "Kansas Jayhawks": "Kansas",
    "Louisville Cardinals": "Louisville",
    "Michigan State Spartans": "Michigan State",
    "UCLA Bruins": "UCLA",
    "UConn Huskies": "UConn",
    "Siena Saints": "Siena",
    "TCU Horned Frogs": "TCU",
    "Northern Iowa Panthers": "Northern Iowa",
    "California Baptist Lancers": "Cal Baptist",
    "South Florida Bulls": "South Florida",
    "North Dakota State Bison": "North Dakota State",
    "UCF Knights": "UCF",
    "Furman Paladins": "Furman",
    "Arizona Wildcats": "Arizona",
    "Villanova Wildcats": "Villanova",
    "Wisconsin Badgers": "Wisconsin",
    "Arkansas Razorbacks": "Arkansas",
    "BYU Cougars": "BYU",
    "Gonzaga Bulldogs": "Gonzaga",
    "Miami Hurricanes": "Miami",
    "Purdue Boilermakers": "Purdue",
    "Long Island University Sharks": "Long Island",
    "Utah State Aggies": "Utah State",
    "High Point Panthers": "High Point",
    "Hawai'i Rainbow Warriors": "Hawai'i",
    "Texas Longhorns": "Texas",
    "Kennesaw State Owls": "Kennesaw State",
    "Missouri Tigers": "Missouri",
    "Queens University Royals": "Queens",
    "Florida Gators": "Florida",
    "Clemson Tigers": "Clemson",
    "Vanderbilt Commodores": "Vanderbilt",
    "Nebraska Cornhuskers": "Nebraska",
    "North Carolina Tar Heels": "North Carolina",
    "Illinois Fighting Illini": "Illinois",
    "Saint Mary's Gaels": "Saint Mary's",
    "Houston Cougars": "Houston",
    "Prairie View A&M Panthers": "Prairie View",
    "Iowa Hawkeyes": "Iowa",
    "McNeese Cowboys": "McNeese",
    "Troy Trojans": "Troy",
    "VCU Rams": "VCU",
    "Pennsylvania Quakers": "Penn",
    "Texas A&M Aggies": "Texas A&M",
    "Idaho Vandals": "Idaho",
    "Michigan Wolverines": "Michigan",
    "Georgia Bulldogs": "Georgia",
    "Texas Tech Red Raiders": "Texas Tech",
    "Alabama Crimson Tide": "Alabama",
    "Tennessee Volunteers": "Tennessee",
    "Virginia Cavaliers": "Virginia",
    "Kentucky Wildcats": "Kentucky",
    "Iowa State Cyclones": "Iowa State",
    "Howard Bison": "Howard",
    "Saint Louis Billikens": "Saint Louis",
    "Akron Zips": "Akron",
    "Hofstra Pride": "Hofstra",
    "Miami (OH) RedHawks": "Miami OH",
    "Wright State Raiders": "Wright St",
    "Santa Clara Broncos": "Santa Clara",
    "Tennessee State Tigers": "Tennessee St",
}

# All valid tournament team names (short form)
ALL_VALID_TEAMS = set(OFFICIAL_TO_SHORT.values())

# Official R64 matchups keyed by (Region, GameIndex) -> (TeamA_short, TeamB_short)
def load_official_r64():
    games = {}
    with open(BRACKET_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            region   = row["Region"]
            game_num = int(row["GameId"][1:])  # E01 -> 1
            ta = OFFICIAL_TO_SHORT.get(row["TeamA"], row["TeamA"])
            tb = OFFICIAL_TO_SHORT.get(row["TeamB"], row["TeamB"])
            games[(region, game_num)] = (ta, tb)
    return games

# ── Parse competitor picks file into individual brackets ──────────────────────
def parse_brackets(path):
    """
    Returns list of (bracket_index, list_of_(line_number, dict)) where
    each dict has keys: Round, Region, GameIndex, TeamA, TeamB, Pick.
    """
    brackets = []
    current  = []
    b_idx    = 1
    header_pattern = re.compile(r'^Round,Region', re.IGNORECASE)

    with open(path, newline="", encoding="utf-8") as f:
        lines = f.readlines()

    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            if current:
                brackets.append((b_idx, current))
                b_idx += 1
                current = []
            continue
        # Skip duplicate header rows embedded in the file
        if header_pattern.match(line):
            continue
        parts = line.split(",")
        if len(parts) < 6:
            # Malformed row; keep it for reporting
            current.append((lineno, {"_raw": line, "_malformed": True}))
            continue
        row = {
            "Round":     parts[0].strip(),
            "Region":    parts[1].strip(),
            "GameIndex": parts[2].strip(),
            "TeamA":     parts[3].strip(),
            "TeamB":     parts[4].strip(),
            "Pick":      ",".join(parts[5:]).strip(),  # in case team name has comma
        }
        current.append((lineno, row))

    if current:
        brackets.append((b_idx, current))

    return brackets

# ── Validate a single bracket ─────────────────────────────────────────────────
def validate_bracket(b_idx, rows, official_r64):
    errors = []

    def err(lineno, msg):
        errors.append(f"  Bracket {b_idx}, line {lineno}: {msg}")

    # Index rows by (Round, Region, GameIndex)
    index = {}
    for lineno, row in rows:
        if row.get("_malformed"):
            err(lineno, f"Malformed row: {row['_raw']}")
            continue
        key = (row["Round"], row["Region"], row["GameIndex"])
        if key in index:
            err(lineno, f"Duplicate game key {key}")
        index[key] = (lineno, row)

    def get_pick(round_, region, game_idx):
        """Return (lineno, pick) or (None, None) if missing."""
        key = (round_, region, str(game_idx))
        entry = index.get(key)
        if entry is None:
            return None, None
        return entry[0], entry[1]["Pick"]

    # ── Check 1: Pick must be TeamA or TeamB ──────────────────────────────────
    for lineno, row in rows:
        if row.get("_malformed"):
            continue
        rnd = row["Round"]
        ta, tb, pick = row["TeamA"], row["TeamB"], row["Pick"]

        if rnd == "Champ":
            # TeamB is "N/A"; pick should equal TeamA
            if pick != ta:
                err(lineno, f"Champ pick '{pick}' != TeamA '{ta}'")
            continue

        if pick not in (ta, tb):
            err(lineno, f"{rnd} {row['Region']} game {row['GameIndex']}: "
                        f"Pick '{pick}' not in ({ta}, {tb})")

    # ── Check 2: Valid team names ─────────────────────────────────────────────
    for lineno, row in rows:
        if row.get("_malformed"):
            continue
        for field in ("TeamA", "TeamB", "Pick"):
            name = row[field]
            if name == "N/A":
                continue
            if name not in ALL_VALID_TEAMS:
                err(lineno, f"Unknown team name '{name}' in field {field}")

    # ── Check 3: Official R64 matchups ────────────────────────────────────────
    for region in ("East", "West", "South", "Midwest"):
        for g in range(1, 9):
            key = ("R64", region, str(g))
            if key not in index:
                errors.append(f"  Bracket {b_idx}: Missing R64 {region} game {g}")
                continue
            lineno, row = index[key]
            official = official_r64.get((region, g))
            if official is None:
                continue
            actual = (row["TeamA"], row["TeamB"])
            if set(actual) != set(official):
                err(lineno, f"R64 {region} game {g}: teams {actual} != official {official}")

    # ── Check 4: Path consistency ─────────────────────────────────────────────
    # R32: each R32 game g's teams = picks from R64 games 2g-1 and 2g
    for region in ("East", "West", "South", "Midwest"):
        for g in range(1, 5):
            r32_key = ("R32", region, str(g))
            if r32_key not in index:
                continue
            lineno, r32_row = index[r32_key]
            _, pick_a = get_pick("R64", region, 2 * g - 1)
            _, pick_b = get_pick("R64", region, 2 * g)
            if pick_a is None or pick_b is None:
                continue
            expected = {pick_a, pick_b}
            actual = {r32_row["TeamA"], r32_row["TeamB"]}
            if actual != expected:
                err(lineno, f"R32 {region} game {g}: teams {sorted(actual)} != "
                            f"expected R64 winners {sorted(expected)}")

    # S16: each S16 game g's teams = picks from R32 games 2g-1 and 2g
    for region in ("East", "West", "South", "Midwest"):
        for g in range(1, 3):
            s16_key = ("S16", region, str(g))
            if s16_key not in index:
                continue
            lineno, s16_row = index[s16_key]
            _, pick_a = get_pick("R32", region, 2 * g - 1)
            _, pick_b = get_pick("R32", region, 2 * g)
            if pick_a is None or pick_b is None:
                continue
            expected = {pick_a, pick_b}
            actual = {s16_row["TeamA"], s16_row["TeamB"]}
            if actual != expected:
                err(lineno, f"S16 {region} game {g}: teams {sorted(actual)} != "
                            f"expected R32 winners {sorted(expected)}")

    # E8: teams = picks from S16 games 1 and 2
    for region in ("East", "West", "South", "Midwest"):
        e8_key = ("E8", region, "1")
        if e8_key not in index:
            continue
        lineno, e8_row = index[e8_key]
        _, pick_a = get_pick("S16", region, 1)
        _, pick_b = get_pick("S16", region, 2)
        if pick_a is None or pick_b is None:
            continue
        expected = {pick_a, pick_b}
        actual = {e8_row["TeamA"], e8_row["TeamB"]}
        if actual != expected:
            err(lineno, f"E8 {region} game 1: teams {sorted(actual)} != "
                        f"expected S16 winners {sorted(expected)}")

    # ── Check 5: F4 pairings (East vs South = game 1, West vs Midwest = game 2) ─
    _, east_e8_pick  = get_pick("E8", "East",    1)
    _, south_e8_pick = get_pick("E8", "South",   1)
    _, west_e8_pick  = get_pick("E8", "West",    1)
    _, mw_e8_pick    = get_pick("E8", "Midwest", 1)

    f4g1_key = ("F4", "Final Four", "1")
    f4g2_key = ("F4", "Final Four", "2")

    if f4g1_key in index:
        lineno, f4g1 = index[f4g1_key]
        expected = {east_e8_pick, south_e8_pick} - {None}
        actual = {f4g1["TeamA"], f4g1["TeamB"]}
        if expected and actual != expected:
            err(lineno, f"F4 game 1: teams {sorted(actual)} != "
                        f"expected East/South E8 winners {sorted(expected)}")

    if f4g2_key in index:
        lineno, f4g2 = index[f4g2_key]
        expected = {west_e8_pick, mw_e8_pick} - {None}
        actual = {f4g2["TeamA"], f4g2["TeamB"]}
        if expected and actual != expected:
            err(lineno, f"F4 game 2: teams {sorted(actual)} != "
                        f"expected West/Midwest E8 winners {sorted(expected)}")

    # ── Check 6: NCG teams = F4 picks ─────────────────────────────────────────
    _, f4g1_pick = get_pick("F4", "Final Four", 1)
    _, f4g2_pick = get_pick("F4", "Final Four", 2)

    ncg_key = ("NCG", "Championship", "1")
    if ncg_key in index:
        lineno, ncg_row = index[ncg_key]
        expected = {f4g1_pick, f4g2_pick} - {None}
        actual = {ncg_row["TeamA"], ncg_row["TeamB"]}
        if expected and actual != expected:
            err(lineno, f"NCG: teams {sorted(actual)} != "
                        f"expected F4 winners {sorted(expected)}")

    # ── Check 7: Champ TeamA = NCG pick ───────────────────────────────────────
    _, ncg_pick = get_pick("NCG", "Championship", 1)
    champ_key = ("Champ", "Winner", "1")
    if champ_key in index:
        lineno, champ_row = index[champ_key]
        if ncg_pick and champ_row["TeamA"] != ncg_pick:
            err(lineno, f"Champ TeamA '{champ_row['TeamA']}' != NCG pick '{ncg_pick}'")

    return errors

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    official_r64 = load_official_r64()
    brackets = parse_brackets(PICKS_PATH)

    print(f"Found {len(brackets)} bracket(s) in {PICKS_PATH.name}\n")

    total_errors = 0
    for b_idx, rows in brackets:
        errors = validate_bracket(b_idx, rows, official_r64)
        if errors:
            print(f"Bracket {b_idx}: {len(errors)} error(s)")
            for e in errors:
                print(e)
            total_errors += len(errors)
        else:
            print(f"Bracket {b_idx}: OK")

    print(f"\nTotal errors: {total_errors}")
    sys.exit(1 if total_errors else 0)

if __name__ == "__main__":
    main()
