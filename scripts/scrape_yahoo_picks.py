#!/usr/bin/env python3
"""Scrape Yahoo Fantasy bracket pick distribution for all rounds.

Usage:
    python scripts/scrape_yahoo_picks.py

Output:
    data/raw/yahoo_pick_distribution.csv  (Team, Round, Probability)

Rounds scraped (Yahoo label -> our round code):
    Round of 64    -> R32   (% picking team to advance past R64)
    Round of 32    -> S16
    Round of 16    -> E8
    Round of 8     -> F4
    Round of 4     -> NCG
    National Final -> Champ
"""

import asyncio
import csv
import re
import sys
from pathlib import Path

# Mapping from Yahoo's abbreviated names to the full names used in round1_games.csv
TEAM_NAME_MAP: dict[str, str] = {
    "Duke":                  "Duke Blue Devils",
    "Michigan":              "Michigan Wolverines",
    "Arizona":               "Arizona Wildcats",
    "Florida":               "Florida Gators",
    "Houston":               "Houston Cougars",
    "Iowa St.":              "Iowa State Cyclones",
    "Purdue":                "Purdue Boilermakers",
    "Connecticut":           "UConn Huskies",
    "Michigan St.":          "Michigan State Spartans",
    "Gonzaga":               "Gonzaga Bulldogs",
    "Virginia":              "Virginia Cavaliers",
    "Illinois":              "Illinois Fighting Illini",
    "Arkansas":              "Arkansas Razorbacks",
    "St. John's":            "St. John's Red Storm",
    "Kansas":                "Kansas Jayhawks",
    "Alabama":               "Alabama Crimson Tide",
    "Nebraska":              "Nebraska Cornhuskers",
    "Tennessee":             "Tennessee Volunteers",
    "Vanderbilt":            "Vanderbilt Commodores",
    "Wisconsin":             "Wisconsin Badgers",
    "Texas Tech":            "Texas Tech Red Raiders",
    "BYU":                   "BYU Cougars",
    "UCLA":                  "UCLA Bruins",
    "Kentucky":              "Kentucky Wildcats",
    "N. Carolina":           "North Carolina Tar Heels",
    "Miami (FL)":            "Miami Hurricanes",
    "Ohio St.":              "Ohio State Buckeyes",
    "Louisville":            "Louisville Cardinals",
    "Iowa":                  "Iowa Hawkeyes",
    "Georgia":               "Georgia Bulldogs",
    "St. Mary's":            "Saint Mary's Gaels",
    "Villanova":             "Villanova Wildcats",
    "Utah St.":              "Utah State Aggies",
    "Texas A&M":             "Texas A&M Aggies",
    "Saint Louis":           "Saint Louis Billikens",
    "Clemson":               "Clemson Tigers",
    "South Florida":         "South Florida Bulls",
    "TCU":                   "TCU Horned Frogs",
    "Missouri":              "Missouri Tigers",
    "VCU":                   "VCU Rams",
    "Santa Clara":           "Santa Clara Broncos",
    "UCF":                   "UCF Knights",
    "Texas":                 "Texas Longhorns",
    "Akron":                 "Akron Zips",
    "High Point":            "High Point Panthers",
    "McNeese":               "McNeese Cowboys",
    "Miami (OH)":            "Miami (OH) RedHawks",
    "Troy":                  "Troy Trojans",
    "Northern Iowa":         "Northern Iowa Panthers",
    "Hofstra":               "Hofstra Pride",
    "Hawaii":                "Hawai'i Rainbow Warriors",
    "Hawai'i":               "Hawai'i Rainbow Warriors",
    "Pennsylvania":          "Pennsylvania Quakers",
    "California Baptist":    "California Baptist Lancers",
    "Wright St.":            "Wright State Raiders",
    "Kennesaw St.":          "Kennesaw State Owls",
    "Tennessee St.":         "Tennessee State Tigers",
    "N. Dak. St.":           "North Dakota State Bison",
    "Idaho":                 "Idaho Vandals",
    "Furman":                "Furman Paladins",
    "Queens University":     "Queens University Royals",
    "Howard":                "Howard Bison",
    "LIU Brooklyn":          "Long Island University Sharks",
    "Siena":                 "Siena Saints",
    "Prairie View A&M":      "Prairie View A&M Panthers",
}

# Play-in game combined entries — drop these; the actual winners are already in the bracket
PLAYIN_ENTRIES = {"MOH/SMU", "TX/NCST", "PV/LEH", "UMBC/HOW"}

URL = "https://tournament.fantasysports.yahoo.com/mens-basketball-bracket/pickdistribution"

ROUNDS = [
    ("Round of 64",    "R32"),
    ("Round of 32",    "S16"),
    ("Round of 16",    "E8"),
    ("Round of 8",     "F4"),
    ("Round of 4",     "NCG"),
    ("National Final", "Champ"),
]

OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "yahoo_pick_distribution.csv"
SESSION_DIR = Path.home() / ".pw_yahoo_ncaa"


async def scrape() -> None:
    from playwright.async_api import async_playwright

    SESSION_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        # Persistent context preserves Yahoo login across runs
        ctx = await p.chromium.launch_persistent_context(
            str(SESSION_DIR),
            headless=False,
            slow_mo=200,
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()

        print(f"Loading {URL} ...")
        await page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
        await page.wait_for_timeout(3_000)

        # Handle login redirect
        if any(x in page.url for x in ("login", "signin")):
            print("\nYahoo login required.")
            print("Please log in in the browser window, then press Enter here.")
            input()
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(2_000)

        # Dump page HTML once for debugging if needed
        html_dump = Path(__file__).parent.parent / "output" / "yahoo_page_dump.html"

        all_rows: list[dict] = []

        for round_label, round_code in ROUNDS:
            print(f"\n--- {round_label} ({round_code}) ---")

            selected = await _select_round(page, round_label)
            if not selected:
                print(f"  Could not select round '{round_label}' — skipping.")
                continue

            await page.wait_for_timeout(1_200)

            teams = await _extract_teams(page)
            if not teams:
                print(f"  No team data found. Dumping HTML to {html_dump} for inspection.")
                html_dump.parent.mkdir(exist_ok=True)
                html_dump.write_text(await page.content(), encoding="utf-8")
            else:
                for name, prob in teams:
                    all_rows.append({"Team": name, "Round": round_code, "Probability": prob})
                print(f"  {len(teams)} teams found.")

        await ctx.close()

    if not all_rows:
        print("\nNo data scraped. Check output/yahoo_page_dump.html for the page structure.")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Team", "Round", "Probability"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows → {OUT_PATH}")


async def _select_round(page, round_label: str) -> bool:
    """Click the round selector. Returns True if successful."""

    # Strategy 1: <select> element with matching option text
    for sel_el in await page.query_selector_all("select"):
        for opt in await sel_el.query_selector_all("option"):
            text = (await opt.text_content() or "").strip()
            if round_label.lower() in text.lower():
                await sel_el.select_option(label=text)
                return True

    # Strategy 2: clickable button / tab / li containing the round label
    for tag in ("button", "[role='tab']", "[role='option']", "a", "li"):
        for el in await page.query_selector_all(tag):
            text = (await el.text_content() or "").strip()
            if round_label.lower() in text.lower():
                try:
                    await el.click()
                    return True
                except Exception:
                    pass

    # Strategy 3: any element whose text exactly matches
    el = await page.query_selector(f"text='{round_label}'")
    if el:
        await el.click()
        return True

    return False


async def _extract_teams(page) -> list[tuple[str, float]]:
    """Return list of (team_name, probability) from the current page state."""

    results: list[tuple[str, float]] = []

    # Strategy 1: look for rows inside a table or list that contain a % value
    candidate_selectors = [
        "table tbody tr",
        "[data-tst*='pick'] ",
        "[class*='PickDist'] li",
        "[class*='pick-dist'] tr",
        "[class*='pickDist'] tr",
        "[class*='row']",
    ]

    for sel in candidate_selectors:
        els = await page.query_selector_all(sel)
        for el in els:
            text = (await el.text_content() or "").strip()
            pct = _parse_pct(text)
            if pct is None:
                continue
            name = _parse_name(text)
            if name:
                results.append((name, pct))
        if results:
            return _dedupe(results)

    # Strategy 2: JavaScript walk — find leaf-ish elements containing a % sign
    candidates = await page.evaluate("""() => {
        const out = [];
        document.querySelectorAll('*').forEach(el => {
            if (el.children.length > 0) return;
            const t = (el.innerText || '').trim();
            if (t.includes('%') && t.length < 120) out.push(t);
        });
        return out;
    }""")

    # Try to pair adjacent team-name and percentage text nodes
    # Collect all text nodes and look for the pattern: name ... XX.XX%
    all_text = await page.evaluate("""() => {
        const out = [];
        const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        while (walk.nextNode()) {
            const t = walk.currentNode.nodeValue.trim();
            if (t) out.push(t);
        }
        return out;
    }""")

    pct_indices = [i for i, t in enumerate(all_text) if re.search(r"\d+\.?\d*\s*%", t)]
    for idx in pct_indices:
        pct_text = all_text[idx]
        pct = _parse_pct(pct_text)
        if pct is None:
            continue
        # Look back up to 3 text nodes for a team name
        for back in range(1, 4):
            if idx - back < 0:
                break
            candidate = all_text[idx - back].strip()
            if candidate and not re.search(r"\d+\.?\d*\s*%", candidate) and len(candidate) > 3:
                results.append((candidate, pct))
                break

    return _dedupe(results)


def _parse_pct(text: str) -> float | None:
    m = re.search(r"(\d+\.?\d*)\s*%", text)
    if not m:
        return None
    val = float(m.group(1)) / 100.0
    return round(val, 6) if 0 < val <= 1.0 else None


def _parse_name(text: str) -> str:
    # Remove the percentage value
    name = re.sub(r"\d+\.?\d*\s*%", "", text)
    # Strip leading rank ordinal (1st, 2nd, 3rd, 4th ... 68th)
    name = re.sub(r"^\d+(?:st|nd|rd|th)", "", name)
    # Strip trailing seed in parentheses e.g. "(1)", "(12)"
    name = re.sub(r"\(\d+\)\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) < 3 or name.isdigit():
        return ""
    # Drop play-in combined entries
    if name in PLAYIN_ENTRIES:
        return ""
    # Map to canonical full name
    return TEAM_NAME_MAP.get(name, name)


def _dedupe(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen: dict[str, float] = {}
    for name, prob in rows:
        if name not in seen:
            seen[name] = prob
    return list(seen.items())


if __name__ == "__main__":
    asyncio.run(scrape())
