"""Which club the single-club analyses are about: `--team "Leicester City"` (optionally `--league premier_league`)."""
import argparse
import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ("ligue_1", "Toulouse")
LEAGUE_NAMES = {"premier_league": "Premier League", "la_liga": "La Liga", "serie_a": "Serie A", "ligue_1": "Ligue 1"}


def pretty_league(league: str) -> str:
    return LEAGUE_NAMES.get(league, league.replace("_", " ").title())


def slug(team: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", team).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name).strip("_")


def known_teams() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / "shots.parquet", columns=["league", "team"]).drop_duplicates()


def resolve(team: str, league: str | None = None) -> tuple[str, str]:
    teams = known_teams()
    league = league.strip().lower().replace(" ", "_") if league else None
    hit = teams[teams.team.str.lower() == team.strip().lower()]
    if league:
        hit = hit[hit.league == league]
    if len(hit) == 1:
        return hit.iloc[0].league, hit.iloc[0].team
    if len(hit) > 1:
        raise SystemExit(f"'{team}' exists in several leagues ({', '.join(hit.league)}); add --league")
    close = difflib.get_close_matches(team, teams.team.tolist(), n=5, cutoff=0.5)
    where = f" in {league}" if league else ""
    raise SystemExit(f"no team '{team}'{where}" + (f"; did you mean: {', '.join(close)}?" if close else "") +
                     ". Teams are listed in reports/xpoints_table.csv")


def focal_from_argv(argv: list[str] | None = None) -> tuple[str, str]:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--team")
    ap.add_argument("--league")
    args, _ = ap.parse_known_args(argv)
    if not args.team and not args.league:
        return DEFAULT
    if not args.team:
        raise SystemExit("--league needs --team")
    return resolve(args.team, args.league)
