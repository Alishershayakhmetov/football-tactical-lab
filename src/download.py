import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
BASE = f"https://raw.githubusercontent.com/statsbomb/open-data/{COMMIT}/data"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# (competition_id, season_id) -> label; only full-league seasons exist for 2015/16
LEAGUES = {
    (2, 27): "premier_league",
    (11, 27): "la_liga",
    (12, 27): "serie_a",
    (7, 27): "ligue_1",
}
# whole tournaments (64 / 64 / 51 / 51 matches), added after the four leagues
TOURNAMENTS = {
    (43, 3): "world_cup_2018",
    (43, 106): "world_cup_2022",
    (55, 43): "euro_2020",
    (55, 282): "euro_2024",
}
COMPETITIONS = {**LEAGUES, **TOURNAMENTS}


def fetch(rel: str, dest: Path, retries: int = 4) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE}/{rel}", timeout=60)
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            dest.write_bytes(r.content)
            return "ok"
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return "fail"


def match_ids(labels: list[str]) -> list[int]:
    ids = []
    for (comp, season), label in COMPETITIONS.items():
        if label in labels:
            matches = json.loads((RAW / "matches" / f"{label}.json").read_text(encoding="utf-8"))
            ids += [m["match_id"] for m in matches]
    return ids


def main() -> None:
    step = sys.argv[1]
    if step == "matches":
        for (comp, season), label in COMPETITIONS.items():
            print(label, fetch(f"matches/{comp}/{season}.json", RAW / "matches" / f"{label}.json"))
        return

    labels = sys.argv[2:] or list(COMPETITIONS.values())
    ids = match_ids(labels)
    jobs = [(f"events/{i}.json", RAW / "events" / f"{i}.json") for i in ids]
    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for status in pool.map(lambda j: fetch(*j), jobs):
            counts[status] = counts.get(status, 0) + 1
    print(labels, len(ids), counts)


if __name__ == "__main__":
    main()
