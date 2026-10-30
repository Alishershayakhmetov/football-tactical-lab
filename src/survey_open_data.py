import collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
COMMIT = "4b73468fc5b0f1950f9f66fada70ad3a4f9327cb"
BASE = f"https://raw.githubusercontent.com/statsbomb/open-data/{COMMIT}/data"


def one(c: dict) -> dict:
    r = requests.get(f"{BASE}/matches/{c['competition_id']}/{c['season_id']}.json", timeout=60)
    m = r.json()
    cnt = collections.Counter()
    for x in m:
        cnt[x["home_team"]["home_team_name"]] += 1
        cnt[x["away_team"]["away_team_name"]] += 1
    top, n = cnt.most_common(1)[0]
    return {"competition": c["competition_name"], "season": c["season_name"], "gender": c["competition_gender"],
            "competition_id": c["competition_id"], "season_id": c["season_id"], "matches": len(m), "teams": len(cnt),
            "top_team": top, "top_team_matches": n, "coverage_share_top_team": round(n / len(m), 2),
            "has_360": c["match_available_360"] is not None}


def main() -> None:
    comps = requests.get(f"{BASE}/competitions.json", timeout=60).json()
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, comps))
    df = pd.DataFrame(rows).sort_values(["gender", "competition", "season"], ascending=[True, True, False])
    # one team in (almost) every match means a single-team extract; a league needs >= 8 teams and nobody in over half of the matches
    df["kind"] = "single-team extract"
    df.loc[(df.teams >= 8) & (df.coverage_share_top_team <= 0.5), "kind"] = "whole competition"
    df.loc[df.matches <= 3, "kind"] = "a few matches"
    (ROOT / "reports").mkdir(exist_ok=True)
    df.to_csv(ROOT / "reports" / "open_data_survey.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(len(df), "competition-seasons")
    print(df.kind.value_counts().to_string())
    cols = ["competition", "season", "gender", "matches", "teams", "has_360"]
    print("\nWHOLE COMPETITIONS:")
    print(df[df.kind == "whole competition"][cols].to_string(index=False))
    print("\nSINGLE-TEAM EXTRACTS (competition, seasons, team):")
    s = df[df.kind == "single-team extract"]
    for (comp, team), g in s.groupby(["competition", "top_team"]):
        print(f"  {comp} | {team}: {len(g)} seasons, {g.season.min()} .. {g.season.max()}, 360 in {int(g.has_360.sum())}")


if __name__ == "__main__":
    main()
