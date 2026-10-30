import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shots import canon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUES = ["premier_league", "la_liga", "serie_a", "ligue_1"]
LONG_PASS = 30.0     # yards
OPP_ZONE_X = 72.0    # opponent passes in their own 60% of the pitch (their orientation)
PRESS_ZONE_X = 48.0  # same zone seen from the pressing team (its orientation)


def main() -> None:
    acc = defaultdict(lambda: defaultdict(float))
    for league in LEAGUES:
        for meta in json.loads((RAW / "matches" / f"{league}.json").read_text(encoding="utf-8")):
            home, away = canon(meta["home_team"]["home_team_name"]), canon(meta["away_team"]["away_team_name"])
            events = json.loads((RAW / "events" / f"{meta['match_id']}.json").read_text(encoding="utf-8"))
            for e in events:
                if e["period"] not in (1, 2):
                    continue
                t = e["type"]["name"]
                team = canon(e["team"]["name"])
                opp = away if team == home else home
                if t == "Pass":
                    p = e["pass"]
                    a = acc[(league, team)]
                    a["passes"] += 1
                    acc[(league, opp)]["passes_faced"] += 1
                    a["long_passes"] += p["length"] >= LONG_PASS
                    a["crosses"] += bool(p.get("cross", False))
                    if e["location"][0] <= OPP_ZONE_X:
                        acc[(league, opp)]["opp_zone_passes_faced"] += 1
                elif t == "Pressure" and e.get("location") and e["location"][0] >= PRESS_ZONE_X:
                    acc[(league, team)]["pressures_opp_zone"] += 1
        print(league, "done")
    df = pd.DataFrame(acc).T.fillna(0.0)
    df.index.names = ["league", "team"]
    df["possession"] = df.passes / (df.passes + df.passes_faced)
    df["long_share"] = df.long_passes / df.passes
    df["cross_share"] = df.crosses / df.passes
    df["press"] = df.pressures_opp_zone / df.opp_zone_passes_faced
    df.to_parquet(OUT / "team_style_events.parquet")
    print(df[["possession", "long_share", "cross_share", "press"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
