import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shots import canon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUES = ["premier_league", "la_liga", "serie_a", "ligue_1", "world_cup_2018", "world_cup_2022", "euro_2020", "euro_2024"]


def lineup_gk(lineup):
    for p in lineup:
        if p["position"]["name"] == "Goalkeeper":
            return p["player"]["id"], p["player"]["name"]
    return None


def main() -> None:
    rows = []
    for league in LEAGUES:
        for meta in json.loads((RAW / "matches" / f"{league}.json").read_text(encoding="utf-8")):
            events = json.loads((RAW / "events" / f"{meta['match_id']}.json").read_text(encoding="utf-8"))
            events.sort(key=lambda e: e["index"])
            cur: dict[str, tuple] = {}
            for e in events:
                team = canon(e["team"]["name"])
                if "tactics" in e:
                    gk = lineup_gk(e["tactics"]["lineup"])
                    if gk:
                        cur[team] = gk
                elif e["type"]["name"] == "Substitution":
                    out_id = e.get("player", {}).get("id")
                    if team in cur and out_id == cur[team][0]:
                        rep = e["substitution"]["replacement"]
                        cur[team] = (rep["id"], rep["name"])
                elif e["type"]["name"] == "Shot" and e["period"] in (1, 2):
                    shooter = team
                    defender = canon(meta["away_team"]["away_team_name"]) if shooter == canon(meta["home_team"]["home_team_name"]) \
                        else canon(meta["home_team"]["home_team_name"])
                    ff = e["shot"].get("freeze_frame") or []
                    fgk = next(((f["player"]["id"], f["player"]["name"]) for f in ff
                                if not f["teammate"] and f["position"]["name"] == "Goalkeeper"), (None, None))
                    lgk = cur.get(defender, (None, None))
                    rows.append({"match_id": meta["match_id"], "period": e["period"], "minute": e["minute"],
                                 "second": e["second"], "player": e.get("player", {}).get("name"),
                                 "x": e["location"][0], "y": e["location"][1],
                                 "gk_ff_id": fgk[0], "gk_ff_name": fgk[1], "gk_lineup_id": lgk[0], "gk_lineup_name": lgk[1]})
        print(league, "done")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "shot_gk.parquet", index=False)
    both = df.dropna(subset=["gk_ff_id", "gk_lineup_id"])
    agree = (both.gk_ff_id == both.gk_lineup_id).mean()
    print(f"{len(df)} shots; freeze-frame GK known for {df.gk_ff_id.notna().sum()}, lineup GK known for {df.gk_lineup_id.notna().sum()}; "
          f"agreement where both known: {agree:.4f} ({int((both.gk_ff_id != both.gk_lineup_id).sum())} disagreements)")
    bad = both[both.gk_ff_id != both.gk_lineup_id]
    if len(bad):
        print(bad.groupby(["gk_ff_name", "gk_lineup_name"]).size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
