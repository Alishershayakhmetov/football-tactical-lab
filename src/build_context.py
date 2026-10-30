import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shots import canon  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUES = ["premier_league", "la_liga", "serie_a", "ligue_1"]
TOURNAMENTS = ["world_cup_2018", "world_cup_2022", "euro_2020", "euro_2024"]


def seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def process_match(league: str, meta: dict):
    mid = meta["match_id"]
    events = json.loads((RAW / "events" / f"{mid}.json").read_text(encoding="utf-8"))
    events.sort(key=lambda e: e["index"])
    home, away = canon(meta["home_team"]["home_team_name"]), canon(meta["away_team"]["away_team_name"])

    period_len = {}
    for e in events:
        if e["period"] in (1, 2):
            period_len[e["period"]] = max(period_len.get(e["period"], 0.0), seconds(e["timestamp"]))

    def elapsed(e) -> float:
        t = seconds(e["timestamp"])
        return t if e["period"] == 1 else period_len.get(1, 0.0) + t

    total = period_len.get(1, 0.0) + period_len.get(2, 0.0)

    goals = []  # (event index, elapsed, scoring team)
    for e in events:
        if e["period"] not in (1, 2):
            continue
        if e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal":
            goals.append((e["index"], elapsed(e), canon(e["team"]["name"])))
        elif e["type"]["name"] == "Own Goal For":
            goals.append((e["index"], elapsed(e), canon(e["team"]["name"])))

    # goals in extra time (periods 3-4) are in the official score but outside the regulation-time timeline
    extra = {home: 0, away: 0}
    for e in events:
        if e["period"] in (3, 4) and ((e["type"]["name"] == "Shot" and e["shot"]["outcome"]["name"] == "Goal")
                                      or e["type"]["name"] == "Own Goal For"):
            extra[canon(e["team"]["name"])] += 1

    by_poss = {}
    for pos, e in enumerate(events):
        by_poss.setdefault(e["possession"], []).append(pos)

    shots = []
    for pos, e in enumerate(events):
        if e["type"]["name"] != "Shot" or e["period"] not in (1, 2):
            continue
        team = canon(e["team"]["name"])
        opp = away if team == home else home
        pteam = canon(e["possession_team"]["name"])
        members = by_poss[e["possession"]]
        start = next((events[p] for p in members if events[p].get("location")), e)
        sx = start["location"][0] if canon(start["team"]["name"]) == pteam else 120 - start["location"][0]
        n_pass = sum(1 for p in members if p < pos and events[p]["type"]["name"] == "Pass" and canon(events[p]["team"]["name"]) == pteam)
        sf = sum(1 for g in goals if g[0] < e["index"] and g[2] == team)
        sa = sum(1 for g in goals if g[0] < e["index"] and g[2] == opp)
        end = e["shot"].get("end_location", [])
        shots.append({
            "match_id": mid, "period": e["period"], "minute": e["minute"], "second": e["second"],
            "player": e.get("player", {}).get("name"), "x": e["location"][0], "y": e["location"][1],
            "start_x": sx, "start_type": start["type"]["name"], "poss_seconds": elapsed(e) - elapsed(start),
            "poss_passes": n_pass, "score_for": sf, "score_against": sa, "elapsed": elapsed(e),
            "end_y": end[1] if len(end) > 1 else np.nan, "end_z": end[2] if len(end) > 2 else np.nan,
        })

    states = []
    for team, opp in ((home, away), (away, home)):
        t_prev, diff = 0.0, 0
        sec = {"lead": 0.0, "tied": 0.0, "trail": 0.0}
        led = trailed = False
        max_lead = max_trail = 0
        for _, t, scorer in sorted(goals, key=lambda g: g[0]):
            sec["lead" if diff > 0 else "trail" if diff < 0 else "tied"] += t - t_prev
            t_prev = t
            diff += 1 if scorer == team else -1
            led |= diff > 0
            trailed |= diff < 0
            max_lead, max_trail = max(max_lead, diff), max(max_trail, -diff)
        sec["lead" if diff > 0 else "trail" if diff < 0 else "tied"] += total - t_prev
        gf = sum(1 for g in goals if g[2] == team)
        ga = sum(1 for g in goals if g[2] == opp)
        states.append({"league": league, "match_id": mid, "week": meta["match_week"], "team": team, "opponent": opp,
                       "is_home": team == home, "goals_for": gf, "goals_against": ga, "total_seconds": total,
                       "sec_lead": sec["lead"], "sec_tied": sec["tied"], "sec_trail": sec["trail"],
                       "led_ever": led, "trailed_ever": trailed, "max_lead": max_lead, "max_trail": max_trail,
                       "official_for": meta["home_score"] if team == home else meta["away_score"],
                       "official_against": meta["away_score"] if team == home else meta["home_score"],
                       "extra_time_for": extra[team], "extra_time_against": extra[opp]})
    return shots, states


def main() -> None:
    shots, states = [], []
    for league in LEAGUES + TOURNAMENTS:
        for meta in json.loads((RAW / "matches" / f"{league}.json").read_text(encoding="utf-8")):
            s, st = process_match(league, meta)
            for r in s:
                r["league"] = league
            shots += s
            states += st
        print(league, "done")
    pd.DataFrame(shots).to_parquet(OUT / "shot_context.parquet", index=False)
    ts = pd.DataFrame(states)
    ts.to_parquet(OUT / "team_match_state.parquet", index=False)
    # regulation-time timeline must equal the official score minus extra-time goals
    bad = ts[(ts.goals_for != ts.official_for - ts.extra_time_for) | (ts.goals_against != ts.official_against - ts.extra_time_against)]
    print(len(shots), "shot rows;", len(ts), "team-matches;", int((ts.extra_time_for + ts.extra_time_against > 0).sum()),
          "with extra-time goals; timeline score != official score minus extra time in", len(bad), "team-matches")
    if len(bad):
        print(bad[["league", "match_id", "team", "goals_for", "official_for", "extra_time_for", "goals_against",
                   "official_against", "extra_time_against"]].head(10).to_string())


if __name__ == "__main__":
    main()
