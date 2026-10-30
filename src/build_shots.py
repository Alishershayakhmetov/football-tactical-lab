import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

LEAGUES = ["premier_league", "la_liga", "serie_a", "ligue_1"]
TOURNAMENTS = ["world_cup_2018", "world_cup_2022", "euro_2020", "euro_2024"]  # appended after the leagues, never mixed into model fits
GOAL_X, POST_LO, POST_HI = 120.0, 36.0, 44.0


def angle_at(x: float, y: float) -> float:
    ax, ay = GOAL_X - x, POST_LO - y
    bx, by = GOAL_X - x, POST_HI - y
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na == 0 or nb == 0:
        return math.pi
    return math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb))))


def in_triangle(p, a, b, c) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1, d2, d3 = sign(p, a, b), sign(p, b, c), sign(p, c, a)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def dist_to_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def freeze_features(shot_xy, frame) -> dict:
    out = {"has_freeze": frame is not None, "gk_x": None, "gk_y": None, "gk_dist_goal": None,
           "gk_dist_line": None, "n_def_tri": None, "n_att_tri": None, "nearest_def": None}
    if not frame:
        return out
    a, b = (GOAL_X, POST_LO), (GOAL_X, POST_HI)
    nearest = None
    n_def = n_att = 0
    for f in frame:
        loc = f["location"]
        if f["teammate"]:
            n_att += in_triangle(loc, shot_xy, a, b)
            continue
        if f["position"]["name"] == "Goalkeeper":
            out["gk_x"], out["gk_y"] = loc[0], loc[1]
            out["gk_dist_goal"] = math.hypot(GOAL_X - loc[0], 40 - loc[1])
            out["gk_dist_line"] = dist_to_segment(loc, shot_xy, (GOAL_X, 40.0))
            continue
        n_def += in_triangle(loc, shot_xy, a, b)
        d = math.hypot(loc[0] - shot_xy[0], loc[1] - shot_xy[1])
        nearest = d if nearest is None else min(nearest, d)
    out["n_def_tri"], out["n_att_tri"], out["nearest_def"] = n_def, n_att, nearest
    return out


# the same club appears under two names in Ligue 1 2015/16 (Caen 9+29 matches, Marseille 11+27)
TEAM_ALIASES = {"Caen": "Stade Malherbe Caen", "Marseille": "Olympique de Marseille"}


def canon(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def match_rows(league: str, meta: dict, kind: str = "league") -> list[dict]:
    mid = meta["match_id"]
    events = json.loads((RAW / "events" / f"{mid}.json").read_text(encoding="utf-8"))
    passes = {e["id"]: e for e in events if e["type"]["name"] == "Pass"}
    home = canon(meta["home_team"]["home_team_name"])
    away = canon(meta["away_team"]["away_team_name"])
    rows = []
    for e in events:
        # regulation time only: knockout matches also have extra time (periods 3-4) and shoot-outs (period 5)
        if e["type"]["name"] != "Shot" or e["period"] not in (1, 2):
            continue
        s = e["shot"]
        x, y = e["location"][0], e["location"][1]
        team = canon(e["team"]["name"])
        row = {
            "league": league, "kind": kind, "stage": meta["competition_stage"]["name"],
            "match_id": mid, "match_week": meta["match_week"], "match_date": meta["match_date"],
            "team": team, "opponent": away if team == home else home, "is_home": team == home,
            "player": e.get("player", {}).get("name"), "period": e["period"], "minute": e["minute"],
            "second": e["second"], "possession": e["possession"], "play_pattern": e["play_pattern"]["name"],
            "under_pressure": bool(e.get("under_pressure", False)),
            "x": x, "y": y, "dist": math.hypot(GOAL_X - x, 40 - y), "angle": angle_at(x, y),
            "body_part": s["body_part"]["name"], "shot_type": s["type"]["name"], "technique": s["technique"]["name"],
            "first_time": bool(s.get("first_time", False)), "one_on_one": bool(s.get("one_on_one", False)),
            "open_goal": bool(s.get("open_goal", False)), "follows_dribble": bool(s.get("follows_dribble", False)),
            "aerial_won": bool(s.get("aerial_won", False)), "redirect": bool(s.get("redirect", False)),
            "deflected": bool(s.get("deflected", False)),
            "outcome": s["outcome"]["name"], "goal": int(s["outcome"]["name"] == "Goal"),
            "sb_xg": s.get("statsbomb_xg"),
        }
        kp = passes.get(s.get("key_pass_id"))
        if kp:
            p = kp["pass"]
            row.update({
                "kp_height": p["height"]["name"], "kp_length": p["length"], "kp_angle": p["angle"],
                "kp_cross": bool(p.get("cross", False)), "kp_through": bool(p.get("through_ball", False)),
                "kp_cutback": bool(p.get("cut_back", False)), "kp_switch": bool(p.get("switch", False)),
                "kp_x": kp["location"][0], "kp_y": kp["location"][1],
            })
        else:
            row["kp_height"] = "none"
        row.update(freeze_features((x, y), s.get("freeze_frame")))
        rows.append(row)
    return rows


def main() -> None:
    rows = []
    for league, kind in [(c, "league") for c in LEAGUES] + [(c, "tournament") for c in TOURNAMENTS]:
        matches = json.loads((RAW / "matches" / f"{league}.json").read_text(encoding="utf-8"))
        have = [m for m in matches if (RAW / "events" / f"{m['match_id']}.json").exists()]
        print(league, len(have), "of", len(matches), "matches downloaded")
        for m in have:
            rows += match_rows(league, m, kind)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "shots.parquet", index=False)
    print(len(df), "shots,", int(df.goal.sum()), "goals")
    print(df.groupby("league", sort=False).agg(shots=("goal", "size"), goals=("goal", "sum"), sb_xg=("sb_xg", "sum")))
    print("missing sb_xg:", int(df.sb_xg.isna().sum()), "| no freeze frame:", int((~df.has_freeze).sum()))


if __name__ == "__main__":
    main()
