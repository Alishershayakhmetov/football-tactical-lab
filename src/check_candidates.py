import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xpoints import HALF, match_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
mt = match_table()
played = pd.concat([mt[["league", "home"]].rename(columns={"home": "team"}),
                    mt[["league", "away"]].rename(columns={"away": "team"})]).groupby(["league", "team"]).size()
print("teams with fewer than 38 matches in the data:")
print(played[played < 38].to_string())
missing = mt.groupby("league").size()
print("\nmatches per league:", missing.to_dict())

base = pd.read_pickle(ROOT / "data" / "processed" / "xpoints_table.pkl")
t = base.loc[("ligue_1", "Toulouse")]
print("\nToulouse season: pts %d, xpts %.1f, gap %.1f, p=%.3f" % (t.pts, t.xpts, t.gap, t.p_two_sided))

shots = pd.read_parquet(ROOT / "data" / "processed" / "shots.parquet")
xg = pd.read_parquet(ROOT / "data" / "processed" / "shots_xg.parquet")
shots["xg_cal"] = xg.xg_cal.values
tou = mt[(mt.home == "Toulouse") | (mt.away == "Toulouse")].copy()
tou["half"] = tou.week.map(lambda w: "weeks 1-%d" % HALF if w <= HALF else "weeks %d-38" % (HALF + 1))
rows = []
for h, g in tou.groupby("half"):
    ids = set(g.match_id)
    s = shots[shots.match_id.isin(ids)]
    gf = s[s.team == "Toulouse"]
    ga = s[s.team != "Toulouse"]
    rows.append({"half": h, "matches": len(g), "gf": gf.goal.sum(), "xgf": round(gf.xg_cal.sum(), 1),
                 "ga": ga.goal.sum(), "xga": round(ga.xg_cal.sum(), 1),
                 "shots_for": len(gf), "shots_against": len(ga)})
print(pd.DataFrame(rows).to_string(index=False))
