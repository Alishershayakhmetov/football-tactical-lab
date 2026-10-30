from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "processed" / "shots.parquet")
g = df.groupby(["league", "team"]).match_id.nunique().reset_index(name="matches_with_shots")
for league, sub in g.groupby("league"):
    print(league, len(sub), "teams")
    print(sub.sort_values("matches_with_shots").head(4).to_string(index=False))
    print("... max", sub.matches_with_shots.max())
