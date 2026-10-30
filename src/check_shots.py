from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_parquet(ROOT / "data" / "processed" / "shots.parquet")

print("rows", len(df), "| matches", df.match_id.nunique(), "| teams", df.team.nunique())
print("\nmatches per league:\n", df.groupby("league").match_id.nunique().to_string())
print("\nshots per match: mean %.1f, min %d, max %d" % (
    df.groupby("match_id").size().mean(), df.groupby("match_id").size().min(), df.groupby("match_id").size().max()))

dups = df.duplicated(["match_id", "period", "minute", "second", "player", "x", "y"]).sum()
print("\nexact duplicate shots:", dups)
print("coords outside pitch:", int(((df.x < 0) | (df.x > 120) | (df.y < 0) | (df.y > 80)).sum()))
print("shots from own half (x<60):", int((df.x < 60).sum()), "| from x<40:", int((df.x < 40).sum()))
print("missing player:", int(df.player.isna().sum()), "| missing sb_xg:", int(df.sb_xg.isna().sum()))
print("no freeze frame:", int((~df.has_freeze).sum()), "| by shot type:\n",
      df[~df.has_freeze].shot_type.value_counts().to_string())
print("\noutcomes:\n", df.outcome.value_counts().to_string())
print("\nshot types:\n", df.shot_type.value_counts().to_string())
print("\nsum provider xG vs goals by league:")
print(df.groupby("league").agg(xg=("sb_xg", "sum"), goals=("goal", "sum")).round(1).to_string())
print("\nmatch weeks per league:\n", df.groupby("league").match_week.agg(["min", "max"]).to_string())
