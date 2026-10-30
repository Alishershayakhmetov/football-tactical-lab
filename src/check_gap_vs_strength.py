from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
b = pd.read_pickle(ROOT / "data" / "processed" / "xpoints_table.pkl")
b["xg_diff"] = b.xgf - b.xga
for name, x in (("xPoints", b.xpts), ("xG difference", b.xg_diff)):
    r, p = pearsonr(x, b.gap)
    rho = spearmanr(x, b.gap)[0]
    print(f"gap vs {name}: Pearson r = {r:+.2f} (p={p:.4f}), Spearman = {rho:+.2f}")
    for league, g in b.groupby(level=0):
        print(f"   {league}: r = {pearsonr(g.xpts if name == 'xPoints' else g.xg_diff, g.gap)[0]:+.2f}")

slope, intercept = np.polyfit(b.xpts, b.gap, 1)
print(f"\nlinear fit: gap = {intercept:+.2f} + {slope:+.3f} * xPoints (about {slope * 20:+.1f} points per 20 xPoints)")
b["gap_adj"] = b.gap - (intercept + slope * b.xpts)
print("\nsmallest gap after removing the strength trend:")
print(b.sort_values("gap_adj")[["pts", "xpts", "gap", "gap_adj"]].head(5).round(1).to_string())
print("\nlargest gap after removing the strength trend:")
print(b.sort_values("gap_adj", ascending=False)[["pts", "xpts", "gap", "gap_adj"]].head(5).round(1).to_string())
