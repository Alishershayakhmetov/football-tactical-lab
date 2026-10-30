import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_structure import FOCAL, KEY, load, psxg  # noqa: E402
from focal import slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, REPORTS = ROOT / "data" / "processed", ROOT / "reports"
TEAM = FOCAL[1]
SEED = 7
RNG = np.random.default_rng(SEED)
MIN_SHOTS = 50  # on-target non-penalty shots to count a goalkeeper as a regular


def goalkeeper_table(ot: pd.DataFrame) -> pd.DataFrame:
    ot = ot.assign(var=ot.psxg * (1 - ot.psxg))
    g = ot.groupby(["gk_lineup_id", "gk_lineup_name", "league", "opponent"]).agg(
        matches=("match_id", "nunique"), n=("goal", "size"), goals=("goal", "sum"), psxg=("psxg", "sum"), var=("var", "sum")).reset_index()
    g = g.rename(columns={"opponent": "team"})
    g["stopping"] = g.psxg - g.goals
    g["rate100"] = g.stopping / g.n * 100
    g["z"] = g.stopping / np.sqrt(g["var"])
    g["se100"] = np.sqrt(g["var"]) / g.n * 100
    return g


def random_effects(g: pd.DataFrame):
    """DerSimonian-Laird: how much of the spread between goalkeepers exceeds pure shot luck."""
    r = g.rate100.values
    v = g.se100.values ** 2
    w = 1 / v
    rbar = (w * r).sum() / w.sum()
    q = (w * (r - rbar) ** 2).sum()
    tau2 = max(0.0, (q - (len(r) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    return rbar, tau2, q, len(r) - 1


def main() -> None:
    pd.set_option("display.width", 220)
    shots, _ = load()
    ot = psxg(shots)
    gk = pd.read_parquet(PROC / "shot_gk.parquet")[KEY + ["gk_lineup_id", "gk_lineup_name"]]
    ot = ot.merge(gk, on=KEY, how="left", validate="one_to_one")
    assert ot.gk_lineup_id.notna().all()

    # skill statistics (regulars, spread, persistence) come from the four league seasons only; a tournament goalkeeper faces
    # 10-30 on-target shots, so national-team goalkeepers are judged against those league estimates
    ot_l = ot[ot.kind == "league"].reset_index(drop=True)
    league_kind = ot.loc[ot.league == FOCAL[0], "kind"].iloc[0] == "league"
    g = goalkeeper_table(ot_l)
    reg = g[g.n >= MIN_SHOTS].copy()
    print(f"\n{len(g)} goalkeepers faced on-target shots; {len(reg)} faced at least {MIN_SHOTS}")

    g_focal = goalkeeper_table(ot[ot.league == FOCAL[0]])
    tou = g_focal[g_focal.team == FOCAL[1]].sort_values("n", ascending=False)
    print(f"\n-- {TEAM} goalkeepers (non-penalty on-target shots) --")
    print(tou[["gk_lineup_name", "matches", "n", "goals", "psxg", "stopping", "rate100", "se100", "z"]].round(2).to_string(index=False))

    rbar, tau2, q, df = random_effects(reg)
    print(f"\n-- spread between regular goalkeepers --\nmean {rbar:+.2f} goals per 100 on-target shots; Q = {q:.1f} on {df} df "
          f"(Q close to df means pure luck); true-skill sd (tau) = {np.sqrt(tau2):.2f} goals per 100 shots")
    reg["shrunk100"] = np.nan
    reg["shrunk_lo"] = np.nan
    reg["shrunk_hi"] = np.nan
    v = reg.se100.values ** 2
    b = tau2 / (tau2 + v)
    reg["shrunk100"] = rbar + b * (reg.rate100.values - rbar)
    sd = np.sqrt(b * v)
    reg["shrunk_lo"], reg["shrunk_hi"] = reg.shrunk100 - 1.96 * sd, reg.shrunk100 + 1.96 * sd
    reg["rank_raw"] = reg.rate100.rank(method="min").astype(int)

    print(f"\n-- {TEAM} regulars: raw vs shrunk towards the league mean --")
    tr = reg[(reg.league == FOCAL[0]) & (reg.team == FOCAL[1])]
    if tr.empty:
        print(f"  none: no {TEAM} goalkeeper faced {MIN_SHOTS} on-target shots; shrunk towards the league mean using the league estimates:")
        vt = tou.se100.values ** 2
        bt = tau2 / (tau2 + vt)
        s = rbar + bt * (tou.rate100.values - rbar)
        sdt = np.sqrt(bt * vt)
        print(pd.DataFrame({"gk_lineup_name": tou.gk_lineup_name.values, "n": tou.n.values, "rate100": tou.rate100.values,
                            "shrunk100": s, "shrunk_lo": s - 1.96 * sdt, "shrunk_hi": s + 1.96 * sdt}).round(2).to_string(index=False))
    else:
        print(tr[["gk_lineup_name", "n", "rate100", "rank_raw", "shrunk100", "shrunk_lo", "shrunk_hi"]].round(2).to_string(index=False))
    print(f"(rank 1 = worst of {len(reg)} regular goalkeepers)")
    worst = reg.sort_values("rate100").head(5)
    print("\nfive worst raw rates among regulars:")
    print(worst[["gk_lineup_name", "team", "n", "rate100", "z", "shrunk100"]].round(2).to_string(index=False))

    # chance-only benchmark at goalkeeper level
    keys = ot_l.groupby("gk_lineup_id").ngroup().values
    ids = ot_l.groupby("gk_lineup_id").size()
    reg_mask = (ids >= MIN_SHOTS).values
    probs = ot_l.psxg.values
    nk = keys.max() + 1
    mins, maxs = [], []
    for _ in range(2000):
        goals = (RNG.random(len(probs)) < probs).astype(float)
        stop = np.bincount(keys, weights=probs - goals, minlength=nk)
        zs = stop / np.sqrt(np.bincount(keys, weights=probs * (1 - probs), minlength=nk))
        mins.append(zs[reg_mask].min())
        maxs.append(zs[reg_mask].max())
    main_gk = tou.iloc[0]  # the club's goalkeeper with the most on-target shots faced
    if not league_kind:
        print(f"\nchance-only comparison with the league regulars skipped: {main_gk.gk_lineup_name} faced {int(main_gk.n)} on-target shots "
              f"(z = {main_gk.z:+.2f}), far fewer than a league regular")
    elif main_gk.z <= 0:
        print(f"\nchance-only: P(the worst of {len(reg)} regulars has z <= {main_gk.z:.2f}, {TEAM}'s main goalkeeper) = "
              f"{(np.array(mins) <= main_gk.z).mean():.2f}; P(z <= {reg.z.min():.2f}, the actual worst) = {(np.array(mins) <= reg.z.min()).mean():.2f}")
    else:
        print(f"\nchance-only: P(the best of {len(reg)} regulars has z >= {main_gk.z:.2f}, {TEAM}'s main goalkeeper) = "
              f"{(np.array(maxs) >= main_gk.z).mean():.2f}; P(z >= {reg.z.max():.2f}, the actual best) = {(np.array(maxs) >= reg.z.max()).mean():.2f}")

    # persistence between halves of the season, goalkeepers with enough shots in both halves
    ot["half"] = np.where(ot.match_week <= 19, "h1", "h2")
    ot_l["half"] = np.where(ot_l.match_week <= 19, "h1", "h2")
    h = ot_l.groupby(["gk_lineup_id", "half"]).agg(n=("goal", "size"), s=("psxg", "sum"), gl=("goal", "sum")).unstack("half")
    keep = (h[("n", "h1")] >= 25) & (h[("n", "h2")] >= 25)
    r1 = (h[("s", "h1")] - h[("gl", "h1")]) / h[("n", "h1")] * 100
    r2 = (h[("s", "h2")] - h[("gl", "h2")]) / h[("n", "h2")] * 100
    r, p = pearsonr(r1[keep], r2[keep])
    print(f"\npersistence across {int(keep.sum())} goalkeepers with >= 25 shots in each half: r = {r:+.2f} (p = {p:.2f})")

    # the club by half of the season (a national team: by tournament stage), per goalkeeper
    split = "half" if league_kind else "stage"
    print(f"\n{TEAM} by {'half of the season' if league_kind else 'tournament stage'}:")
    tt = ot[(ot.league == FOCAL[0]) & (ot.opponent == FOCAL[1])]
    print(tt.groupby([split, "gk_lineup_name"]).agg(matches=("match_id", "nunique"), n=("goal", "size"), goals=("goal", "sum"),
                                                    psxg=("psxg", "sum")).assign(stopping=lambda d: d.psxg - d.goals).round(1).to_string())

    reg.round(3).to_csv(REPORTS / "gk_shot_stopping.csv", index=False)
    if league_kind:   # the figure shows the league regulars with the club highlighted
        figure(reg, r1[keep], r2[keep], r)


def figure(reg: pd.DataFrame, r1: pd.Series, r2: pd.Series, r: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    ax = axes[0]
    d = reg.sort_values("rate100").reset_index(drop=True)
    is_tou = (d.league == FOCAL[0]) & (d.team == FOCAL[1])
    ax.errorbar(d.index, d.rate100, yerr=1.96 * d.se100, fmt="none", ecolor="#c9d1d9", lw=1.2, zorder=1)
    ax.scatter(d.index[~is_tou], d.rate100[~is_tou], s=14, color="#5b7a99", zorder=2)
    ax.scatter(d.index[is_tou], d.rate100[is_tou], s=60, color="#c0392b", zorder=3, label=f"{TEAM} goalkeepers")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(d)} regular goalkeepers, ordered (bars = 95% range from shot luck alone)")
    ax.set_ylabel("goals prevented per 100 on-target shots (PSxG - goals)")
    ax.set_title("Shot-stopping per goalkeeper, 2015/16")
    ax.legend()
    ax = axes[1]
    ax.scatter(r1, r2, s=28, color="#5b7a99", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("weeks 1-19 (goals prevented per 100 shots)")
    ax.set_ylabel("weeks 20-38")
    ax.set_title(f"Does shot-stopping persist? r = {r:+.2f} over {len(r1)} goalkeepers")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{slug(TEAM)}_goalkeepers.png", dpi=130)


if __name__ == "__main__":
    main()
