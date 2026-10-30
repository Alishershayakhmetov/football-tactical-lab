import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_structure import FOCAL, load  # noqa: E402
from focal import pretty_league, slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, REPORTS = ROOT / "data" / "processed", ROOT / "reports"
LEAGUE, TEAM = FOCAL
LEAGUE_NAME = pretty_league(LEAGUE)
SEED = 7
RNG = np.random.default_rng(SEED)
FEATURES = ["possession", "long_share", "cross_share", "press", "fast_share", "setpiece_xg_share"]
NAMES = {1: "Possession + pressing", 2: "Direct, fast attacks", 3: "Direct, set-piece reliant"}  # from the centroids


def team_features() -> pd.DataFrame:
    style = pd.read_parquet(PROC / "team_style_events.parquet")[["possession", "long_share", "cross_share", "press"]]
    shots, _ = load()
    np_shots = shots[shots.shot_type != "Penalty"]
    fast = np_shots.assign(f=np_shots.cls.isin(["Fast attack", "Counter"])).groupby(["league", "team"]).f.mean()
    sp = np_shots.assign(w=np_shots.xg * np_shots.cls.isin(["Corner", "Free kick"])).groupby(["league", "team"]).w.sum() \
        / np_shots.groupby(["league", "team"]).xg.sum()
    out = style.join(fast.rename("fast_share")).join(sp.rename("setpiece_xg_share"))
    assert out.notna().all().all() and len(out) == 80
    return out


def fit_labels(X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    return km.labels_, km


def choose_k(X: np.ndarray) -> int:
    print("\n-- choosing the number of opponent types (80 teams, 6 standardised style features) --")
    rows = []
    for k in range(2, 7):
        labels, _ = fit_labels(X, k)
        aris = []
        for _ in range(200):
            idx = RNG.integers(0, len(X), len(X))
            km = KMeans(n_clusters=k, n_init=5, random_state=int(RNG.integers(1e6))).fit(X[idx])
            aris.append(adjusted_rand_score(labels, km.predict(X)))
        rows.append({"k": k, "silhouette": silhouette_score(X, labels), "bootstrap_ARI_mean": np.mean(aris),
                     "bootstrap_ARI_p10": np.percentile(aris, 10), "smallest_cluster": np.bincount(labels).min()})
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))
    cand = t[t.k.isin([3, 4, 5])]
    k = int(cand.sort_values(["bootstrap_ARI_mean", "silhouette"], ascending=False).iloc[0].k)
    print(f"rule fixed in advance: among k = 3, 4, 5 take the most stable by bootstrap ARI -> k = {k}")
    return k


def permutation_p(gap: np.ndarray, labels: np.ndarray, n: int = 20000) -> float:
    def stat(lab):
        overall = gap.mean()
        return sum(len(gap[lab == t]) * (gap[lab == t].mean() - overall) ** 2 for t in np.unique(lab))

    obs = stat(labels)
    perms = np.array([stat(RNG.permutation(labels)) for _ in range(n)])
    return float((perms >= obs).mean())


def main() -> None:
    if LEAGUE not in ("premier_league", "la_liga", "serie_a", "ligue_1"):
        print(f"skipped: style types are fitted on the 80 club teams of the four league seasons; {LEAGUE_NAME} has only "
              f"3-7 matches per team and national teams play a different game")
        return
    pd.set_option("display.width", 220)
    feats = team_features()
    print("correlation of style features:\n", feats[FEATURES].corr().round(2).to_string())
    X = ((feats[FEATURES] - feats[FEATURES].mean()) / feats[FEATURES].std()).values
    k = choose_k(X)
    labels, km = fit_labels(X, k)
    order = np.argsort(-pd.Series(feats.possession.values).groupby(labels).mean().values)  # type 1 = most possession
    remap = {old: new + 1 for new, old in enumerate(order)}
    feats["type"] = [remap[l] for l in labels]

    xp = pd.read_pickle(PROC / "xpoints_table.pkl")
    feats["xpts"] = xp.xpts
    feats["name"] = feats.type.map(lambda t: NAMES.get(t, f"Type {t}") if k == 3 else f"Type {t}")  # names were written for k = 3
    prof = feats.groupby("type")[FEATURES + ["xpts"]].mean()
    prof.insert(0, "teams", feats.groupby("type").size())
    prof.insert(1, "league_teams", feats[feats.index.get_level_values(0) == LEAGUE].groupby("type").size().reindex(prof.index).fillna(0).astype(int))
    print("\n-- type profiles (raw means) --\n", prof.round(3).to_string())
    z = pd.DataFrame(X, index=feats.index, columns=FEATURES).groupby(feats.type).mean()
    print("\n-- type profiles in standard deviations from the 80-team mean --\n", z.round(2).to_string())
    print(f"\n-- {LEAGUE_NAME} teams by type --")
    for t, g in feats[feats.index.get_level_values(0) == LEAGUE].groupby("type"):
        print(f"  type {t}: " + ", ".join(sorted(g.index.get_level_values(1))))
    feats.round(4).to_csv(REPORTS / "team_style_types.csv")

    # ---- results of the chosen club against each type, relative to its league peers ----
    tmx = pd.read_parquet(PROC / "team_match_xpts.parquet")
    tmx = tmx[tmx.league == LEAGUE].copy()
    tmx["gap"] = tmx.pts - tmx.xpts
    types = feats.loc[LEAGUE, "type"]
    tmx["opp_type"] = tmx.opponent.map(types)
    tmx["opp_xpts"] = tmx.opponent.map(xp.loc[LEAGUE, "xpts"])
    tier = pd.qcut(xp.loc[LEAGUE, "xpts"].rank(method="first"), 3, labels=["bottom third", "middle third", "top third"])
    tmx["opp_tier"] = tmx.opponent.map(tier)
    state = pd.read_parquet(PROC / "team_match_state.parquet")[["match_id", "team", "goals_for", "goals_against"]]
    tmx = tmx.merge(state, on=["match_id", "team"])
    f = tmx[tmx.team == TEAM]
    p = tmx[tmx.team != TEAM]
    assert len(f) >= 30 and f.opp_type.notna().all()  # a full league season has 38 matches; a few are missing in Ligue 1

    def table(col: str) -> pd.DataFrame:
        rows = []
        for key, g in f.groupby(col):
            pg = p[p[col] == key].gap.values
            d = g.gap.values
            ia = RNG.integers(0, len(d), size=(4000, len(d)))
            ib = RNG.integers(0, len(pg), size=(4000, len(pg)))
            diff = d[ia].mean(1) - pg[ib].mean(1)
            rows.append({col: key, "matches": len(g), "points": g.pts.sum(), "xpts": g.xpts.sum(), "gap_total": g.gap.sum(),
                         "gap_per_match": g.gap.mean(), "peers_gap_per_match": pg.mean(),
                         "diff_vs_peers": d.mean() - pg.mean(), "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5),
                         "goals_for": g.goals_for.sum(), "goals_against": g.goals_against.sum()})
        return pd.DataFrame(rows)

    tt = table("opp_type")
    print(f"\n-- {TEAM} against each opponent type (gap = points - xPoints per match; peers = other {len(types) - 1} {LEAGUE_NAME} teams) --")
    print(tt.round(2).to_string(index=False))
    pt = permutation_p(f.gap.values, f.opp_type.values)
    print(f"permutation test, do {TEAM}'s gaps differ by opponent type? p = {pt:.3f}")

    tr = table("opp_tier")
    print(f"\n-- the same by opponent strength (thirds of {LEAGUE_NAME} by xPoints), for comparison --")
    print(tr.round(2).to_string(index=False))
    print(f"permutation test by strength tier: p = {permutation_p(f.gap.values, f.opp_tier.astype(str).values):.3f}")
    tt.round(4).to_csv(REPORTS / f"{slug(TEAM)}_by_opponent_type.csv", index=False)

    tp = tmx.groupby("opp_type").gap.mean()
    print(f"\nall {LEAGUE_NAME} teams' mean gap against each type (does xPoints misjudge a type for everybody?):", tp.round(2).to_dict())
    figure(feats, tt, X)


def figure(feats: pd.DataFrame, tt: pd.DataFrame, X: np.ndarray) -> None:
    pcs = PCA(n_components=2, random_state=SEED).fit(X)
    xy = pcs.transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    ax = axes[0]
    palette = ["#1f3a5f", "#c0392b", "#2e8b57", "#d49a00", "#7d5ba6", "#555555"]
    for t in sorted(feats.type.unique()):
        m = (feats.type == t).values
        ax.scatter(xy[m, 0], xy[m, 1], s=40, color=palette[t - 1], alpha=0.75, label=feats.name[feats.type == t].iloc[0])
    l1 = feats.index.get_level_values(0) == LEAGUE
    for i in np.where(l1)[0]:
        team = feats.index[i][1]
        ax.annotate(team.replace("Olympique de Marseille", "Marseille").replace("Stade Malherbe Caen", "Caen"),
                    xy[i], fontsize=7.5, xytext=(3, 3), textcoords="offset points")
    t_i = list(feats.index).index(FOCAL)
    ax.scatter(xy[t_i, 0], xy[t_i, 1], s=220, facecolors="none", edgecolors="k", linewidths=2)
    ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"Playing-style types of 80 teams ({LEAGUE_NAME} labelled, {TEAM} circled)")
    ax.legend(fontsize=8)
    ax = axes[1]
    y = np.arange(len(tt))[::-1]
    ax.hlines(y, tt.ci_lo, tt.ci_hi, color="#b9c2cc", lw=8, zorder=1)
    ax.scatter(tt.diff_vs_peers, y, c="#1f3a5f", s=50, zorder=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{feats.name[feats.type == t].iloc[0]}\n({int(n)} matches)" for t, n in zip(tt.opp_type, tt.matches)])
    ax.set_xlabel(f"{TEAM} points - xPoints per match, minus other {LEAGUE_NAME} teams vs the same type (95% CI)")
    ax.set_title(f"Is {TEAM} different from its peers against some type?")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{slug(TEAM)}_opponent_types.png", dpi=130)


if __name__ == "__main__":
    main()
