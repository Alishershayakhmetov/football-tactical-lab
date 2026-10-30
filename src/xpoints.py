import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_shots import canon  # noqa: E402
from xg_model import BOOL, CAT, NUM, SEED, load, to_lgb_frame  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW, REPORTS = ROOT / "data" / "raw", ROOT / "reports"
LEAGUES = ["premier_league", "la_liga", "serie_a", "ligue_1"]
TOURNAMENTS = ["world_cup_2018", "world_cup_2022", "euro_2020", "euro_2024"]
MAXG = 12
N_SIMS = 10_000
HALF = 19  # weeks 1-19 vs 20-38


def oof_xg(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold LightGBM xG: each match is predicted by a model that never saw it."""
    levels = {c: sorted(df[c].unique()) for c in CAT}
    X, y = to_lgb_frame(df, levels), df.goal.values
    out = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(X, y, df.match_id.values):
        m = lgb.LGBMClassifier(num_leaves=8, n_estimators=300, learning_rate=0.03, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                               random_state=SEED, verbose=-1)
        m.fit(X.iloc[tr], y[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


def league_calibrate(df: pd.DataFrame, col: str) -> np.ndarray:
    """One logit shift per league so that league xG equals league goals (absorbs the league-wide offset)."""
    out = df[col].values.copy()
    for league, idx in df.groupby("league").indices.items():
        p = np.clip(df[col].values[idx], 1e-6, 1 - 1e-6)
        z, goals = np.log(p / (1 - p)), df.goal.values[idx].sum()
        shift = brentq(lambda d: (1 / (1 + np.exp(-(z + d)))).sum() - goals, -2, 2)
        out[idx] = 1 / (1 + np.exp(-(z + shift)))
    return out


def fit_external(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """xG for `test` from one model fitted on all of `train` (same settings as oof_xg); `test` is never seen in fitting."""
    levels = {c: sorted(train[c].unique()) for c in CAT}
    m = lgb.LGBMClassifier(num_leaves=8, n_estimators=300, learning_rate=0.03, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                           random_state=SEED, verbose=-1)
    m.fit(to_lgb_frame(train, levels), train.goal.values)
    return m.predict_proba(to_lgb_frame(test, levels))[:, 1]


def regulation_scores(mt: pd.DataFrame) -> pd.DataFrame:
    """Score after 90 minutes from the event timeline: the official score of a knockout match may include extra time."""
    st = pd.read_parquet(ROOT / "data" / "processed" / "team_match_state.parquet")[["match_id", "team", "goals_for"]]
    h = st.rename(columns={"team": "home", "goals_for": "hg_reg"})
    a = st.rename(columns={"team": "away", "goals_for": "ag_reg"})
    mt = mt.merge(h, on=["match_id", "home"], how="left", validate="one_to_one").merge(a, on=["match_id", "away"], how="left", validate="one_to_one")
    assert mt.hg_reg.notna().all() and mt.ag_reg.notna().all()
    return mt.assign(hg=mt.hg_reg, ag=mt.ag_reg).drop(columns=["hg_reg", "ag_reg"])


def match_table(labels: list[str], regulation: bool = False) -> pd.DataFrame:
    rows = []
    for league in labels:
        for m in json.loads((RAW / "matches" / f"{league}.json").read_text(encoding="utf-8")):
            rows.append({"league": league, "match_id": m["match_id"], "week": m["match_week"],
                         "stage": m["competition_stage"]["name"],
                         "home": canon(m["home_team"]["home_team_name"]), "away": canon(m["away_team"]["away_team_name"]),
                         "hg": m["home_score"], "ag": m["away_score"]})
    mt = pd.DataFrame(rows)
    if regulation:
        mt = regulation_scores(mt)
    mt["hp"] = np.where(mt.hg > mt.ag, 3, np.where(mt.hg == mt.ag, 1, 0))
    mt["ap"] = np.where(mt.ag > mt.hg, 3, np.where(mt.hg == mt.ag, 1, 0))
    return mt


def poisson_binomial(ps: np.ndarray) -> np.ndarray:
    dp = np.zeros(MAXG + 1)
    dp[0] = 1.0
    for p in ps:
        new = dp * (1 - p)
        new[1:] += dp[:-1] * p
        new[-1] += dp[-1] * p
        dp = new
    return dp


def match_pmfs(df: pd.DataFrame, col: str, mt: pd.DataFrame):
    groups = {k: g[col].values for k, g in df.groupby(["match_id", "team"])}
    empty = np.zeros(MAXG + 1)
    empty[0] = 1.0
    H = np.array([poisson_binomial(groups[(m, h)]) if (m, h) in groups else empty for m, h in zip(mt.match_id, mt.home)])
    A = np.array([poisson_binomial(groups[(m, a)]) if (m, a) in groups else empty for m, a in zip(mt.match_id, mt.away)])
    return H, A


def expected_points(H: np.ndarray, A: np.ndarray):
    M = H[:, :, None] * A[:, None, :]
    pw = np.tril(M, -1).sum((1, 2))
    pd_ = np.einsum("nii->n", M)
    pl = 1 - pw - pd_
    return 3 * pw + pd_, 3 * pl + pd_


def team_frame(mt: pd.DataFrame, xh: np.ndarray, xa: np.ndarray) -> pd.DataFrame:
    h = pd.DataFrame({"league": mt.league, "match_id": mt.match_id, "team": mt.home, "opponent": mt.away,
                      "is_home": True, "week": mt.week, "pts": mt.hp, "xpts": xh})
    a = pd.DataFrame({"league": mt.league, "match_id": mt.match_id, "team": mt.away, "opponent": mt.home,
                      "is_home": False, "week": mt.week, "pts": mt.ap, "xpts": xa})
    return pd.concat([h, a], ignore_index=True)


def simulate(mt: pd.DataFrame, H: np.ndarray, A: np.ndarray, teams: list, seed: int = SEED) -> np.ndarray:
    """Whole-season simulation: sampled points per team, shape (N_SIMS, n_teams)."""
    rng = np.random.default_rng(seed)
    tix = {t: i for i, t in enumerate(teams)}
    Mh = np.zeros((len(mt), len(teams)))
    Ma = np.zeros((len(mt), len(teams)))
    for i, (lg, h, a) in enumerate(zip(mt.league, mt.home, mt.away)):
        Mh[i, tix[(lg, h)]] = 1
        Ma[i, tix[(lg, a)]] = 1
    CH, CA = np.cumsum(H, 1), np.cumsum(A, 1)
    out = np.zeros((N_SIMS, len(teams)))
    chunk = 500
    for s in range(0, N_SIMS, chunk):
        u1 = rng.random((chunk, len(mt)))
        u2 = rng.random((chunk, len(mt)))
        hg = (u1[:, :, None] > CH[None]).sum(-1)
        ag = (u2[:, :, None] > CA[None]).sum(-1)
        ph = 3 * (hg > ag) + (hg == ag)
        pa = 3 * (ag > hg) + (hg == ag)
        out[s:s + chunk] = ph @ Mh + pa @ Ma
    return out


def half_gaps(tf: pd.DataFrame, shots: pd.DataFrame, xcol: str) -> pd.DataFrame:
    sh = shots.assign(week=shots.match_week)
    gf = sh.groupby(["league", "team", "week"]).agg(gf=("goal", "sum"), xgf=(xcol, "sum")).reset_index()
    ga = sh.groupby(["league", "opponent", "week"]).agg(ga=("goal", "sum"), xga=(xcol, "sum")).reset_index() \
        .rename(columns={"opponent": "team"})
    t = tf.merge(gf, on=["league", "team", "week"], how="left").merge(ga, on=["league", "team", "week"], how="left").fillna(0)
    t["half"] = np.where(t.week <= HALF, "h1", "h2")
    g = t.groupby(["league", "team", "half"]).agg(pts=("pts", "sum"), xpts=("xpts", "sum"), gf=("gf", "sum"),
                                                  xgf=("xgf", "sum"), ga=("ga", "sum"), xga=("xga", "sum")).reset_index()
    g["pts_gap"], g["att_gap"], g["def_gap"] = g.pts - g.xpts, g.gf - g.xgf, g.xga - g.ga  # >0 is better for all three
    return g.pivot(index=["league", "team"], columns="half", values=["pts_gap", "att_gap", "def_gap"])


def boot_corr(a: np.ndarray, b: np.ndarray, n=5000):
    rng = np.random.default_rng(SEED)
    r = pearsonr(a, b)[0]
    idx = rng.integers(0, len(a), size=(n, len(a)))
    bs = [pearsonr(a[i], b[i])[0] for i in idx]
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main() -> None:
    all_shots = load()
    shots = all_shots[all_shots.kind == "league"].copy()   # the league analysis below is unchanged; tournaments come after it
    mt = match_table(LEAGUES)
    shots["xg_oof"] = oof_xg(shots)
    shots["xg_cal"] = league_calibrate(shots, "xg_oof")
    print("league totals (goals from shots | oof xG | calibrated xG | provider xG):")
    print(shots.groupby("league").agg(goals=("goal", "sum"), oof=("xg_oof", "sum"), cal=("xg_cal", "sum"),
                                      provider=("sb_xg", "sum")).round(0).to_string())
    off = mt.groupby("league").agg(official=("hg", "sum"), away=("ag", "sum"))
    off["official"] += off.pop("away")
    print("official goals (incl. own goals) vs shot goals:", dict(zip(off.index, off.official - shots.groupby("league").goal.sum().reindex(off.index))))

    variants = {"calibrated": "xg_cal", "raw_oof": "xg_oof", "provider": "sb_xg"}
    res, pm, frames = {}, {}, {}
    for name, col in variants.items():
        H, A = match_pmfs(shots, col, mt)
        xh, xa = expected_points(H, A)
        pm[name] = (H, A)
        frames[name] = team_frame(mt, xh, xa)
        season = frames[name].groupby(["league", "team"]).agg(pts=("pts", "sum"), xpts=("xpts", "sum"))
        season["gap"] = season.pts - season.xpts
        res[name] = season
    teams = list(res["calibrated"].index)
    frames["calibrated"].to_parquet(ROOT / "data" / "processed" / "team_match_xpts.parquet", index=False)

    sims = simulate(mt, *pm["calibrated"], teams)
    base = res["calibrated"]
    base["sim_lo"] = np.percentile(sims, 2.5, axis=0)
    base["sim_hi"] = np.percentile(sims, 97.5, axis=0)
    base["p_two_sided"] = [min(1.0, 2 * min((sims[:, i] <= p).mean(), (sims[:, i] >= p).mean()))
                           for i, p in enumerate(base.pts)]
    base["xpts_raw"] = res["raw_oof"].xpts
    base["xpts_provider"] = res["provider"].xpts
    base["gap_raw"] = res["raw_oof"].gap
    base["gap_provider"] = res["provider"].gap

    g = shots.groupby(["league", "team"]).agg(gf=("goal", "sum"), xgf=("xg_cal", "sum"))
    gc = shots.groupby(["league", "opponent"]).agg(ga=("goal", "sum"), xga=("xg_cal", "sum"))
    gc.index.names = ["league", "team"]
    base = base.join(g).join(gc)
    base["rank_actual"] = base.groupby("league").pts.rank(ascending=False, method="min")
    base["rank_xpts"] = base.groupby("league").xpts.rank(ascending=False, method="min")
    base.round(3).to_csv(REPORTS / "xpoints_table.csv")

    pd.set_option("display.width", 220)
    cols = ["pts", "xpts", "gap", "sim_lo", "sim_hi", "p_two_sided", "gf", "xgf", "ga", "xga", "rank_actual", "rank_xpts"]
    print("\nLargest shortfalls (points - xPoints), calibrated xG:")
    print(base.sort_values("gap")[cols].head(8).round(1).to_string())
    print("\nLargest surpluses:")
    print(base.sort_values("gap", ascending=False)[cols].head(8).round(1).to_string())

    # is the most extreme team more extreme than the best/worst of 80 teams by chance alone?
    xp = base.xpts.values
    sim_gap = sims - xp[None, :]
    obs_min, obs_max = base.gap.min(), base.gap.max()
    p_min = (sim_gap.min(1) <= obs_min).mean()
    p_max = (sim_gap.max(1) >= obs_max).mean()
    print(f"\nextreme test over 80 teams: observed min gap {obs_min:.1f} -> P(chance gives one this low) = {p_min:.3f}; "
          f"observed max gap {obs_max:.1f} -> P = {p_max:.3f}")
    print(f"teams outside their own 95% interval: {int(((base.pts < base.sim_lo) | (base.pts > base.sim_hi)).sum())} of 80 "
          f"(about 4 expected by chance)")
    print("rank agreement of gaps: calibrated vs raw Spearman %.3f, vs provider %.3f" % (
        spearmanr(base.gap, base.gap_raw)[0], spearmanr(base.gap, base.gap_provider)[0]))

    # persistence of over/under-performance between halves of the season
    hg = half_gaps(frames["calibrated"], shots, "xg_cal")
    print("\npersistence across 80 teams, weeks 1-%d vs %d-38 (Pearson r, 95%% bootstrap CI over teams):" % (HALF, HALF + 1))
    for k, label in (("pts_gap", "points - xPoints"), ("att_gap", "goals - xG (finishing)"),
                     ("def_gap", "xGA - goals against (goalkeeping/luck defending)")):
        a, b = hg[(k, "h1")].values, hg[(k, "h2")].values
        r, lo, hi = boot_corr(a, b)
        print(f"  {label}: r = {r:+.2f} [{lo:+.2f}, {hi:+.2f}]")

    plot(base)
    base.to_pickle(ROOT / "data" / "processed" / "xpoints_table.pkl")
    parts = [shots[["match_id", "team", "xg_oof", "xg_cal"]]]
    if (all_shots.kind == "tournament").any():
        parts.append(tournament_xpoints(all_shots, shots))
    # rows stay in the order of shots.parquet: the four leagues first, then the tournaments
    pd.concat(parts, ignore_index=True).to_parquet(ROOT / "data" / "processed" / "shots_xg.parquet", index=False)


def tournament_table(t: pd.DataFrame, mt: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Points vs xPoints per team over the matches in `mt` (all matches, or the group stage only).
    Each tournament is simulated on its own, so the others cannot change its numbers."""
    ts = t[t.match_id.isin(mt.match_id)]
    variants = {"calibrated": "xg_cal", "raw": "xg_oof", "provider": "sb_xg"}
    res, pm, frames = {}, {}, {}
    for name, col in variants.items():
        H, A = match_pmfs(ts, col, mt)
        xh, xa = expected_points(H, A)
        pm[name] = (H, A)
        frames[name] = team_frame(mt, xh, xa)
        season = frames[name].groupby(["league", "team"]).agg(pts=("pts", "sum"), xpts=("xpts", "sum"), matches=("pts", "size"))
        season["gap"] = season.pts - season.xpts
        res[name] = season

    base = res["calibrated"].copy()
    pd.set_option("display.width", 220)
    cols = ["matches", "pts", "xpts", "gap", "sim_lo", "sim_hi", "p_two_sided", "gf", "xgf", "ga", "xga"]
    tables = []
    for label in TOURNAMENTS:
        m = (mt.league == label).values
        teams = [k for k in base.index if k[0] == label]
        sims = simulate(mt[m], pm["calibrated"][0][m], pm["calibrated"][1][m], teams)
        part = base.loc[teams].copy()
        part["sim_lo"] = np.percentile(sims, 2.5, axis=0)
        part["sim_hi"] = np.percentile(sims, 97.5, axis=0)
        part["p_two_sided"] = [min(1.0, 2 * min((sims[:, i] <= p).mean(), (sims[:, i] >= p).mean())) for i, p in enumerate(part.pts)]
        sim_gap = sims - part.xpts.values[None, :]
        p_min = (sim_gap.min(1) <= part.gap.min()).mean()
        p_max = (sim_gap.max(1) >= part.gap.max()).mean()
        out = int(((part.pts < part.sim_lo) | (part.pts > part.sim_hi)).sum())
        print(f"\n-- {scope}: {label}, {int(m.sum())} matches, {len(teams)} teams -- extreme test: biggest shortfall {part.gap.min():.1f} "
              f"(P by chance {p_min:.3f}), biggest surplus {part.gap.max():.1f} (P {p_max:.3f}); outside own 95% band: {out} of "
              f"{len(teams)} (about {0.05 * len(teams):.1f} expected)")
        tables.append(part)
    base = pd.concat(tables)
    base["xpts_raw"], base["xpts_provider"] = res["raw"].xpts, res["provider"].xpts
    base["gap_raw"], base["gap_provider"] = res["raw"].gap, res["provider"].gap
    g = ts.groupby(["league", "team"]).agg(gf=("goal", "sum"), xgf=("xg_cal", "sum"))
    gc = ts.groupby(["league", "opponent"]).agg(ga=("goal", "sum"), xga=("xg_cal", "sum"))
    gc.index.names = ["league", "team"]
    base = base.join(g).join(gc)
    base["rank_actual"] = base.groupby("league").pts.rank(ascending=False, method="min")   # by points in these matches, not the final standing
    base["rank_xpts"] = base.groupby("league").xpts.rank(ascending=False, method="min")
    for label in TOURNAMENTS:
        d = base.loc[label].sort_values("gap")
        print(f"\n{scope}, {label}: largest shortfalls / surpluses (points - xPoints over {int(d.matches.min())}-{int(d.matches.max())} matches)")
        print(pd.concat([d.head(3), d.tail(3)])[cols].round(1).to_string())
    return base, frames["calibrated"]


def tournament_xpoints(all_shots: pd.DataFrame, league_shots: pd.DataFrame) -> pd.DataFrame:
    """World Cups and Euros: xG from a model fitted on the league seasons only, regulation-time (90 min) points."""
    t = all_shots[all_shots.kind == "tournament"].copy()
    t["xg_oof"] = fit_external(league_shots, t)   # column name kept for the shared table; this is not out-of-fold but never saw these shots
    t["xg_cal"] = league_calibrate(t, "xg_oof")   # one logit shift per tournament, as for the leagues
    print("\n\n== tournaments: xG from a model fitted on the four league seasons only ==")
    print("totals per tournament (goals from shots | model xG | calibrated xG | provider xG):")
    print(t.groupby("league", sort=False).agg(goals=("goal", "sum"), model=("xg_oof", "sum"), cal=("xg_cal", "sum"),
                                              provider=("sb_xg", "sum")).round(0).to_string())

    mt = match_table(TOURNAMENTS, regulation=True)
    print("regulation-time points: a knockout match level after 90 minutes counts as a draw (1 point each)")
    base_all, frames_all = tournament_table(t, mt, "ALL MATCHES")
    base_grp, _ = tournament_table(t, mt[mt.stage == "Group Stage"].reset_index(drop=True), "GROUP STAGE")

    # a team plays on only if it keeps winning, so over all matches the number of matches played and the gap move together
    print("\nsurvivorship check, all matches: Spearman correlation of matches played with points - xPoints across teams")
    for label in TOURNAMENTS:
        d = base_all.loc[label]
        rho, p = spearmanr(d.matches, d.gap)
        print(f"  {label}: rho = {rho:+.2f} (p = {p:.3f}, {len(d)} teams)")
    d = base_all
    rho, p = spearmanr(d.matches, d.gap)
    print(f"  all four: rho = {rho:+.2f} (p = {p:.4f}, {len(d)} teams) -> use the group-stage table for comparisons")

    proc = ROOT / "data" / "processed"
    base_grp.round(3).to_csv(REPORTS / "xpoints_table_tournaments.csv")
    base_grp.to_pickle(proc / "xpoints_table_tournaments.pkl")
    base_all.round(3).to_csv(REPORTS / "xpoints_table_tournaments_all_matches.csv")
    base_all.to_pickle(proc / "xpoints_table_tournaments_all_matches.pkl")
    frames_all.to_parquet(proc / "team_match_xpts_tournaments.parquet", index=False)
    plot_tournaments(base_grp, "xpoints_tournaments.png", "group stage, 3 matches per team")
    plot_tournaments(base_all, "xpoints_tournaments_all_matches.png", "all matches: teams that go deeper are biased upward")
    return t[["match_id", "team", "xg_oof", "xg_cal"]]


def plot_tournaments(base: pd.DataFrame, filename: str, subtitle: str) -> None:
    fig, axes = plt.subplots(1, len(TOURNAMENTS), figsize=(5.8 * len(TOURNAMENTS), 9), sharex=True)
    for ax, label in zip(axes, TOURNAMENTS):
        d = base.loc[label].sort_values("gap")
        y = np.arange(len(d))
        lo, hi = d.sim_lo - d.xpts, d.sim_hi - d.xpts
        ax.hlines(y, lo, hi, color="#b9c2cc", lw=5, zorder=1)
        out = (d.gap < lo) | (d.gap > hi)
        ax.scatter(d.gap, y, c=np.where(out, "#c0392b", "#1f3a5f"), s=30, zorder=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index, fontsize=8)
        ax.set_title(label.replace("_", " ").title())
        ax.set_xlabel("points - xPoints (90 minutes)")
    fig.suptitle(f"Tournaments: points minus xPoints, {subtitle}. Grey = what shot luck alone produces (95%); red = outside it", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / filename, dpi=120)


def plot(base: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 8), sharex=True)
    for ax, league in zip(axes, LEAGUES):
        d = base.loc[league].sort_values("gap")
        y = np.arange(len(d))
        lo, hi = d.sim_lo - d.xpts, d.sim_hi - d.xpts
        ax.hlines(y, lo, hi, color="#b9c2cc", lw=6, zorder=1)
        out = (d.gap < lo) | (d.gap > hi)
        ax.scatter(d.gap, y, c=np.where(out, "#c0392b", "#1f3a5f"), s=36, zorder=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index, fontsize=9)
        ax.set_title(league.replace("_", " ").title())
        ax.set_xlabel("points - xPoints")
    fig.suptitle("Points minus xPoints, 2015/16. Grey band = what finishing luck alone produces (95%); red = outside it",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(REPORTS / "xpoints_gap.png", dpi=120)


if __name__ == "__main__":
    main()
