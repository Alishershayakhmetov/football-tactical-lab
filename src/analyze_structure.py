import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from focal import focal_from_argv, pretty_league, slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, REPORTS = ROOT / "data" / "processed", ROOT / "reports"
FOCAL = focal_from_argv()  # (league, team); default Toulouse, change with --team / --league
SEED = 7
RNG = np.random.default_rng(SEED)
KEY = ["match_id", "period", "minute", "second", "player", "x", "y"]
CLASS_ORDER = ["Built-up", "Fast attack", "Counter", "Corner", "Free kick", "Throw-in", "Restart from keeper", "Penalty", "Other"]


def reseed() -> None:
    """Every club analysis starts from the same random stream, so a club's numbers do not depend on what ran before it."""
    global RNG
    RNG = np.random.default_rng(SEED)


def source_class(r) -> str:
    if r.shot_type == "Penalty":
        return "Penalty"
    pp = r.play_pattern
    if pp == "From Corner":
        return "Corner"
    if pp == "From Free Kick":
        return "Free kick"
    if pp == "From Throw In":
        return "Throw-in"
    if pp == "From Counter":
        return "Counter"
    if pp in ("From Goal Kick", "From Keeper"):
        return "Restart from keeper"
    if pp == "Regular Play":
        return "Fast attack" if r.poss_seconds <= 12 else "Built-up"
    return "Other"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    shots = pd.read_parquet(PROC / "shots.parquet")
    xg = pd.read_parquet(PROC / "shots_xg.parquet")
    assert (xg.match_id.values == shots.match_id.values).all() and (xg.team.values == shots.team.values).all()
    shots["xg"] = xg.xg_cal.values
    ctx = pd.read_parquet(PROC / "shot_context.parquet").drop(columns=["league"])
    shots = shots.merge(ctx, on=KEY, how="left", validate="one_to_one")
    assert shots.start_x.notna().all()
    shots["cls"] = shots.apply(source_class, axis=1)
    shots["start_zone"] = pd.cut(shots.start_x, [-1, 40, 80, 121], labels=["own third", "middle", "final third"])
    return shots, pd.read_parquet(PROC / "team_match_state.parquet")


def boot_mean_diff(a: np.ndarray, b: np.ndarray, n: int = 4000):
    ia = RNG.integers(0, len(a), size=(n, len(a)))
    ib = RNG.integers(0, len(b), size=(n, len(b)))
    d = a[ia].mean(1) - b[ib].mean(1)
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def per_match_matrix(shots: pd.DataFrame, tm: pd.DataFrame, by: str, value: str, defending: bool) -> pd.DataFrame:
    """rows = team-matches (index league, match_id, team), columns = categories, values = summed `value`."""
    side = "opponent" if defending else "team"
    g = shots.groupby(["league", "match_id", side, by], observed=False)[value].sum().unstack(by, fill_value=0.0)
    g.index.names = ["league", "match_id", "team"]
    idx = pd.MultiIndex.from_frame(tm[["league", "match_id", "team"]])
    return g.reindex(idx, fill_value=0.0)


def is_focal(index: pd.MultiIndex, focal) -> np.ndarray:
    return (index.get_level_values("league") == focal[0]) & (index.get_level_values("team") == focal[1])


def peer_mask(index: pd.MultiIndex, focal) -> np.ndarray:
    return (index.get_level_values("league") == focal[0]) & ~is_focal(index, focal)


def n_peers(tm: pd.DataFrame, focal) -> int:
    return tm[tm.league == focal[0]].team.nunique() - 1


def peers_label(focal) -> str:
    return f"{pretty_league(focal[0])} peers"


def class_table(shots, tm, focal) -> pd.DataFrame:
    xga = per_match_matrix(shots, tm, "cls", "xg", True)
    goals = per_match_matrix(shots, tm, "cls", "goal", True)
    f, p = is_focal(xga.index, focal), peer_mask(xga.index, focal)
    rows = []
    for c in CLASS_ORDER:
        if c not in xga.columns:
            continue
        d, lo, hi = boot_mean_diff(xga.loc[f, c].values, xga.loc[p, c].values)
        rows.append({"class": c, "focal_xga_per_match": xga.loc[f, c].mean(), "peers_xga_per_match": xga.loc[p, c].mean(),
                     "diff": d, "ci_lo": lo, "ci_hi": hi, "focal_xga_total": xga.loc[f, c].sum(),
                     "focal_goals": goals.loc[f, c].sum(), "peers_goal_minus_xg_per_match": (goals.loc[p, c] - xga.loc[p, c]).mean(),
                     "focal_goal_minus_xg_per_match": (goals.loc[f, c] - xga.loc[f, c]).mean()})
    return pd.DataFrame(rows)


def zone_table(shots, tm, focal) -> pd.DataFrame:
    xga = per_match_matrix(shots, tm, "start_zone", "xg", True)
    f, p = is_focal(xga.index, focal), peer_mask(xga.index, focal)
    rows = []
    for c in xga.columns:
        d, lo, hi = boot_mean_diff(xga.loc[f, c].values, xga.loc[p, c].values)
        rows.append({"attack_started_in": c, "focal": xga.loc[f, c].mean(), "peers": xga.loc[p, c].mean(), "diff": d,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def ratio_ci(num: np.ndarray, den: np.ndarray, n: int = 4000):
    idx = RNG.integers(0, len(num), size=(n, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def state_table(shots, tm, focal) -> pd.DataFrame:
    s = shots.copy()
    s["shooter_diff"] = s.score_for - s.score_against
    # state of the DEFENDING team when it concedes: shooter trailing = defender leading
    s["def_state"] = np.where(s.shooter_diff < 0, "lead", np.where(s.shooter_diff > 0, "trail", "tied"))
    xga = per_match_matrix(s, tm, "def_state", "xg", True)
    gc = per_match_matrix(s, tm, "def_state", "goal", True)
    f, p = is_focal(xga.index, focal), peer_mask(xga.index, focal)
    secs = tm.set_index(["league", "match_id", "team"])
    rows = []
    for st in ("lead", "tied", "trail"):
        sec = secs.loc[xga.index, f"sec_{st}"].values
        for who, mask in ((focal[1], f), (peers_label(focal), p)):
            r, lo, hi = ratio_ci(xga.loc[mask, st].values, sec[mask])
            rows.append({"state": st, "team": who, "share_of_time": sec[mask].sum() / secs.loc[xga.index, "total_seconds"].values[mask].sum(),
                         "xga_per_90": r * 5400, "ci_lo": lo * 5400, "ci_hi": hi * 5400,
                         "goals_per_90": gc.loc[mask, st].sum() / sec[mask].sum() * 5400})
    return pd.DataFrame(rows)


def lead_stats(tm: pd.DataFrame, focal) -> dict:
    """Results from winning / losing positions against the club's league peers; the tail follows the observed direction."""
    tm = tm.assign(pts=np.where(tm.goals_for > tm.goals_against, 3, np.where(tm.goals_for == tm.goals_against, 1, 0)))
    f = (tm.league == focal[0]) & (tm.team == focal[1])
    p = (tm.league == focal[0]) & ~f
    rows = []
    for label, col in (("led at some point", "led_ever"), ("trailed at some point", "trailed_ever")):
        tf, tp = tm[f & tm[col]], tm[p & tm[col]]
        if col == "led_ever":
            kf, kp, what = (tf.pts == 3).sum(), (tp.pts == 3).sum(), "won"
        else:
            kf, kp, what = (tf.pts >= 1).sum(), (tp.pts >= 1).sum(), "took a point or more"
        rate = kp / len(tp)
        if len(tf) == 0:
            rows.append({"label": label, "what": what, "n": 0})
            continue
        low = kf / len(tf) <= rate
        pval = binom.cdf(kf, len(tf), rate) if low else binom.sf(kf - 1, len(tf), rate)
        rows.append({"label": label, "what": what, "n": len(tf), "k": int(kf), "n_peers": len(tp), "peer_rate": kp / len(tp),
                     "p_tail": float(pval), "low": bool(low), "points": int(tf.pts.sum()), "expected_points": float(tp.pts.mean() * len(tf))})
    peer_rates = tm[p & tm.led_ever].groupby("team").apply(lambda g: (g.pts == 3).mean())
    own = tm[f & tm.led_ever]
    rank = int((peer_rates > (own.pts == 3).mean()).sum()) + 1 if len(own) else None
    return {"rows": rows, "rank": rank, "n_teams": len(peer_rates) + 1}


def print_lead(res: dict, focal) -> None:
    team, peers = focal[1], peers_label(focal)
    print("\n-- results from leading / trailing positions --")
    for r in res["rows"]:
        if r["n"] == 0:
            print(f"  {r['label']}: {team} never")
            continue
        sign = "<=" if r["low"] else ">="
        print(f"  {r['label']}: {team} {r['n']} matches, {r['what']} {r['k']} ({r['k'] / r['n']:.0%}); {peers} {r['what']} {r['peer_rate']:.0%}"
              f" of {r['n_peers']}; P({sign} {r['k']} of {r['n']} at the peer rate) = {r['p_tail']:.3f}; points {r['points']} vs {r['expected_points']:.1f} expected")
    if res["rank"] is not None:
        print(f"  {team} conversion of leads ranks {res['rank']} of {res['n_teams']} {pretty_league(focal[0])} teams (1 = best)")


def lead_extreme_test(tm: pd.DataFrame, focal) -> dict | None:
    l1 = tm[(tm.league == focal[0]) & tm.led_ever].assign(win=lambda d: d.goals_for > d.goals_against)
    per = l1.groupby("team").win.agg(["sum", "count"])
    if focal[1] not in per.index:
        return None
    rate = per["sum"].sum() / per["count"].sum()
    k, n = per.loc[focal[1], "sum"], per.loc[focal[1], "count"]
    low = k / n <= rate
    tail = (lambda kk, nn: binom.cdf(kk, nn, rate)) if low else (lambda kk, nn: binom.sf(kk - 1, nn, rate))
    obs = tail(k, n)
    sims = np.array([tail(RNG.binomial(per["count"].values, rate), per["count"].values).min() for _ in range(4000)])
    return {"rate": float(rate), "tail": float(obs), "p_extreme": float((sims <= obs).mean()), "n_teams": len(per), "low": bool(low)}


def _psxg_features(ot: pd.DataFrame) -> pd.DataFrame:
    p = np.clip(ot.xg.values, 1e-4, 1 - 1e-4)
    ot["lx"] = np.log(p / (1 - p))
    ot["ey"] = (ot.end_y - 40).abs()
    ot["ez"] = ot.end_z.fillna(0.0)
    return ot


def _psxg_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(num_leaves=4, n_estimators=200, learning_rate=0.03, min_child_samples=40, subsample=0.8,
                              subsample_freq=1, random_state=SEED, verbose=-1)


def psxg(shots: pd.DataFrame) -> pd.DataFrame:
    """Post-shot xG for on-target non-penalty shots. League shots get out-of-fold values; tournament shots are scored
    by a model fitted on all league shots, so the league numbers do not depend on whether tournaments are loaded."""
    ot_all = shots[shots.outcome.isin(["Goal", "Saved", "Saved to Post"]) & (shots.shot_type != "Penalty")].copy()
    ot_all = ot_all[ot_all.end_y.notna()]
    ot = _psxg_features(ot_all[ot_all.kind == "league"].reset_index(drop=True))
    y, groups = ot.goal.values, ot.match_id.values
    base, full = np.zeros(len(ot)), np.zeros(len(ot))
    X = ot[["lx", "ey", "ez"]]
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        base[te] = LogisticRegression().fit(X.iloc[tr][["lx"]], y[tr]).predict_proba(X.iloc[te][["lx"]])[:, 1]
        m = _psxg_model().fit(X.iloc[tr], y[tr])
        full[te] = m.predict_proba(X.iloc[te])[:, 1]
    print(f"\n-- post-shot xG on {len(ot)} on-target non-penalty shots --")
    print(f"  log loss: pre-shot xG only {log_loss(y, base):.4f}  |  + end position (PSxG) {log_loss(y, full):.4f}  | goal rate {y.mean():.3f}")
    ot["psxg"] = full
    t = _psxg_features(ot_all[ot_all.kind == "tournament"].reset_index(drop=True))
    if len(t):
        Xt = t[["lx", "ey", "ez"]]
        t["psxg"] = _psxg_model().fit(X, y).predict_proba(Xt)[:, 1]
        t_base = LogisticRegression().fit(X[["lx"]], y).predict_proba(Xt[["lx"]])[:, 1]
        print(f"  tournaments, {len(t)} on-target shots scored by the league-fitted model: log loss pre-shot xG only "
              f"{log_loss(t.goal.values, t_base):.4f}  |  PSxG {log_loss(t.goal.values, t.psxg.values):.4f}  | goal rate {t.goal.mean():.3f}")
        ot = pd.concat([ot, t], ignore_index=True)
    return ot


def gk_team_table(shots: pd.DataFrame, ot: pd.DataFrame) -> pd.DataFrame:
    """Shot-stopping and its split for every team as defenders; does not depend on the focal club."""
    np_shots = shots[shots.shot_type != "Penalty"]
    g = ot.groupby(["league", "opponent"]).agg(ot_goals=("goal", "sum"), psxg=("psxg", "sum"), ot_xg=("xg", "sum"), n_ot=("goal", "size"))
    g["all_goals"] = np_shots.groupby(["league", "opponent"]).goal.sum()
    g["all_xg"] = np_shots.groupby(["league", "opponent"]).xg.sum()
    g["stopping"] = g.psxg - g.ot_goals             # >0: fewer goals than on-target shots deserved (shot-stopping)
    g["placement"] = g.psxg - g.all_xg              # >0: opponents' shots were better placed than their xG implies
    g["conceded_gap"] = g.all_goals - g.all_xg      # = -stopping + placement
    g.index.names = ["league", "team"]
    return g


def gk_focal(shots: pd.DataFrame, ot: pd.DataFrame, g: pd.DataFrame, focal, verbose: bool, league_kind: bool = True) -> dict:
    team = focal[1]
    own = g.loc[tuple(focal)]
    n_teams = len(g)
    rank = int((g.stopping > own.stopping).sum()) + 1
    if verbose:
        print(f"\n-- shot-stopping (non-penalty): goals prevented = PSxG - goals on target; {n_teams} teams --")
        print(f"  {team}: faced {int(own.n_ot)} on-target shots, conceded {int(own.ot_goals)} vs PSxG {own.psxg:.1f} -> stopping {own.stopping:+.1f} (rank {rank} of {n_teams}, 1 = best)")
        print(f"  goals conceded {int(own.all_goals)} vs xG against {own.all_xg:.1f}: gap {own.conceded_gap:+.1f} = {-own.stopping:+.1f} (shot-stopping) {own.placement:+.1f} (placement of opponents' shots)")
    f = ot[(ot.league == focal[0]) & (ot.opponent == focal[1])]
    per_match = f.groupby("match_id").agg(a=("psxg", "sum"), b=("goal", "sum"))
    allm = shots[(shots.league == focal[0]) & (shots.opponent == focal[1])].match_id.unique()
    pm = per_match.reindex(allm, fill_value=0.0)
    idx = RNG.integers(0, len(pm), size=(4000, len(pm)))
    boots = (pm.a.values[idx].sum(1) - pm.b.values[idx].sum(1))
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    if league_kind:   # halves of a 38-match season; a tournament team has 3-7 matches, so there is nothing to split
        half = ot.assign(h=np.where(ot.match_week <= 19, "h1", "h2")).groupby(["league", "opponent", "h"]).apply(
            lambda d: d.psxg.sum() - d.goal.sum()).unstack("h")
        r, p = pearsonr(half.h1, half.h2)
    else:
        r = p = float("nan")

    # pure-chance benchmark: every on-target shot scores with probability PSxG, nobody has a shot-stopping skill
    keys = ot.groupby(["league", "opponent"]).ngroup().values
    probs = ot.psxg.values
    n_t = keys.max() + 1
    sim_min, sim_max, sim_sd = [], [], []
    for _ in range(2000):
        goals = (RNG.random(len(probs)) < probs).astype(float)
        stop = np.bincount(keys, weights=probs - goals, minlength=n_t)
        sim_min.append(stop.min())
        sim_max.append(stop.max())
        sim_sd.append(stop.std())
    obs = g.stopping.values
    bad = own.stopping <= 0
    p_chance = float((np.array(sim_min) <= own.stopping).mean() if bad else (np.array(sim_max) >= own.stopping).mean())
    p_league = float((np.array(sim_min) <= obs.min()).mean() if bad else (np.array(sim_max) >= obs.max()).mean())
    if verbose:
        print(f"  bootstrap over matches, stopping 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}]")
        if league_kind:
            print(f"  persistence of shot-stopping, weeks 1-19 vs 20-38 across {n_teams} teams: r = {r:+.2f} (p = {p:.2f})")
        if bad:
            print(f"  chance-only benchmark: sd of shot-stopping across teams {np.mean(sim_sd):.2f} (observed {obs.std():.2f}); "
                  f"P(worst of {n_teams} is at least as bad as the league's worst {obs.min():+.1f}) = {p_league:.2f}; "
                  f"as bad as {team} ({own.stopping:+.1f}) = {p_chance:.2f}")
            print(f"  league's worst shot-stopper: {g.stopping.idxmin()}")
        else:
            print(f"  chance-only benchmark: sd of shot-stopping across teams {np.mean(sim_sd):.2f} (observed {obs.std():.2f}); "
                  f"P(best of {n_teams} is at least as good as the league's best {obs.max():+.1f}) = {p_league:.2f}; "
                  f"as good as {team} ({own.stopping:+.1f}) = {p_chance:.2f}")
            print(f"  league's best shot-stopper: {g.stopping.idxmax()}")
    return {"row": own, "rank": rank, "n_teams": n_teams, "ci": ci, "persistence_r": float(r), "p_chance": p_chance, "bad": bool(bad)}


def finishing_table(shots: pd.DataFrame, focal, verbose: bool) -> pd.DataFrame:
    t = shots[(shots.league == focal[0]) & (shots.team == focal[1]) & (shots.shot_type != "Penalty")]
    p = t.groupby("player").agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum")).sort_values("xg", ascending=False)
    p["goals_minus_xg"] = p.goals - p.xg
    if verbose:
        print(f"\n-- {focal[1]} attack, non-penalty: top 8 shooters by xG --")
        print(p.head(8).round(1).to_string())
        print(f"  all players: goals {int(p.goals.sum())} vs xG {p.xg.sum():.1f}; the 8 above account for {p.head(8).goals_minus_xg.sum():+.1f} of {p.goals_minus_xg.sum():+.1f}")
    return p


def figure(ct: pd.DataFrame, g: pd.DataFrame, focal) -> None:
    team, league = focal[1], pretty_league(focal[0])
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    ax = axes[0]
    y = np.arange(len(ct))[::-1]
    ax.hlines(y, ct.ci_lo, ct.ci_hi, color="#b9c2cc", lw=7, zorder=1)
    ax.scatter(ct["diff"], y, c="#1f3a5f", s=45, zorder=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ct["class"])
    ax.set_xlabel(f"xG conceded per match: {team} minus other {league} teams (95% CI)")
    ax.set_title(f"Where {team} concede more or fewer chances")
    ax = axes[1]
    d = g.sort_values("stopping")
    cols = np.where((d.index.get_level_values("league") == focal[0]) & (d.index.get_level_values("team") == focal[1]), "#c0392b", "#8aa0b8")
    ax.bar(np.arange(len(d)), d.stopping, color=cols, width=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks([])
    ax.set_xlabel(f"{len(d)} teams, ordered")
    ax.set_ylabel("goals prevented = PSxG - goals on target")
    ax.set_title(f"Shot-stopping, non-penalty ({team} in red)")
    fig.tight_layout()
    fig.savefig(REPORTS / f"{slug(team)}_structure.png", dpi=130)


def analyze_club(shots: pd.DataFrame, tm: pd.DataFrame, focal, ot: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """All single-club tables for `focal` = (league, team) against the other teams of its league."""
    reseed()
    team, peers = focal[1], peers_label(focal)
    league_kind = shots.loc[shots.league == focal[0], "kind"].iloc[0] == "league"
    f = shots[(shots.league == focal[0]) & (shots.opponent == team)]
    if verbose:
        print(f"{team} conceded {len(f)} shots, {int(f.goal.sum())} goals, xG {f.xg.sum():.1f}")
        if not league_kind:
            n_m = int(((tm.league == focal[0]) & (tm.team == team)).sum())
            print(f"NOTE: {pretty_league(focal[0])} is a tournament: {team} played {n_m} matches (90 minutes only), so every interval is wide; "
                  f"knockout matches exist only for teams that kept winning; shot-stopping is ranked among the teams of this tournament.")
    ct = class_table(shots, tm, focal)
    if verbose:
        print(f"\n-- conceded xG per match by how the attack arose: {team} vs {n_peers(tm, focal)} {peers} (diff with 95% CI) --")
        print(ct.round(3).to_string(index=False))
        print(f"({len(ct)} comparisons: about one CI in twenty excludes 0 by chance alone)")
    zt = zone_table(shots, tm, focal)
    if verbose:
        print("\n-- conceded xG per match by where the attack started (attackers' orientation) --")
        print(zt.round(3).to_string(index=False))
    st = state_table(shots, tm, focal)
    if verbose:
        print("\n-- conceded xG and goals per 90 minutes by score state of the defending team --")
        print(st.round(3).to_string(index=False))
    lead = lead_stats(tm, focal)
    if verbose:
        print_lead(lead, focal)
    extreme = lead_extreme_test(tm, focal)
    if verbose and extreme:
        print(f"\n  lead-conversion extreme test over {extreme['n_teams']} {pretty_league(focal[0])} teams (equal win rate {extreme['rate']:.0%} for everyone): "
              f"{team} tail probability {extreme['tail']:.4f}; P(some team this extreme by chance) = {extreme['p_extreme']:.3f}")
    players = finishing_table(shots, focal, verbose)
    if ot is None:
        ot = psxg(shots)
    # reference group for shot-stopping: all 80 league teams for a club, the teams of the same tournament for a national team
    if league_kind:
        ref_shots, ref_ot = shots[shots.kind == "league"], ot[ot.kind == "league"]
    else:
        ref_shots, ref_ot = shots[shots.league == focal[0]], ot[ot.league == focal[0]]
    g = gk_team_table(ref_shots, ref_ot)
    gk = gk_focal(shots, ref_ot, g, focal, verbose, league_kind)
    return {"class": ct, "zone": zt, "state": st, "lead": lead, "lead_extreme": extreme, "players": players, "gk": gk,
            "gk_table": g, "conceded_shots": len(f), "conceded_goals": int(f.goal.sum()), "conceded_xg": float(f.xg.sum()),
            "league_kind": bool(league_kind)}


def main() -> None:
    shots, tm = load()
    pd.set_option("display.width", 220)
    res = analyze_club(shots, tm, FOCAL)
    slug_ = slug(FOCAL[1])
    res["class"].round(4).to_csv(REPORTS / f"{slug_}_conceded_by_class.csv", index=False)
    res["state"].round(4).to_csv(REPORTS / f"{slug_}_score_state.csv", index=False)
    shot_stopping_file = "shot_stopping_80_teams.csv" if res["league_kind"] else f"shot_stopping_{FOCAL[0]}.csv"
    res["gk_table"].round(3).to_csv(REPORTS / shot_stopping_file)
    figure(res["class"], res["gk_table"], FOCAL)


if __name__ == "__main__":
    main()
