import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

NUM = ["x", "y", "dist", "angle", "kp_length", "kp_angle", "kp_x", "kp_y", "gk_x", "gk_y", "gk_dist_goal",
       "gk_dist_line", "n_def_tri", "n_att_tri", "nearest_def"]
CAT = ["body_part", "shot_type", "technique", "play_pattern", "kp_height"]
BOOL = ["under_pressure", "first_time", "one_on_one", "open_goal", "follows_dribble", "aerial_won", "redirect",
        "kp_cross", "kp_through", "kp_cutback", "kp_switch", "has_freeze"]
# excluded: outcome, end_location (leak), deflected (known only after contact), sb_xg (benchmark only)

SEED = 7
TRAIN_LAST_WEEK = 26


def load() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "shots.parquet")
    for c in BOOL:
        df[c] = df[c].fillna(False).astype(float)
    for c in CAT:
        df[c] = df[c].fillna("none").astype(str)
    return df


def logistic(cols_num, cols_cat=(), cols_bool=()):
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), list(cols_num))]
    if cols_cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cols_cat)))
    if cols_bool:
        parts.append(("bool", "passthrough", list(cols_bool)))
    return Pipeline([("prep", ColumnTransformer(parts)), ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def to_lgb_frame(df: pd.DataFrame, cat_levels: dict) -> pd.DataFrame:
    X = df[NUM + BOOL + CAT].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    return X


def fit_lgb(train: pd.DataFrame):
    levels = {c: sorted(train[c].unique()) for c in CAT}
    X, y, groups = to_lgb_frame(train, levels), train.goal.values, train.match_id.values
    best, best_ll = None, 9.0
    for leaves in (4, 8):
        for n in (150, 300):
            ll = []
            for tr, va in GroupKFold(n_splits=3).split(X, y, groups):
                m = lgb.LGBMClassifier(num_leaves=leaves, n_estimators=n, learning_rate=0.03, min_child_samples=40,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                                       random_state=SEED, verbose=-1)
                m.fit(X.iloc[tr], y[tr])
                ll.append(log_loss(y[va], m.predict_proba(X.iloc[va])[:, 1]))
            if np.mean(ll) < best_ll:
                best_ll, best = np.mean(ll), (leaves, n)
    model = lgb.LGBMClassifier(num_leaves=best[0], n_estimators=best[1], learning_rate=0.03, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                               random_state=SEED, verbose=-1)
    model.fit(X, y)
    return model, levels, best, best_ll


def predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, dict]:
    base = train.goal.mean()
    preds = {"base_rate": np.full(len(test), base)}
    m1 = logistic(["dist"]).fit(train, train.goal)
    preds["logit_distance"] = m1.predict_proba(test)[:, 1]
    m2 = logistic(["dist", "angle"]).fit(train, train.goal)
    preds["logit_dist_angle"] = m2.predict_proba(test)[:, 1]
    m3 = logistic(NUM, CAT, BOOL).fit(train, train.goal)
    preds["logit_full"] = m3.predict_proba(test)[:, 1]
    gbm, levels, best, cv_ll = fit_lgb(train)
    preds["lightgbm"] = gbm.predict_proba(to_lgb_frame(test, levels))[:, 1]
    preds["statsbomb_xg"] = test.sb_xg.values
    info = {"lgb_params": best, "lgb_cv_logloss": cv_ll}
    return preds, info


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p) if len(set(p)) > 1 else 0.5,
            "sum_xg": p.sum(), "goals": int(y.sum())}


def bootstrap_delta(test: pd.DataFrame, pa, pb, n=1000) -> tuple[float, float, float]:
    """log-loss(a) - log-loss(b), resampling whole matches; negative = a better."""
    y = test.goal.values
    la = -(y * np.log(np.clip(pa, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pa, 1e-6, 1)))
    lb = -(y * np.log(np.clip(pb, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - pb, 1e-6, 1)))
    d = pd.DataFrame({"m": test.match_id.values, "d": la - lb}).groupby("m").d.agg(["sum", "count"])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    s, c = d["sum"].values, d["count"].values
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float((la - lb).mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate(name: str, train: pd.DataFrame, test: pd.DataFrame, rows: list, calib: dict, by: str | None = None) -> None:
    preds, info = predictions(train, test)
    print(f"\n=== {name}: train {len(train)} shots, test {len(test)} shots; lgb {info['lgb_params']} ===")
    if by:  # the same predictions, scored separately for each group of the test set (e.g. each tournament)
        print(f"  per {by} (no penalties): shots, goals | xG sums: lightgbm, provider | log loss: lightgbm, provider, distance+angle")
        for grp in test[by].unique():
            mask = ((test[by] == grp) & (test.shot_type != "Penalty")).values
            y = test.goal.values[mask]
            m = {k: metrics(y, preds[k][mask]) for k in ("lightgbm", "statsbomb_xg", "logit_dist_angle")}
            print(f"  {grp:15s} {mask.sum():5d} {int(y.sum()):4d} | {m['lightgbm']['sum_xg']:7.1f} {m['statsbomb_xg']['sum_xg']:7.1f}"
                  f" | {m['lightgbm']['log_loss']:.4f} {m['statsbomb_xg']['log_loss']:.4f} {m['logit_dist_angle']['log_loss']:.4f}")
            for model, p in preds.items():
                rows.append({"split": f"{name} | {grp}", "subset": "open_play_no_pens", "model": model, **metrics(y, p[mask])})
    for subset, mask in (("all", np.ones(len(test), bool)), ("open_play_no_pens", (test.shot_type != "Penalty").values)):
        t = test[mask]
        for model, p in preds.items():
            r = {"split": name, "subset": subset, "model": model, **metrics(t.goal.values, p[mask])}
            rows.append(r)
    calib[name] = (test.goal.values, preds)
    for a, b in (("lightgbm", "logit_dist_angle"), ("lightgbm", "logit_full"), ("lightgbm", "statsbomb_xg"),
                 ("logit_full", "statsbomb_xg")):
        mask = (test.shot_type != "Penalty").values
        m, lo, hi = bootstrap_delta(test[mask], preds[a][mask], preds[b][mask])
        print(f"  delta logloss {a} - {b} (no pens): {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        rows.append({"split": name, "subset": "delta_no_pens", "model": f"{a} - {b}", "log_loss": m,
                     "brier": lo, "auc": hi, "sum_xg": np.nan, "goals": np.nan})


def reliability_plot(calib: dict) -> None:
    fig, axes = plt.subplots(1, len(calib), figsize=(6.5 * len(calib), 5.5), squeeze=False)
    for ax, (name, (y, preds)) in zip(axes[0], calib.items()):
        ax.plot([0, 0.8], [0, 0.8], "k--", lw=1)
        for model in ("logit_dist_angle", "logit_full", "lightgbm", "statsbomb_xg"):
            p = preds[model]
            bins = pd.qcut(p, 10, duplicates="drop")
            g = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True).mean()
            ax.plot(g.p, g.y, marker="o", ms=4, label=model)
        ax.set_title(f"Reliability: {name}")
        ax.set_xlabel("predicted xG")
        ax.set_ylabel("observed goal rate")
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "xg_calibration.png", dpi=130)


def main() -> None:
    all_shots = load()
    df = all_shots[all_shots.kind == "league"]       # every model is fitted on the four league seasons only
    tournaments = all_shots[all_shots.kind == "tournament"]
    print(df.groupby("league").size().to_dict(), "league shots;", len(df))
    rows, calib = [], {}

    train = df[df.match_week <= TRAIN_LAST_WEEK]
    test = df[df.match_week > TRAIN_LAST_WEEK]
    evaluate(f"time_split(weeks<= {TRAIN_LAST_WEEK} -> later)", train, test, rows, calib)

    held = "ligue_1"
    if (df.league == held).sum() > 0:
        evaluate(f"cross_league(train others -> {held})", df[df.league != held], df[df.league == held], rows, calib)

    if len(tournaments):  # does xG learned on club football carry over to World Cups and Euros?
        evaluate("cross_competition(train 4 leagues -> tournaments)", df, tournaments, rows, calib, by="league")

    REPORTS.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "xg_metrics.csv", index=False)
    reliability_plot(calib)
    pd.set_option("display.width", 200)
    print("\n", out[out.subset != "delta_no_pens"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
