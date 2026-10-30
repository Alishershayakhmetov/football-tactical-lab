"""Several clubs side by side, each measured against the other teams of its own league.

    python src/compare_clubs.py --clubs "Leicester City,Toulouse,Sevilla"
    python src/compare_clubs.py --clubs "premier_league:Chelsea,Torino"     # league prefix if a name is ambiguous
    python src/compare_clubs.py --top 3                                     # default: 3 largest shortfalls and 3 largest surpluses
    python src/compare_clubs.py --pool world_cup_2022 --top 2               # ... among the teams of one tournament
    python src/compare_clubs.py --clubs "world_cup_2022:Morocco,Leicester City"   # national teams and clubs together
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_structure import analyze_club, load, psxg  # noqa: E402
from focal import pretty_league, resolve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROC, REPORTS = ROOT / "data" / "processed", ROOT / "reports"
LEAGUE_LABELS = ("premier_league", "la_liga", "serie_a", "ligue_1")


def parse_clubs(spec: str) -> list[tuple[str, str]]:
    clubs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        league, _, name = item.rpartition(":") if ":" in item else ("", "", item)
        clubs.append(resolve(name, league or None))
    if len(set(clubs)) != len(clubs):
        raise SystemExit("a club is listed twice")
    return clubs


def default_clubs(xp: pd.DataFrame, top: int, pool: str | None = None) -> list[tuple[str, str]]:
    """Largest shortfalls and surpluses among the four leagues, or inside one competition if `pool` names it."""
    lg = xp.index.get_level_values(0)
    pick = xp[lg == pool] if pool else xp[lg.isin(LEAGUE_LABELS)]
    by_gap = pick.sort_values("gap")
    return list(by_gap.head(top).index) + list(by_gap.tail(top).index[::-1])


def club_row(focal, res: dict, xp: pd.DataFrame, style: pd.DataFrame | None) -> dict:
    x = xp.loc[focal]
    gk, lead, ct, pl = res["gk"], res["lead"]["rows"][0], res["class"], res["players"]
    own = gk["row"]
    lg = xp.index.get_level_values(0)
    # p-values are adjusted for having picked the club out of its pool: the 80 league clubs, or the teams of one tournament
    pool = int((lg.isin(LEAGUE_LABELS)).sum()) if focal[0] in LEAGUE_LABELS else int((lg == focal[0]).sum())
    tournament = focal[0] not in LEAGUE_LABELS
    row = {
        "league": pretty_league(focal[0]), "team": focal[1],
        "scope": "group stage, 3 matches" if tournament else "league season",
        "points": int(x.pts), "xpoints": x.xpts, "points_minus_xpoints": x.gap,
        "band95_lo": x.sim_lo - x.xpts, "band95_hi": x.sim_hi - x.xpts,
        "p_two_sided": x.p_two_sided, "pool_size": pool, "p_if_picked_from_pool": min(1.0, x.p_two_sided * pool),
        "rank_by_points_in_league": int(x.rank_actual), "rank_by_xpoints_in_league": int(x.rank_xpts),
        "style_type": None if style is None or focal not in style.index else style.loc[focal, "name"],
        "xga_per_match": ct.focal_xga_per_match.sum(), "league_peers_xga_per_match": ct.peers_xga_per_match.sum(),
        "finishing_goals_minus_xg": pl.goals.sum() - pl.xg.sum(),
        "defence_xga_minus_goals_against": -own.conceded_gap,
        "shot_stopping_psxg_minus_goals": own.stopping, "shot_stopping_ci_lo": gk["ci"][0], "shot_stopping_ci_hi": gk["ci"][1],
        "shot_stopping_rank": gk["rank"], "shot_stopping_pool": gk["n_teams"],
        "matches_led": lead["n"],
    }
    if lead["n"]:
        row.update({"led_and_won": lead["k"], "win_rate_when_led": lead["k"] / lead["n"], "league_peers_win_rate_when_led": lead["peer_rate"],
                    "lead_tail_p": lead["p_tail"], "lead_direction": "below peers" if lead["low"] else "above peers",
                    "lead_p_extreme_in_league": res["lead_extreme"]["p_extreme"] if res["lead_extreme"] else np.nan,
                    "points_from_leads_minus_expected": lead["points"] - lead["expected_points"]})
    return row


def figure(table: pd.DataFrame) -> None:
    n = len(table)
    labels = [f"{t}\n({lg})" for t, lg in zip(table.team, table.league)]
    y = np.arange(n)[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.9 * n + 3.2), sharey=True)

    ax = axes[0]
    ax.hlines(y, table.band95_lo, table.band95_hi, color="#b9c2cc", lw=9, zorder=1)
    outside = (table.points_minus_xpoints < table.band95_lo) | (table.points_minus_xpoints > table.band95_hi)
    ax.scatter(table.points_minus_xpoints, y, c=np.where(outside, "#c0392b", "#1f3a5f"), s=60, zorder=2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("points - xPoints (grey = what shot luck alone gives, 95%)")
    ax.set_title("Season: results vs chances")

    ax = axes[1]
    h = 0.26
    for off, col, name, color in ((h, "finishing_goals_minus_xg", "finishing: goals - xG", "#2e8b57"),
                                  (0, "defence_xga_minus_goals_against", "defence: xG against - goals against", "#1f3a5f"),
                                  (-h, "shot_stopping_psxg_minus_goals", "shot-stopping: PSxG - goals on target", "#d49a00")):
        ax.barh(y + off, table[col], height=h, color=color, label=name)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("goals, non-penalty (positive = better than the chances imply)")
    ax.set_title("Where the goals went")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[2]
    has = table.matches_led.fillna(0) > 0
    ax.scatter(table.league_peers_win_rate_when_led[has], y[has.values], marker="D", s=50, color="#8aa0b8", label="league peers", zorder=1)
    ax.scatter(table.win_rate_when_led[has], y[has.values], s=70, color="#1f3a5f", label="the club", zorder=2)
    for yy, k, nn in zip(y[has.values], table.led_and_won[has], table.matches_led[has]):
        ax.annotate(f"{int(k)}/{int(nn)}", (0.02, yy), xycoords=("axes fraction", "data"), fontsize=9, va="center")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("share of matches won after leading at some point")
    ax.set_title("Holding a lead")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "compare_clubs.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clubs", help='comma separated, e.g. "Leicester City,Toulouse" or "premier_league:Chelsea"')
    ap.add_argument("--top", type=int, default=3, help="without --clubs: how many largest shortfalls and surpluses")
    ap.add_argument("--pool", help="without --clubs: pick them inside one competition, e.g. world_cup_2022 (default: the four leagues)")
    args = ap.parse_args()

    shots, tm = load()
    # league clubs: the whole season; tournament teams: the group stage only (3 matches each), because over all matches the
    # teams that go deeper are biased upward (see xpoints.py)
    xp = pd.concat([pd.read_pickle(PROC / "xpoints_table.pkl"), pd.read_pickle(PROC / "xpoints_table_tournaments.pkl")])
    clubs = parse_clubs(args.clubs) if args.clubs else default_clubs(xp, args.top, args.pool)
    style_path = REPORTS / "team_style_types.csv"
    style = pd.read_csv(style_path).set_index(["league", "team"]) if style_path.exists() else None
    ot = psxg(shots)

    rows, class_rows = [], []
    for focal in clubs:
        print(f"analysing {focal[1]} ({pretty_league(focal[0])}) ...", flush=True)
        res = analyze_club(shots, tm, focal, ot=ot, verbose=False)
        rows.append(club_row(focal, res, xp, style))
        class_rows.append(res["class"].assign(team=focal[1], league=pretty_league(focal[0])))
    table = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    table.round(4).to_csv(REPORTS / "compare_clubs.csv", index=False)
    pd.concat(class_rows).round(4).to_csv(REPORTS / "compare_clubs_classes.csv", index=False)
    figure(table)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    shown = table.set_index("team").drop(columns=["league", "band95_lo", "band95_hi", "led_and_won"], errors="ignore")
    print("\n" + shown.T.map(lambda v: f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)).to_string())
    print("\nSigns: finishing, defence and shot-stopping are all 'positive = better than the chances imply'.")
    print("Each club is compared with the other teams of its own competition. p_if_picked_from_pool multiplies the two-sided p by pool_size")
    print("(80 league clubs, or the teams of one tournament): use it when a club was chosen because its gap was large; with plain p,")
    print("about 5% of teams look 'significant' by chance. It is deliberately conservative; the exact simulated value for the single most")
    print("extreme team of each competition is in the xpoints.py log.")
    if (~table.league.isin([pretty_league(c) for c in LEAGUE_LABELS])).any():
        print("Tournament rows: points - xPoints covers the group stage only (3 matches); every other column uses all the 90-minute data of")
        print("the tournament, so its intervals are wide, and shot-stopping is ranked among that tournament's teams.")


if __name__ == "__main__":
    main()
