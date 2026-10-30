"""
    python run_all.py
    python run_all.py --skip-download                  # reuse data/raw
    python run_all.py --skip-download --team "Leicester City"          # single-club analyses for another club
    python run_all.py --skip-download --clubs "Leicester City,Sevilla" # clubs shown side by side
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# (name, script and its fixed arguments, kind): kind "download" is skipped with --skip-download,
# "club" receives --team/--league, "clubs" receives --clubs/--top
STEPS = [
    ("download match lists", ["download.py", "matches"], "download"),
    ("download events", ["download.py", "events"], "download"),
    ("shot table", ["build_shots.py"], ""),
    ("shot context and score timeline", ["build_context.py"], ""),
    ("team style features", ["build_style.py"], ""),
    ("goalkeeper per shot", ["build_gk.py"], ""),
    ("xG model", ["xg_model.py"], ""),
    ("xPoints and season simulation", ["xpoints.py"], ""),
    ("single club: structure of conceded chances", ["analyze_structure.py"], "club"),
    ("single club: opponent types", ["opponent_types.py"], "club"),
    ("single club: goalkeepers", ["gk_split.py"], "club"),
    ("several clubs side by side", ["compare_clubs.py"], "clubs"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--team")
    ap.add_argument("--league")
    ap.add_argument("--clubs")
    ap.add_argument("--top", type=int)
    ap.add_argument("--pool")
    a = ap.parse_args()
    extra = {"club": [x for k, v in (("--team", a.team), ("--league", a.league)) if v for x in (k, v)],
             "clubs": [x for k, v in (("--clubs", a.clubs), ("--top", a.top), ("--pool", a.pool)) if v is not None for x in (k, str(v))]}
    logs = ROOT / "reports" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for name, args, kind in STEPS:
        if kind == "download" and a.skip_download:
            print(f"[skip] {name}")
            continue
        start = time.time()
        print(f"[run ] {name} ...", flush=True)
        log = logs / (Path(args[0]).stem + (f"_{args[1]}" if len(args) > 1 else "") + ".log")
        with open(log, "w", encoding="utf-8") as fh:
            code = subprocess.run([sys.executable, "-I", "-X", "utf8", str(SRC / args[0]), *args[1:], *extra.get(kind, [])],
                                  stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"[{'ok  ' if code == 0 else 'FAIL'}] {name} in {time.time() - start:.0f}s (log: {log.name})", flush=True)
        if code != 0:
            sys.exit(code)
    print("done")


if __name__ == "__main__":
    main()
