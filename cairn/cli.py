"""Command-line entry point: `cairn predict ...`"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .inference import predict_period, coverage, verify_checkpoints
from .metrics import format_report, summarise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cairn", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="run walk-forward inference and score it")
    p.add_argument("--species", choices=["h2s", "ch4"], default="h2s")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--out", type=Path, default=None, help="directory for CSV/JSON output")
    p.add_argument("--device", default="cpu")

    sub.add_parser("coverage", help="show the released walk-forward window")
    sub.add_parser("verify", help="sha256 all checkpoints against the manifest")

    a = ap.parse_args(argv)

    if a.cmd == "coverage":
        for sp in ("h2s", "ch4"):
            lo, hi = coverage(sp)
            print(f"{sp}: {lo} .. {hi}  (13 weekly checkpoints)")
        return 0

    if a.cmd == "verify":
        ok = verify_checkpoints()
        bad = [k for k, v in ok.items() if not v]
        print(f"checkpoints verified: {sum(ok.values())}/{len(ok)}")
        if bad:
            print("FAILED:", bad)
            return 1
        return 0

    try:
        res = predict_period(a.species, a.start, a.end, device=a.device)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print()
    print(format_report(res, a.species))
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.out / f"{a.species}_predictions.csv", index=False)
        (a.out / f"{a.species}_metrics.json").write_text(json.dumps(summarise(res), indent=1))
        print(f"\nwrote {a.out}/{a.species}_predictions.csv and _metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
