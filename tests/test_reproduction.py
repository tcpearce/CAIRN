"""Reproduction tests.

These check that the released code reproduces the predictions and metrics
reported in the manuscript. Test 1 compares every predicted probability with
the reference values; the others check the reported metrics, checkpoint
integrity, coverage limits and the exclusion of odour-report data.

    pytest tests/ -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from cairn.inference import predict_period, coverage, verify_checkpoints
from cairn.metrics import summarise, per_week

ROOT = Path(__file__).resolve().parent.parent
FULL = ("2024-10-01", "2024-12-30")


@pytest.fixture(scope="session")
def h2s():
    return predict_period("h2s", *FULL, verbose=False)


# ---------------------------------------------------------------- test 1
def test_proba_high_matches_reference(h2s):
    """The load-bearing test: predictions must match the figures in the paper."""
    ref = pd.read_csv(ROOT / "reference/h2s_timestep_metrics.csv", parse_dates=["datetime"])
    m = ref.merge(h2s[["datetime", "proba_high"]], on="datetime", suffixes=("_ref", "_new"))
    assert len(m) == len(ref), f"joined {len(m)} of {len(ref)} reference rows"
    d = np.abs(m.proba_high_ref - m.proba_high_new)
    assert d.max() < 1e-5, f"max|diff| = {d.max():.3e}"


# ---------------------------------------------------------------- tests 2-4
def test_timestep_count(h2s):
    assert len(h2s) == 8040


def test_high_class_support(h2s):
    assert int(h2s.is_high_true.sum()) == 872


def test_pooled_confusion(h2s):
    p = summarise(h2s)["pooled"]
    assert (p["tp"], p["fp"], p["fn"]) == (539, 612, 333)


# ---------------------------------------------------------------- tests 5-6
def test_pooled_f1_high(h2s):
    """Table 2 convention: pooled counts."""
    assert summarise(h2s)["pooled"]["f1_high"] == pytest.approx(0.533, abs=5e-4)


def test_unweighted_mean_f1_high(h2s):
    """Supplementary Table S13 convention: plain mean over the 13 weeks."""
    s = summarise(h2s)
    assert s["n_weeks"] == 13
    assert s["unweighted_mean_f1_high"] == pytest.approx(0.528, abs=5e-4)


def test_per_week_f1_matches_reference(h2s):
    ref = pd.read_csv(ROOT / "reference/h2s_timestep_metrics.csv", parse_dates=["datetime"])
    rows = []
    for wk, g in ref.groupby("week_idx"):
        tp = int(((g.is_high_pred == 1) & (g.is_high_true == 1)).sum())
        fp = int(((g.is_high_pred == 1) & (g.is_high_true == 0)).sum())
        fn = int(((g.is_high_pred == 0) & (g.is_high_true == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        rows.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    got = per_week(h2s).sort_values("week_idx").f1_high.values
    assert np.allclose(got, rows, atol=1e-6)


# ---------------------------------------------------------------- test 7
def test_feature_count():
    spec = json.loads((ROOT / "models/scalers.json").read_text())
    for sp in ("h2s", "ch4"):
        assert len(spec[sp]["features"]) == 16, f"{sp}: {len(spec[sp]['features'])}"


# ------------------------------------------------- test 8: no complaint data
def test_release_data_carries_no_complaint_information():
    """Dropping the column is not enough — the metadata must be clean too."""
    p = ROOT / "data/MMF9_15min_2023-09_2024-12.parquet"
    banned = ("odour", "odor", "complain", "report")
    cols = [c.lower() for c in pd.read_parquet(p).columns]
    assert not [c for c in cols if any(t in c for t in banned)], cols
    md = pq.ParquetFile(p).schema_arrow.metadata or {}
    keys = [k.decode().lower() for k in md]
    assert not [k for k in keys if any(t in k for t in banned)], keys
    blob = b" ".join(md.values()).decode("utf8", "replace").lower()
    # built at runtime so this file does not itself contain the banned literals
    probes = ["".join(x) for x in (("odour", "_reports"), ("compl", "aint"))]
    assert not any(x in blob for x in probes), "complaint text in metadata"


def test_no_complaint_columns_in_reference_csvs():
    banned = ("odour", "odor", "complain")
    for f in (ROOT / "reference").glob("*.csv"):
        cols = [c.lower() for c in pd.read_csv(f, nrows=0).columns]
        assert not [c for c in cols if any(t in c for t in banned)], f"{f.name}: {cols}"


# ------------------------------------------------- test 9: integrity
def test_checkpoint_integrity():
    ok = verify_checkpoints()
    assert len(ok) == 26
    assert all(ok.values()), [k for k, v in ok.items() if not v]


# ------------------------------------------------- guardrails
def test_out_of_range_request_is_refused():
    with pytest.raises(ValueError, match="outside the released"):
        predict_period("h2s", "2025-01-01", "2025-01-07", verbose=False)


def test_coverage_reported():
    lo, hi = coverage("h2s")
    assert str(lo.date()) == "2024-10-01"
    assert str(hi.date()) == "2024-12-30"


def test_no_training_code_shipped():
    banned = ("train_s4", "walk_forward", "run_experiment", "hyperopt", "optimizer.pt")
    hits = [str(f.relative_to(ROOT)) for f in ROOT.rglob("*")
            if f.is_file() and any(b in f.name for b in banned)]
    assert not hits, hits


# ------------------------------------------------- release hygiene scrub
def test_no_identifying_strings_anywhere():
    """Site names, personal paths and account identifiers must not appear.

    Two subtleties, both learned the hard way:

    * the probes are assembled from fragments so that this file does not
      itself contain the literals and therefore does not flag itself;
    * numeric identifiers are matched only in a path-like context. A bare
      digit string false-positives against float probabilities in the
      reference CSVs -- 0.0035447946283966 contains the account number.
    """
    probes = ["".join(x) for x in (
        ("wall", "eys"), ("silver", "dale"), ("newcastle-", "under-lyme"),
        ("one", "drive"), ("maries", "_way"), ("engageenvironment", "agency"),
        ("c:\\", "users"), ("users/", "44794"), ("users\\", "44794"),
    )]
    hits = []
    for f in ROOT.rglob("*"):
        if not f.is_file() or f.suffix in (".pt", ".parquet", ".pyc"):
            continue
        if ".git" in f.parts or "__pycache__" in f.parts:
            continue
        try:
            low = f.read_text(errors="ignore").lower()
        except Exception:
            continue
        hits += [(str(f.relative_to(ROOT)), w) for w in probes if w in low]
    assert not hits, hits
