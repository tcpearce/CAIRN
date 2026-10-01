"""Walk-forward inference for the CAIRN dual-pathway S4 nowcaster.

The walk-forward protocol trains a *separate* model for each weekly block, so
reproducing the paper requires routing every timestamp to the checkpoint that
owns it. A single-model run will not reproduce anything.

Per week:
  * inputs are the 16-feature vector built from raw meteorology
  * a 383-step (95.75 h) context reserve precedes the first prediction
  * rows with any missing feature are dropped listwise
  * features are standardised with that week's stored scaler
  * the predicted class is the argmax of the three-class softmax
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import build_features
from .labeling import apply_class_bins
from .model import S4TemporalNowcaster

PKG = Path(__file__).resolve().parent.parent
DATA = PKG / "data/MMF9_15min_2023-09_2024-12.parquet"
MODELS = PKG / "models"

# Fixed for this release — matches the production configuration.
FEATURE_CFG = {
    "include": ["WD", "WS", "TEMP", "Pressure"],
    "engineering": {
        "hour_sin_cos": True, "dow_sin_cos": False, "wd_sin_cos": True,
        "met_derivatives": False, "met_accumulation": False,
        "fourier_seasonality": True, "season_harmonics": 4,
    },
}


def _load_spec(species: str) -> dict:
    spec = json.loads((MODELS / "scalers.json").read_text())
    if species not in spec:
        raise ValueError(f"unknown species {species!r}; available: {sorted(spec)}")
    return spec[species]


def _load_manifest(species: str) -> dict:
    return json.loads((MODELS / "MANIFEST.json").read_text())["species"][species]


def coverage(species: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    wk = _load_manifest(species)["weeks"]
    return (pd.Timestamp(wk["week_00"]["val_start"]),
            pd.Timestamp(wk["week_12"]["val_end"]) + timedelta(days=1) - timedelta(minutes=1))


def prepare_frame(species: str) -> tuple[pd.DataFrame, List[str]]:
    """Load the release parquet and build the model input features."""
    spec = _load_spec(species)
    target = "H2S" if species == "h2s" else "CH4"
    df = pd.read_parquet(DATA)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["Station_ID"] = "MMF9"          # build_features groups on this
    df["label"] = apply_class_bins(df, target, spec["classes"])
    df = build_features(df, FEATURE_CFG, "datetime")
    feat = spec["features"]
    missing = [c for c in feat if c not in df.columns]
    if missing:
        raise RuntimeError(f"feature construction failed, missing: {missing}")
    return df, feat


def predict_period(species: str, start: str, end: str,
                   device: str = "cpu", verbose: bool = True) -> pd.DataFrame:
    """Run walk-forward inference over [start, end]; returns per-timestep results."""
    spec, man = _load_spec(species), _load_manifest(species)
    cov_lo, cov_hi = coverage(species)
    req_lo = pd.Timestamp(start)
    req_hi = pd.Timestamp(end)
    # A bare date on --end means the whole of that day, not midnight; otherwise
    # `--end 2024-12-30` would silently drop that day's 95 timesteps.
    if req_hi == req_hi.normalize():
        req_hi = req_hi + timedelta(days=1) - timedelta(minutes=1)
    if req_lo < cov_lo or req_hi > cov_hi:
        raise ValueError(
            f"requested {req_lo.date()}..{req_hi.date()} is outside the released "
            f"walk-forward coverage {cov_lo.date()}..{cov_hi.date()}.\n"
            f"The release contains 13 weekly checkpoints for that window only; "
            f"predicting outside it would require a model that was never trained "
            f"for those weeks."
        )

    df, feat = prepare_frame(species)
    seq_len = spec["seq_len"]
    ctx = timedelta(minutes=15 * (seq_len - 1))
    target = "H2S" if species == "h2s" else "CH4"
    cls = sorted(spec["classes"].items(), key=lambda kv: kv[1].get("ge", kv[1].get("lt", 0)))
    name2lab = {n: i for i, (n, _) in enumerate(cls)}

    out: List[pd.DataFrame] = []
    for wk in range(13):
        info = man["weeks"][f"week_{wk:02d}"]
        w_lo, w_hi = pd.Timestamp(info["val_start"]), pd.Timestamp(info["val_end"])
        w_end = w_hi + timedelta(days=1) - timedelta(minutes=1)
        if w_end < req_lo or w_lo > req_hi:
            continue

        sl = df[(df.datetime >= w_lo - ctx) & (df.datetime <= w_end)]
        sl = sl[sl["label"].notna() & sl[feat].notna().all(axis=1)]
        if len(sl) <= seq_len:
            continue

        sc = spec["weeks"][f"week_{wk:02d}"]
        X = (sl[feat].values - np.asarray(sc["mean"])) / np.asarray(sc["scale"])

        model = S4TemporalNowcaster.load(MODELS / f"{species}_mmf9/week_{wk:02d}",
                                         map_location=device)
        model.device = device
        if hasattr(model, "model"):
            model.model.to(device).eval()
        proba = model.predict_proba(X)

        n = len(proba)
        tail = sl.iloc[seq_len - 1:][:n]
        res = pd.DataFrame({
            "datetime": tail.datetime.values,
            "week_idx": wk,
            "y_true": tail["label"].map(name2lab).values,
            "y_pred": proba.argmax(axis=1),
            target.lower(): tail[target].values,
        })
        for i, (nm, _) in enumerate(cls):
            res[f"proba_{nm}"] = proba[:, i]
        out.append(res)
        if verbose:
            print(f"  week {wk:02d}  {w_lo.date()}..{w_hi.date()}  {n:>4} predictions")

    if not out:
        raise ValueError("no weekly blocks overlap the requested period")
    res = pd.concat(out, ignore_index=True)
    res = res[(res.datetime >= req_lo) & (res.datetime <= req_hi)].reset_index(drop=True)
    res["is_high_true"] = (res.y_true == 2).astype(int)
    res["is_high_pred"] = (res.y_pred == 2).astype(int)
    return res


def verify_checkpoints() -> Dict[str, bool]:
    """sha256 every shipped checkpoint against MANIFEST.json."""
    import hashlib
    man = json.loads((MODELS / "MANIFEST.json").read_text())["species"]
    ok: Dict[str, bool] = {}
    for sp, meta in man.items():
        for wk, info in meta["weeks"].items():
            p = MODELS / f"{sp}_mmf9/{wk}/s4_temporal_model.pt"
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            ok[f"{sp}/{wk}"] = (h.hexdigest() == info["sha256"])
    return ok
