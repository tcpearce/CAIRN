"""Metric definitions matching the walk-forward evaluation reported in the manuscript.

Two aggregation rules appear in the paper and they give different numbers:

  pooled      confusion counts summed over all timesteps, then one metric
              -> Table 2
  unweighted  per-week metric, then a plain mean over the 13 weeks
              -> Supplementary Table S13 (the x-bar row)

Both are reported here so either can be checked. The class decision is the
argmax of the softmax, not a threshold on P(High).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def high_class_metrics(y_true_high: np.ndarray, y_pred_high: np.ndarray,
                       proba_high: np.ndarray | None = None) -> Dict[str, float]:
    """Binary High-vs-rest metrics for one block of timesteps."""
    yt = np.asarray(y_true_high).astype(int)
    yp = np.asarray(y_pred_high).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
         "precision_high": prec, "recall_high": rec, "f1_high": f1,
         "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
         "n": int(len(yt)), "n_high": int(yt.sum())}
    if proba_high is not None:
        p = np.asarray(proba_high, dtype=float)
        m["brier_high"] = float(np.mean((p - yt) ** 2))
        eps = 1e-15
        pc = np.clip(p, eps, 1 - eps)
        m["log_loss_high"] = float(-np.mean(yt * np.log(pc) + (1 - yt) * np.log(1 - pc)))
    return m


def per_week(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wk, g in res.groupby("week_idx"):
        m = high_class_metrics(g.is_high_true, g.is_high_pred, g.get("proba_high"))
        m["week_idx"] = int(wk)
        rows.append(m)
    cols = ["week_idx", "n", "n_high", "tp", "fp", "fn", "tn",
            "f1_high", "precision_high", "recall_high", "false_alarm_rate"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


def summarise(res: pd.DataFrame) -> Dict[str, object]:
    pooled = high_class_metrics(res.is_high_true, res.is_high_pred, res.get("proba_high"))
    wk = per_week(res)
    return {
        "pooled": pooled,
        "unweighted_mean_f1_high": float(wk.f1_high.mean()),
        "unweighted_mean_precision_high": float(wk.precision_high.mean()),
        "unweighted_mean_recall_high": float(wk.recall_high.mean()),
        "n_weeks": int(len(wk)),
        "per_week": wk.to_dict(orient="records"),
    }


def format_report(res: pd.DataFrame, species: str) -> str:
    s = summarise(res)
    p = s["pooled"]
    L = [f"CAIRN walk-forward inference — {species.upper()} @ MMF9",
         f"  timesteps evaluated : {p['n']:,}",
         f"  High-class support  : {p['n_high']:,}",
         "",
         "  POOLED (Table 2 convention)",
         f"    TP {p['tp']}   FP {p['fp']}   FN {p['fn']}   TN {p['tn']}",
         f"    F1-High        {p['f1_high']:.4f}",
         f"    Precision-High {p['precision_high']:.4f}",
         f"    Recall-High    {p['recall_high']:.4f}",
         f"    False-alarm    {p['false_alarm_rate']:.4f}"]
    if "brier_high" in p:
        L.append(f"    Brier-High     {p['brier_high']:.4f}")
    L += ["",
          "  UNWEIGHTED MEAN OVER WEEKS (Supplementary Table S13 convention)",
          f"    F1-High        {s['unweighted_mean_f1_high']:.4f}",
          f"    Precision-High {s['unweighted_mean_precision_high']:.4f}",
          f"    Recall-High    {s['unweighted_mean_recall_high']:.4f}"]
    return "\n".join(L)
