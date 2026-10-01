# CAIRN — inference release

Reference implementation accompanying *"Meteorology-driven Causal Nowcasting of Fugitive Landfill Emissions from Measured Coupling Timescales."*
Preprint: [arXiv:2608.14254](https://arxiv.org/abs/2608.14254) (doi:[10.48550/arXiv.2608.14254](https://doi.org/10.48550/arXiv.2608.14254)).

This repository reproduces the paper's walk-forward H₂S and CH₄ High-class predictions from the released model checkpoints. **It contains inference code only** — no training code, no hyperparameter search, and no community odour-report records.

```bash
pip install -r requirements.txt
python -m cairn.cli predict --species h2s --start 2024-10-01 --end 2024-12-30 --out results/
```

```
CAIRN walk-forward inference — H2S @ MMF9
  timesteps evaluated : 8,040
  High-class support  : 872

  POOLED (Table 2 convention)
    TP 539   FP 612   FN 333   TN 6556
    F1-High        0.5329
    Precision-High 0.4683
    Recall-High    0.6181

  UNWEIGHTED MEAN OVER WEEKS (Supplementary Table S13 convention)
    F1-High        0.5280
```

These are the values reported in the manuscript, and the test suite checks them.

## What it reproduces

| Paper item | Reproduced |
|---|---|
| Fig. 6a — per-timestep outcomes, rolling metrics | ✅ |
| Fig. 6b — ROC and reliability, against the matched benchmark | ✅ |
| Table 2 — pooled walk-forward metrics | ✅ |
| Supplementary Table S13 — per-week H₂S | ✅ |
| Supplementary Table S14 — H₂S vs CH₄ per-week | ✅ |

Verify with `pytest tests/ -v`. The main test compares every predicted `proba_high` against `reference/`, at a tolerance of 1e-5.

Analyses that rest on the community odour-report record are **not** reproducible from this release, because that record is withheld as personal data (see **Data**). The causal (MSTE), phenomenology and feature-attribution analyses are outside the scope of this inference package.

## Selection protocol

Each weekly model stops on an **inner temporal probe** drawn from the seven days preceding its evaluation week, so no checkpoint is selected on the block it is then scored against. This is the protocol the manuscript reports.

The CH₄ model is the one exception: it selects its checkpoint on the evaluation week. Supplementary Table S14 of the manuscript states this, and the cross-species comparison it reports applies evaluation-week selection to both species so that the two columns share a protocol. CH₄ numbers from this release are therefore not directly comparable with the H₂S figures above.

## Two aggregation conventions

The paper reports F1-High under two rules, which give different numbers. Both are printed:

- **Pooled** — confusion counts summed over all 8,040 timesteps, then one metric. Table 2 → **0.533**
- **Unweighted mean** — per-week metric, then a plain mean over 13 weeks. Supplementary Table S13 → **0.528**

The pooled value is the larger because the weeks with least High-class support are also the weakest, and an unweighted mean gives them disproportionate influence.

## How the walk-forward protocol works

The protocol trains a **separate model for each weekly block**, so there are 13 checkpoints per species rather than one model. `predict_period` routes each timestamp to the checkpoint that owns it. A single-model run would not reproduce the paper.

Per week: build the 16-feature vector from raw meteorology → reserve 383 steps (95.75 h) of context before the first prediction → drop rows with any missing feature → standardise with that week's stored scaler → predict → take the **argmax** of the three-class softmax.

The argmax detail matters: thresholding `P(High) > 0.5` gives a different confusion matrix from the one the paper reports.

### Coverage

Checkpoints exist only for **2024-10-01 → 2024-12-30**. Requests outside that window are refused rather than silently served by a model that was never trained for those weeks:

```
$ python -m cairn.cli predict --species h2s --start 2025-01-01 --end 2025-01-07
error: requested 2025-01-01..2025-01-07 is outside the released walk-forward
coverage 2024-10-01..2024-12-30.
```

## Data

`data/MMF9_15min_2023-09_2024-12.parquet` — 46,134 rows of 15-minute meteorology (WD, WS, TEMP, Pressure) plus H₂S and CH₄ concentrations at the principal receptor, 2023-09-01 to 2024-12-31. The pre-evaluation span is included so the per-week scalers can be independently re-derived.

**Community odour-report records are excluded from this release.** They are personal data and are not required for any result reproduced here. The exclusion is enforced by a test that checks both the columns *and* the parquet file-level metadata — dropping the column alone would leave metadata keys recording that the data existed.

The monitoring site is identified only by station ID.

## Scalers

The training pipeline fits a `StandardScaler` per week and does not persist it, so `models/scalers.json` ships the reconstructed parameters (mean and scale per feature per week per species), rebuilt by replaying each week's training-window selection. Shipping them means inference does not depend on the training window being present.

## Layout

```
cairn/          model.py, features.py, labeling.py   (from the research codebase)
                inference.py, metrics.py, cli.py     (this release)
models/         h2s_mmf9/, ch4_mmf9/ — 13 checkpoints each
                scalers.json, MANIFEST.json (sha256 + fold boundaries)
data/           the release parquet
reference/      the test targets
tests/          test_reproduction.py
```

`python -m cairn.cli verify` re-hashes all 26 checkpoints against the manifest.

## Citation

Please cite the accompanying paper, available as a preprint at [arXiv:2608.14254](https://arxiv.org/abs/2608.14254) (doi:[10.48550/arXiv.2608.14254](https://doi.org/10.48550/arXiv.2608.14254)). Full citation metadata is in `CITATION.cff`.
