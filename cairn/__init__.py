"""CAIRN — Causal-Anchored Inference for Receptor Nowcasting.

Inference-only release accompanying the manuscript. Loads the released
walk-forward checkpoints and reproduces the reported H2S / CH4 High-class
predictions. Contains no training code.
"""
from .inference import predict_period, coverage, verify_checkpoints  # noqa: F401
from .metrics import summarise, per_week, format_report              # noqa: F401

__version__ = "0.1.0"
