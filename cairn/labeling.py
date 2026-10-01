from __future__ import annotations
from typing import Dict
import pandas as pd


def apply_class_bins(df: pd.DataFrame, target_col: str, classes_cfg: Dict) -> pd.Series:
    ser = df[target_col].copy()
    labels = pd.Series(index=ser.index, dtype=object)
    
    # Handle both old (low/medium/high) and new (MATLAB-style) class names
    class_keys = list(classes_cfg.keys())
    
    for class_name, class_def in classes_cfg.items():
        mask = pd.Series(True, index=ser.index)
        
        if "lt" in class_def:
            mask &= (ser < float(class_def["lt"]))
        if "ge" in class_def:
            mask &= (ser >= float(class_def["ge"]))
        
        # Apply the class label where mask is true
        labels[mask] = class_name
    
    return labels.astype("category")
