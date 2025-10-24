"""Utility functions for geometry and detection processing."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def add_detection_names(
    df: pd.DataFrame,
    *,
    class_col: str = "class_name",
    name_col: str = "detection_name",
) -> pd.DataFrame:
    """Assign sequential names per detection class.

    Each detection receives a unique name composed of its class name and an
    incremental counter (e.g., ``wall_1``, ``wall_2``). Counters are tracked per
    class in the order detections appear in the DataFrame.

    Args:
        df: DataFrame containing a column with class labels.
        class_col: Name of the column containing class labels.
        name_col: Name of the column to store the generated detection names.

    Returns:
        DataFrame including the ``name_col`` with generated detection names. A
        copy of the original DataFrame is returned to avoid mutating the input.
    """
    if df.empty:
        df = df.copy()
        df[name_col] = []
        return df

    if class_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{class_col}' column")

    class_counts: Dict[str, int] = {}
    detection_names = []

    for class_name in df[class_col].astype(str):
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        detection_names.append(f"{class_name}_{class_counts[class_name]}")

    df_with_names = df.copy()
    df_with_names[name_col] = detection_names
    return df_with_names
