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


def add_wall_direction(
    df: pd.DataFrame,
    *,
    class_col: str = "class_name",
    width_col: str = "width",
    height_col: str = "height",
    direction_col: str = "wall_direction",
) -> pd.DataFrame:
    """Annotate wall detections with a direction label.

    A wall is considered horizontal when its width exceeds its height and
    vertical when the height exceeds the width. Non-wall detections receive an
    empty string to keep the column present without implying direction.

    Args:
        df: DataFrame containing detection geometry.
        class_col: Name of the column holding class labels.
        width_col: Name of the width column.
        height_col: Name of the height column.
        direction_col: Name of the output direction column.

    Returns:
        Copy of the DataFrame including the ``direction_col`` column.
    """
    required_columns = {class_col, width_col, height_col}
    missing = required_columns - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"DataFrame must contain columns: {missing_list}")

    if df.empty:
        df_with_direction = df.copy()
        df_with_direction[direction_col] = []
        return df_with_direction

    df_with_direction = df.copy()

    def _determine_direction(row: pd.Series) -> str:
        if str(row[class_col]).lower() != "wall":
            return ""

        width = row[width_col]
        height = row[height_col]

        if width > height:
            return "horizontal"
        if height > width:
            return "vertical"
        return "square"

    df_with_direction[direction_col] = df_with_direction.apply(_determine_direction, axis=1)
    return df_with_direction
