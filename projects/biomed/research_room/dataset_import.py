"""Loss-conscious delimited input; transformations are recorded, not guessed."""

from typing import Any, Dict, Tuple

import pandas as pd


def read_dataset(path, *, separator=",", nrows=None, missing_policy="blank_only", description_row=False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if missing_policy not in {"blank_only", "standard_markers"}:
        raise ValueError("Unknown missing-value import policy.")
    frame = pd.read_csv(path, sep=separator, dtype="string", keep_default_na=False,
                        low_memory=False, nrows=nrows)
    if description_row:
        frame = frame.iloc[1:].reset_index(drop=True)
    marker_count = 0
    for column in frame.columns:
        values = frame[column]
        blank = values.str.strip().eq("")
        markers = values.str.strip().str.lower().isin({"na", "n/a", "nan", "null", "none"})
        missing = blank | markers if missing_policy == "standard_markers" else blank
        marker_count += int((missing & ~blank).sum())
        values = values.mask(missing, pd.NA)
        present = values.dropna()
        numeric = pd.to_numeric(values, errors="coerce")
        leading_zero = present.str.match(r"^[+-]?0\d+").any()
        identifier = str(column).lower() in {"id", "subject_id", "patient_id", "sample_id", "participant_id"} or str(column).lower().endswith("_id")
        if len(present) and numeric.notna().sum() == len(present) and not leading_zero and not identifier:
            frame[column] = numeric.astype(float) if values.isna().any() else numeric
        else:
            frame[column] = values
    return frame, {
        "missing_policy": missing_policy,
        "description_row_removed": bool(description_row),
        "converted_marker_cells": marker_count,
        "preserve_identifiers": True,
    }
