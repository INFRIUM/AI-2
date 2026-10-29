
"""Load provenance-verified measured AI1 outputs for AI2 training.

This loader rejects missing or unapproved data provenance, validates
experiment-level split integrity, and prevents AI1 training/validation
experiments from entering AI2 candidate data.

No dummy or synthetic training source is supported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw" / "measured" / "for_AI2"
SPLIT_PATH = (
    ROOT / "data" / "processed" / "measured"
    / "ai2_experiment_split.csv"
)
LABEL_PATH = DATA_DIR / "experiment_labels.csv"
PROVENANCE_PATH = DATA_DIR / "data_provenance.json"

LABEL_MAP = {
    "D0": "NORMAL",
    "D1": "D1",
    "D2": "D2",
    "D3": "D3",
}


class DataValidationError(ValueError):
    """Raised when source provenance or split integrity is invalid."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required measured data file is missing: {path}"
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(
            f"Cannot read measured data file: {path}"
        ) from exc

    if frame.empty:
        raise DataValidationError(
            f"Measured data file is empty: {path}"
        )

    return frame


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"{source} lacks required columns: {missing}"
        )


def _validate_provenance() -> None:
    """Require a recorded, approved measured-data provenance manifest.

    Manifest approval is a procedural gate. Its contents must be
    independently checked against the actual measurement records.
    """
    if not PROVENANCE_PATH.is_file():
        raise DataValidationError(
            "Measured data provenance manifest is missing: "
            f"{PROVENANCE_PATH}"
        )

    try:
        with PROVENANCE_PATH.open(
            encoding="utf-8"
        ) as handle:
            provenance = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataValidationError(
            "Measured data provenance manifest cannot be read"
        ) from exc

    if not isinstance(provenance, dict):
        raise DataValidationError(
            "Measured data provenance manifest must be an object"
        )

    if provenance.get("data_source") != "MEASURED":
        raise DataValidationError(
            "Only MEASURED source data is allowed"
        )

    if provenance.get("validation_status") != "VALIDATED":
        raise DataValidationError(
            "Measured data has not been validated"
        )

    required_metadata = (
        "dataset_id",
        "measurement_protocol",
        "source_record_reference",
        "validated_by",
        "validated_at",
    )

    for field in required_metadata:
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DataValidationError(
                f"Provenance metadata is missing: {field}"
            )

    for field in (
        "box_anomaly_results.csv",
        "truss_anomaly_results.csv",
        "experiment_labels.csv",
    ):
        if field not in provenance.get("source_files", []):
            raise DataValidationError(
                f"Provenance manifest does not list {field}"
            )


def load_validated_data(bridge_type: str) -> pd.DataFrame:
    """Load measured Window data with validated AI2 split assignments.

    Test rows are returned for split-integrity checking but must not
    be used for model fitting or validation-driven selection.
    """
    if not isinstance(bridge_type, str):
        raise ValueError(
            "bridge_type must be BOX or TRUSS"
        )

    bridge_type = bridge_type.upper()

    if bridge_type not in {"BOX", "TRUSS"}:
        raise ValueError(
            "bridge_type must be BOX or TRUSS"
        )

    _validate_provenance()

    raw = _read_csv(
        DATA_DIR / f"{bridge_type.lower()}_anomaly_results.csv"
    )
    labels = _read_csv(LABEL_PATH)
    split = _read_csv(SPLIT_PATH)

    required_split = {
        "experiment_id",
        "bridge_type",
        "condition_code",
        "damage_type",
        "severity_level",
        "ai1_split",
        "ai2_split",
    }

    required_labels = {
        "experiment_id",
        "bridge_type",
        "damage_type",
        "severity_level",
        "ai1_split",
    }

    required_raw = {
        "experiment_id",
        "bridge_type",
    }

    _require_columns(
        split,
        required_split,
        "AI2 split",
    )
    _require_columns(
        labels,
        required_labels,
        "Experiment labels",
    )
    _require_columns(
        raw,
        required_raw,
        "AI1 Window results",
    )

    if (
        split["experiment_id"].isna().any()
        or split["experiment_id"].duplicated().any()
    ):
        raise DataValidationError(
            "Each experiment_id must occur exactly once "
            "in split data"
        )

    if (
        labels["experiment_id"].isna().any()
        or labels["experiment_id"].duplicated().any()
    ):
        raise DataValidationError(
            "Each experiment_id must occur exactly once "
            "in experiment labels"
        )

    if raw["experiment_id"].isna().any():
        raise DataValidationError(
            "AI1 Window data contains missing experiment_id"
        )

    if not set(split["ai2_split"]).issubset(
        {"train", "val", "test"}
    ):
        raise DataValidationError(
            "Unknown AI2 split label"
        )

    label_fields = [
        "experiment_id",
        "bridge_type",
        "damage_type",
        "severity_level",
        "ai1_split",
    ]

    joined = split.merge(
        labels[label_fields],
        on="experiment_id",
        how="left",
        suffixes=("_split", "_label"),
        validate="one_to_one",
    )

    if joined["damage_type_label"].isna().any():
        raise DataValidationError(
            "AI2 split contains experiment_id "
            "missing from labels"
        )

    for field in (
        "bridge_type",
        "damage_type",
        "severity_level",
        "ai1_split",
    ):
        if not (
            joined[f"{field}_split"]
            == joined[f"{field}_label"]
        ).all():
            raise DataValidationError(
                f"Split/label mismatch in {field}"
            )

    if not set(
        joined["damage_type_split"]
    ).issubset(LABEL_MAP):
        raise DataValidationError(
            "Unsupported damage_type label"
        )

    if not (
        joined["ai1_split_split"] == "test"
    ).all():
        raise DataValidationError(
            "AI1 train/validation experiment included "
            "in AI2 candidate split"
        )

    other_bridge = (
        "TRUSS" if bridge_type == "BOX" else "BOX"
    )

    other_raw = _read_csv(
        DATA_DIR
        / f"{other_bridge.lower()}_anomaly_results.csv"
    )

    _require_columns(
        other_raw,
        required_raw,
        "Other bridge AI1 Window results",
    )

    if other_raw["experiment_id"].isna().any():
        raise DataValidationError(
            "Other bridge data contains missing experiment_id"
        )

    overlap = (
        set(raw["experiment_id"])
        & set(other_raw["experiment_id"])
    )

    if overlap:
        raise DataValidationError(
            "BOX and TRUSS share experiment_id values: "
            f"{sorted(map(str, overlap))[:5]}"
        )

    all_raw_ids = (
        set(raw["experiment_id"])
        | set(other_raw["experiment_id"])
    )

    if not set(
        split["experiment_id"]
    ).issubset(all_raw_ids):
        raise DataValidationError(
            "AI2 split contains experiment_id "
            "absent from raw source"
        )

    if not (
        raw["bridge_type"] == bridge_type
    ).all():
        raise DataValidationError(
            "Raw bridge_type does not match requested bridge"
        )

    if not (
        other_raw["bridge_type"] == other_bridge
    ).all():
        raise DataValidationError(
            "Other raw source contains incorrect bridge_type"
        )

    bridge_split = joined.loc[
        joined["bridge_type_split"] == bridge_type
    ].copy()

    data = raw.merge(
        bridge_split[
            [
                "experiment_id",
                "ai2_split",
                "damage_type_split",
                "condition_code",
            ]
        ],
        on="experiment_id",
        how="inner",
        validate="many_to_one",
    )

    if data.empty:
        raise DataValidationError(
            f"No AI2 candidate Windows found for {bridge_type}"
        )

    data["target"] = (
        data["damage_type_split"].map(LABEL_MAP)
    )

    if data["target"].isna().any():
        raise DataValidationError(
            "Ground-truth conversion failed"
        )

    return data
