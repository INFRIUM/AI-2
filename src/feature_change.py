
"""Compare AI1 Window features with an explicitly supplied measured baseline.

This module calculates feature deltas only. It does not perform XGBoost
prediction, anomaly detection, RiskScore calculation, or risk-level decisions.

No dummy baseline is loaded automatically. A baseline must be derived from
measured data, validated, and explicitly provided by the caller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.baseline_manager import FEATURES


ROOT = Path(__file__).resolve().parents[1]

# Same output path used by src/baseline_manager.py.
DEFAULT_LOOKUP_PATH = (
    ROOT
    / "results"
    / "baselines"
    / "measured"
    / "measured_baseline_lookup.json"
)

BASELINE_SOURCE = "MEASURED"
ZERO_BASELINE_EPSILON = 1e-12


class FeatureChangeInputError(ValueError):
    """Machine-readable input or baseline validation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _validate_lookup(lookup: Any) -> dict[str, Any]:
    """Require explicit provenance and validation of a measured baseline."""
    if not isinstance(lookup, dict):
        raise FeatureChangeInputError(
            "BASELINE_LOOKUP_INVALID",
            "Baseline lookup must be an object",
        )

    if lookup.get("data_source") != "MEASURED":
        raise FeatureChangeInputError(
            "BASELINE_NOT_VALIDATED",
            "Baseline must originate from measured sensor data",
        )

    if lookup.get("validation_status") != "VALIDATED":
        raise FeatureChangeInputError(
            "BASELINE_NOT_VALIDATED",
            "Baseline must have validation_status=VALIDATED",
        )

    if not isinstance(lookup.get("baselines"), dict):
        raise FeatureChangeInputError(
            "BASELINE_LOOKUP_INVALID",
            "Baseline lookup lacks a baselines object",
        )

    return lookup


def load_baseline_lookup(
    path: str | Path = DEFAULT_LOOKUP_PATH,
) -> dict[str, Any]:
    """Load and validate a measurement-based baseline lookup.

    This function is explicit: importing the module never loads a baseline.
    The default path is reserved for future measured artifacts only.
    """
    path = Path(path)

    if not path.is_file():
        raise FeatureChangeInputError(
            "BASELINE_LOOKUP_MISSING",
            f"Baseline lookup does not exist: {path}",
        )

    try:
        with path.open(encoding="utf-8") as handle:
            lookup = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FeatureChangeInputError(
            "BASELINE_LOOKUP_INVALID",
            f"Baseline lookup is not valid UTF-8 JSON: {error}",
        ) from error

    return _validate_lookup(lookup)


def normalize_load_condition(
    environment: dict[str, Any],
) -> str:
    """Normalize load conditions to keys such as 1kg-C."""
    if not isinstance(environment, dict):
        raise FeatureChangeInputError(
            "ENVIRONMENT_INVALID",
            "environment must be an object",
        )

    if "load_level_kg" not in environment:
        raise FeatureChangeInputError(
            "LOAD_LEVEL_MISSING",
            "environment.load_level_kg is required",
        )

    value = environment["load_level_kg"]

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise FeatureChangeInputError(
            "LOAD_LEVEL_INVALID",
            "environment.load_level_kg must be a finite number",
        )

    position = environment.get("load_position")
    position = (
        "NONE"
        if position is None or position == ""
        else position
    )

    if not isinstance(position, str):
        raise FeatureChangeInputError(
            "LOAD_POSITION_INVALID",
            "environment.load_position must be a string",
        )

    position = position.upper()

    if value == 0:
        if position not in {"NONE", "C", "E"}:
            raise FeatureChangeInputError(
                "LOAD_POSITION_INVALID",
                "0kg load_position must be NONE, C, E, or empty",
            )

        return (
            "0kg-NONE"
            if position == "NONE"
            else f"0kg-{position}"
        )

    if position in {"NONE", ""}:
        raise FeatureChangeInputError(
            "LOAD_POSITION_MISSING",
            "Nonzero load requires environment.load_position",
        )

    if position not in {"C", "E"}:
        raise FeatureChangeInputError(
            "LOAD_POSITION_INVALID",
            "Nonzero load_position must be C or E",
        )

    return f"{value:g}kg-{position}"


def _validate_payload(
    payload: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise FeatureChangeInputError(
            "PAYLOAD_INVALID",
            "Input payload must be an object",
        )

    required_fields = (
        "schema_version",
        "bridge_type",
        "window_id",
        "timestamp",
        "features",
        "environment",
    )

    for field in required_fields:
        if field not in payload:
            raise FeatureChangeInputError(
                "FIELD_MISSING",
                f"Required field is missing: {field}",
            )

    bridge_type = payload["bridge_type"]

    if bridge_type not in {"BOX", "TRUSS"}:
        raise FeatureChangeInputError(
            "BRIDGE_TYPE_INVALID",
            "bridge_type must be BOX or TRUSS",
        )

    features = payload["features"]

    if not isinstance(features, dict):
        raise FeatureChangeInputError(
            "FEATURES_INVALID",
            "features must be an object",
        )

    missing = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    if missing:
        raise FeatureChangeInputError(
            "FEATURE_MISSING",
            f"Required feature(s) missing: {missing}",
        )

    for feature in FEATURES:
        value = features[feature]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise FeatureChangeInputError(
                "FEATURE_NON_NUMERIC",
                f"Feature {feature} must be numeric",
            )

        if not math.isfinite(value):
            raise FeatureChangeInputError(
                "FEATURE_NON_FINITE",
                f"Feature {feature} must not be NaN or Infinity",
            )

    condition = normalize_load_condition(
        payload["environment"]
    )

    return bridge_type, condition, features


def calculate_feature_changes(
    payload: dict[str, Any],
    baseline_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate feature changes using a validated measured baseline only."""
    bridge_type, condition, features = _validate_payload(payload)

    if baseline_lookup is None:
        raise FeatureChangeInputError(
            "BASELINE_NOT_READY",
            "No validated measured baseline has been supplied",
        )

    lookup = _validate_lookup(baseline_lookup)

    baseline = (
        lookup["baselines"]
        .get(bridge_type, {})
        .get(condition)
    )

    if baseline is None:
        raise FeatureChangeInputError(
            "BASELINE_NOT_FOUND",
            (
                "No measured baseline exists for "
                f"bridge_type={bridge_type}, "
                f"load_condition={condition}"
            ),
        )

    if not isinstance(baseline, dict):
        raise FeatureChangeInputError(
            "BASELINE_INVALID",
            "Baseline entry must be an object",
        )

    coverage = baseline.get("coverage")
    baseline_features = baseline.get("features")

    if (
        not isinstance(coverage, dict)
        or not isinstance(baseline_features, dict)
    ):
        raise FeatureChangeInputError(
            "BASELINE_INVALID",
            "Baseline entry lacks coverage or feature statistics",
        )

    changes = {}

    for feature in FEATURES:
        stats = baseline_features.get(feature)

        if (
            not isinstance(stats, dict)
            or "mean" not in stats
            or "median" not in stats
        ):
            raise FeatureChangeInputError(
                "BASELINE_FEATURE_MISSING",
                f"Baseline statistics missing for {feature}",
            )

        mean = stats["mean"]
        median = stats["median"]

        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or not math.isfinite(mean)
        ):
            raise FeatureChangeInputError(
                "BASELINE_MEAN_INVALID",
                f"Baseline mean is invalid for {feature}",
            )

        if (
            isinstance(median, bool)
            or not isinstance(median, (int, float))
            or not math.isfinite(median)
        ):
            raise FeatureChangeInputError(
                "BASELINE_MEDIAN_INVALID",
                f"Baseline median is invalid for {feature}",
            )

        current = float(features[feature])
        mean = float(mean)
        median = float(median)

        difference = current - mean

        relative = (
            None
            if abs(mean) <= ZERO_BASELINE_EPSILON
            else difference / abs(mean) * 100.0
        )

        changes[feature] = {
            "current_value": current,
            "baseline_mean": mean,
            "baseline_median": median,
            "absolute_difference": abs(difference),
            "relative_change_percent": relative,
            "absolute_relative_change_percent": (
                abs(relative)
                if relative is not None
                else None
            ),
        }

    return {
        "schema_version": payload["schema_version"],
        "bridge_type": bridge_type,
        "window_id": payload["window_id"],
        "timestamp": payload["timestamp"],
        "load_condition": condition,
        "baseline_status": coverage.get("coverage_status"),
        "baseline_experiment_count": coverage.get(
            "normal_experiment_count"
        ),
        "baseline_window_count": coverage.get(
            "normal_window_count"
        ),
        "baseline_source": BASELINE_SOURCE,
        "feature_changes": changes,
        "note": (
            "Feature deltas only. "
            "No XGBoost prediction, RiskScore, "
            "or risk-level decision is included."
        ),
    }


def safe_calculate_feature_changes(
    payload: dict[str, Any],
    baseline_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe error object instead of raising."""
    try:
        return calculate_feature_changes(
            payload,
            baseline_lookup,
        )
    except FeatureChangeInputError as error:
        return {
            "error": {
                "code": error.code,
                "message": error.message,
            }
        }
