
"""Validated measurement-based evidence interface for INFRIUM AI2.

Only approved, measurement-based Model C artifacts may be used for inference.
Approval manifest and SHA-256 checks occur before joblib deserialization.

No RiskScore or risk level is calculated.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.baseline_manager import FEATURES
from src.feature_change import (
    FeatureChangeInputError,
    calculate_feature_changes,
    load_baseline_lookup,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ROOT = (
    ROOT / "models" / "measured" / "approved"
)

CLASS_ORDER = ["NORMAL", "D1", "D2", "D3"]
OUTPUT_SCHEMA_VERSION = "ai2-risk-evidence-v1"

MODEL_FILENAME = "model_c.joblib"
APPROVAL_FILENAME = "approval_manifest.json"

REQUIRED_DATA_SOURCE = "MEASURED"
REQUIRED_VALIDATION_STATUS = "VALIDATED"

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RiskEvaluationError(RuntimeError):
    """Explicit, machine-readable evaluation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _base_fields(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "bridge_type": payload.get("bridge_type"),
        "window_id": payload.get("window_id"),
        "timestamp": payload.get("timestamp"),
        "experiment_id": payload.get("experiment_id"),
        "event_time": payload.get("event_time"),
    }


def _error_result(
    payload: Any,
    error: RiskEvaluationError | FeatureChangeInputError,
    status: str,
) -> dict[str, Any]:
    return {
        **_base_fields(payload),
        "damage_class": None,
        "damage_probabilities": None,
        "risk_score": None,
        "risk_level": None,
        "risk_status": "EVALUATION_ERROR",
        "data_quality": {
            "statuses": [status],
            "primary_status": status,
        },
        "evidence": {
            "error": {
                "code": error.code,
                "message": error.message,
            }
        },
    }


def _model_not_ready() -> RiskEvaluationError:
    return RiskEvaluationError(
        "MODEL_NOT_READY",
        "A validated measurement-based AI2 model is not available.",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _load_approval_manifest(model_root: Path) -> dict[str, Any]:
    """Read the approval record without loading any model pickle."""
    path = model_root / APPROVAL_FILENAME

    if not path.is_file() or path.is_symlink():
        raise _model_not_ready()

    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, UnicodeError) as error:
        raise _model_not_ready() from error

    if not isinstance(manifest, dict):
        raise _model_not_ready()

    required = {
        "schema_version": "ai2-model-approval-v1",
        "data_source": REQUIRED_DATA_SOURCE,
        "validation_status": REQUIRED_VALIDATION_STATUS,
    }

    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise _model_not_ready()

    for field in (
        "training_run_id",
        "approved_by",
        "approved_at",
        "test_evaluation_reference",
        "source_record_reference",
    ):
        value = manifest.get(field)

        if not isinstance(value, str) or not value.strip():
            raise _model_not_ready()

    if not isinstance(manifest.get("models"), dict):
        raise _model_not_ready()

    return manifest


def _verify_model_approval(
    bridge_type: str,
    model_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Check approval metadata and file digest before deserialization."""
    if bridge_type not in {"BOX", "TRUSS"}:
        raise RiskEvaluationError(
            "BRIDGE_TYPE_INVALID",
            "bridge_type must be BOX or TRUSS",
        )

    manifest = _load_approval_manifest(model_root)

    models = manifest["models"]
    record = models.get(bridge_type)

    if not isinstance(record, dict):
        raise _model_not_ready()

    expected_path = (
        Path(bridge_type.lower()) / MODEL_FILENAME
    ).as_posix()

    required = {
        "bridge_type": bridge_type,
        "model_name": "C",
        "relative_path": expected_path,
        "data_source": REQUIRED_DATA_SOURCE,
        "validation_status": REQUIRED_VALIDATION_STATUS,
    }

    for field, expected in required.items():
        if record.get(field) != expected:
            raise _model_not_ready()

    expected_hash = record.get("sha256")

    if (
        not isinstance(expected_hash, str)
        or SHA256_PATTERN.fullmatch(expected_hash) is None
    ):
        raise _model_not_ready()

    path = model_root / expected_path

    if not path.is_file() or path.is_symlink():
        raise _model_not_ready()

    # The approved directory itself must be a real directory.
    # Do not follow symlinked bridge subdirectories.
    if (
        model_root.is_symlink()
        or (model_root / bridge_type.lower()).is_symlink()
    ):
        raise _model_not_ready()

    try:
        actual_hash = _sha256_file(path)
    except OSError as error:
        raise _model_not_ready() from error

    if actual_hash != expected_hash:
        raise _model_not_ready()

    return path, manifest


def _load_model(
    bridge_type: str,
    model_root: Path,
) -> dict[str, Any]:
    """Load a model only after approval and integrity checks."""
    path, manifest = _verify_model_approval(
        bridge_type,
        model_root,
    )

    # joblib files must originate from a trusted training/review process.
    # SHA-256 detects changes relative to the manifest; it does not
    # authenticate a manifest that an attacker can also modify.
    try:
        artifact = joblib.load(path)
    except Exception as error:
        raise RiskEvaluationError(
            "MODEL_ARTIFACT_INVALID",
            "Approved AI2 model artifact could not be loaded.",
        ) from error

    if not isinstance(artifact, dict):
        raise RiskEvaluationError(
            "MODEL_ARTIFACT_INVALID",
            "Approved AI2 model artifact is invalid.",
        )

    if (
        artifact.get("data_source") != REQUIRED_DATA_SOURCE
        or artifact.get("validation_status")
        != REQUIRED_VALIDATION_STATUS
    ):
        raise _model_not_ready()

    if artifact.get("class_order") != CLASS_ORDER:
        raise RiskEvaluationError(
            "MODEL_CLASS_ORDER_INVALID",
            "Model class order does not match the AI2 contract.",
        )

    if artifact.get("raw_features") != FEATURES:
        raise RiskEvaluationError(
            "MODEL_FEATURE_MISMATCH",
            "Model features do not match the AI1-AI2 contract.",
        )

    if (
        artifact.get("bridge_type") != bridge_type
        or artifact.get("model_name") != "C"
        or not hasattr(
            artifact.get("pipeline"),
            "predict_proba",
        )
    ):
        raise RiskEvaluationError(
            "MODEL_ARTIFACT_INVALID",
            "Model bridge type, name, or pipeline is invalid.",
        )

    run_id = manifest["training_run_id"]

    if artifact.get("training_run_id") != run_id:
        raise RiskEvaluationError(
            "MODEL_ARTIFACT_INVALID",
            "Model training run does not match approval.",
        )

    return artifact


def preload_model_artifacts(
    model_root: str | Path = DEFAULT_MODEL_ROOT,
) -> None:
    """Check both approved models when strict preloading is requested."""
    root = Path(model_root)

    for bridge_type in ("BOX", "TRUSS"):
        _load_model(bridge_type, root)


def _model_evidence(
    payload: dict[str, Any],
    bridge_type: str,
    model_root: Path,
    *,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate approved Model C classification evidence."""
    if artifact is None:
        artifact = _load_model(bridge_type, model_root)

    features = payload["features"]

    frame = pd.DataFrame(
        [{name: features[name] for name in FEATURES}]
    )

    try:
        probabilities = np.asarray(
            artifact["pipeline"].predict_proba(frame)[0],
            dtype=float,
        )
    except Exception as error:
        raise RiskEvaluationError(
            "MODEL_INFERENCE_FAILED",
            "AI2 model inference failed.",
        ) from error

    if (
        probabilities.ndim != 1
        or len(probabilities) != len(CLASS_ORDER)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.isclose(
            float(np.sum(probabilities)),
            1.0,
            atol=1e-6,
        )
    ):
        raise RiskEvaluationError(
            "MODEL_INFERENCE_INVALID",
            "Model returned invalid class probabilities.",
        )

    mapped = {
        label: float(value)
        for label, value in zip(CLASS_ORDER, probabilities)
    }

    return {
        "class_order": CLASS_ORDER,
        "predicted_class": CLASS_ORDER[
            int(np.argmax(probabilities))
        ],
        "probabilities": mapped,
        "max_probability": float(np.max(probabilities)),
        "model_name": artifact.get("model_name"),
        "model_bridge_type": bridge_type,
        "data_source": artifact.get("data_source"),
        "validation_status": artifact.get(
            "validation_status"
        ),
        "note": (
            "Classification probability is not safety "
            "probability or risk severity."
        ),
    }


def _ai1_evidence(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Preserve AI1 evidence without calculating a risk score."""
    ai1 = payload.get("ai1_result")

    if ai1 is None:
        return {
            "status": "MISSING_OPTIONAL",
            "raw": None,
            "note": (
                "AI1 result was not supplied; "
                "no AI1 evidence is inferred."
            ),
        }

    if not isinstance(ai1, dict):
        return {
            "status": "INVALID_OPTIONAL",
            "raw": ai1,
            "note": (
                "AI1 result is not an object; "
                "no AI1 evidence is interpreted."
            ),
        }

    return {
        "status": "PROVIDED_UNVALIDATED",
        "raw": ai1,
        "anomaly_score": ai1.get("anomaly_score"),
        "decision_ratio": ai1.get("decision_ratio"),
        "raw_prediction": ai1.get("raw_prediction"),
        "prediction": ai1.get("prediction"),
        "note": (
            "Values are preserved only. "
            "They are not combined into RiskScore."
        ),
    }


def _baseline_quality(
    change: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Describe baseline coverage without assigning severity."""
    status = change.get("baseline_status")

    if status == "INSUFFICIENT_EXPERIMENTS":
        statuses = ["INSUFFICIENT_BASELINE"]
    elif status == "SUFFICIENT":
        statuses = ["VALID"]
    else:
        statuses = ["PROVISIONAL_BASELINE"]

    return statuses, {
        "baseline_status": status,
        "baseline_experiment_count": change.get(
            "baseline_experiment_count"
        ),
        "baseline_window_count": change.get(
            "baseline_window_count"
        ),
        "baseline_source": change.get("baseline_source"),
        "load_condition": change.get("load_condition"),
    }


def evaluate_risk_evidence(
    payload: dict[str, Any],
    *,
    baseline_lookup: dict[str, Any] | None = None,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
) -> dict[str, Any]:
    """Return classification evidence only when a model is approved."""
    model_root = Path(model_root)

    if not isinstance(payload, dict):
        error = RiskEvaluationError(
            "PAYLOAD_INVALID",
            "Input payload must be an object.",
        )
        return _error_result(
            payload,
            error,
            "MISSING_REQUIRED",
        )

    bridge_type = payload.get("bridge_type")

    if bridge_type not in {"BOX", "TRUSS"}:
        error = RiskEvaluationError(
            "BRIDGE_TYPE_INVALID",
            "bridge_type must be BOX or TRUSS.",
        )
        return _error_result(
            payload,
            error,
            "MISSING_REQUIRED",
        )

    # Preserve existing input-validation precedence.
    from src.feature_change import _validate_payload

    try:
        _validate_payload(payload)
    except FeatureChangeInputError as error:
        return _error_result(
            payload,
            error,
            "MISSING_REQUIRED",
        )

    # Approval and SHA-256 checks occur before loading any baseline.
    try:
        artifact = _load_model(
            bridge_type,
            model_root,
        )
    except RiskEvaluationError as error:
        return _error_result(
            payload,
            error,
            (
                "VERSION_MISMATCH"
                if error.code in {
                    "MODEL_CLASS_ORDER_INVALID",
                    "MODEL_FEATURE_MISMATCH",
                }
                else "MISSING_REQUIRED"
            ),
        )

    try:
        lookup = (
            load_baseline_lookup()
            if baseline_lookup is None
            else baseline_lookup
        )

        changes = calculate_feature_changes(
            payload,
            lookup,
        )
    except FeatureChangeInputError as error:
        status = (
            "OUT_OF_CONDITION"
            if error.code == "BASELINE_NOT_FOUND"
            else "MISSING_REQUIRED"
        )

        return _error_result(
            payload,
            error,
            status,
        )

    try:
        model = _model_evidence(
            payload,
            bridge_type,
            model_root,
            artifact=artifact,
        )
    except RiskEvaluationError as error:
        return _error_result(
            payload,
            error,
            "MISSING_REQUIRED",
        )

    statuses, baseline_summary = _baseline_quality(
        changes
    )

    ai1 = _ai1_evidence(payload)

    sensor_status = (
        ai1.get("raw", {}).get("sensor_status")
        if isinstance(ai1.get("raw"), dict)
        else None
    )

    return {
        **_base_fields(payload),
        "damage_class": model["predicted_class"],
        "damage_probabilities": model["probabilities"],
        "risk_score": None,
        "risk_level": None,
        "risk_status": "PENDING_VALIDATION",
        "data_quality": {
            "statuses": statuses,
            "primary_status": statuses[0],
            "not_evaluated": [
                "STALE_DATA",
                "DUPLICATE_WINDOW",
                "VERSION_MISMATCH",
            ],
            "sensor_status": sensor_status,
            "sensor_status_note": (
                "Sensor status is preserved but not "
                "interpreted until the AI1 schema is finalized."
            ),
        },
        "evidence": {
            "baseline": baseline_summary,
            "feature_changes": changes["feature_changes"],
            "ai1_result": ai1,
            "xgboost_model_c": model,
        },
    }


def evaluate_risk_evidence_safe(
    payload: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a JSON-serializable evaluation response."""
    try:
        result = evaluate_risk_evidence(
            payload,
            **kwargs,
        )

        json.dumps(
            result,
            allow_nan=False,
        )

        return result

    except (RiskEvaluationError, FeatureChangeInputError) as error:
        return _error_result(
            payload,
            error,
            "MISSING_REQUIRED",
        )

    except (TypeError, ValueError) as error:
        wrapped = RiskEvaluationError(
            "OUTPUT_SERIALIZATION_FAILED",
            "Evaluation output is not JSON serializable.",
        )

        return _error_result(
            payload,
            wrapped,
            "MISSING_REQUIRED",
        )
