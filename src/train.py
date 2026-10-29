
"""Train measured-data AI2 XGBoost candidates for BOX and TRUSS.

Training uses AI2 Train data only.
Candidate comparison uses AI2 Validation data only.
AI2 Test data is never used for fitting or model selection.

Only provenance-validated measured data can enter this pipeline.

Artifacts are saved separately from legacy models and remain
PENDING_VALIDATION. Training does not authorize real-time inference.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_validated_data
from src.evaluate import evaluate_probabilities
from src.preprocessing import (
    make_preprocessor,
    prepare_features,
)


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT / "configs" / "ai2_feature_config.json"
)

MODELS_DIR = (
    ROOT
    / "models"
    / "measured"
    / "pending_validation"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "measured"
    / "training"
)

BRIDGE_TYPES = ("BOX", "TRUSS")
MODEL_NAMES = ("A", "B", "C", "D", "E")

EXPECTED_CLASSES = (
    "NORMAL",
    "D1",
    "D2",
    "D3",
)


def cumulative_features(models, model_name):
    """Return cumulative feature groups A through model_name."""
    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    selected = []

    for name in MODEL_NAMES[
        :MODEL_NAMES.index(model_name) + 1
    ]:
        if name not in models:
            raise ValueError(
                f"Feature configuration missing Model {name}"
            )

        selected.extend(models[name])

    if len(selected) != len(set(selected)):
        raise ValueError(
            f"Duplicate features in Model {model_name}"
        )

    return selected


def validate_config(config):
    """Validate the fixed AI2 classification contract."""
    if not isinstance(config, dict):
        raise ValueError(
            "Training configuration must be an object"
        )

    classes = config.get("class_order")

    if classes != list(EXPECTED_CLASSES):
        raise ValueError(
            "class_order must be "
            "['NORMAL', 'D1', 'D2', 'D3']"
        )

    models = config.get("models")

    if not isinstance(models, dict):
        raise ValueError(
            "Training configuration lacks models"
        )

    for model_name in MODEL_NAMES:
        features = models.get(model_name)

        if (
            not isinstance(features, list)
            or not features
            or not all(
                isinstance(feature, str)
                and feature
                for feature in features
            )
        ):
            raise ValueError(
                f"Invalid features for Model {model_name}"
            )

        cumulative_features(
            models,
            model_name,
        )

    parameters = config.get(
        "xgboost_parameters"
    )

    if not isinstance(parameters, dict):
        raise ValueError(
            "Missing xgboost_parameters"
        )

    if parameters.get("objective") != "multi:softprob":
        raise ValueError(
            "XGBoost objective must be multi:softprob"
        )

    if parameters.get("num_class") != 4:
        raise ValueError(
            "XGBoost num_class must be 4"
        )

    if "random_state" not in config:
        raise ValueError(
            "Training configuration lacks random_state"
        )


def validate_training_splits(
    data,
    bridge_type,
    classes,
):
    """Check split isolation and target coverage."""
    required_columns = {
        "experiment_id",
        "ai2_split",
        "target",
    }

    missing = (
        required_columns
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{bridge_type}: missing columns "
            f"{sorted(missing)}"
        )

    train_data = data.loc[
        data["ai2_split"] == "train"
    ].copy()

    validation_data = data.loc[
        data["ai2_split"] == "val"
    ].copy()

    test_data = data.loc[
        data["ai2_split"] == "test"
    ].copy()

    if train_data.empty or validation_data.empty:
        raise ValueError(
            f"{bridge_type}: Train and Validation "
            "data are both required"
        )

    train_ids = set(
        train_data["experiment_id"]
    )

    validation_ids = set(
        validation_data["experiment_id"]
    )

    test_ids = set(
        test_data["experiment_id"]
    )

    if train_ids & validation_ids:
        raise ValueError(
            f"{bridge_type}: Train/Validation "
            "experiment leakage detected"
        )

    if train_ids & test_ids:
        raise ValueError(
            f"{bridge_type}: Train/Test "
            "experiment leakage detected"
        )

    if validation_ids & test_ids:
        raise ValueError(
            f"{bridge_type}: Validation/Test "
            "experiment leakage detected"
        )

    expected = set(classes)

    train_classes = set(
        train_data["target"]
    )

    validation_classes = set(
        validation_data["target"]
    )

    if train_classes != expected:
        raise ValueError(
            f"{bridge_type}: Train must contain "
            f"all four classes; found "
            f"{sorted(map(str, train_classes))}"
        )

    if not validation_classes.issubset(expected):
        raise ValueError(
            f"{bridge_type}: Validation contains "
            "unknown classes"
        )

    return train_data, validation_data


def train_bridge(
    bridge_type,
    config,
    *,
    output_dir=None,
):
    """Fit five candidates using measured Train/Validation data.

    This function returns trained candidate artifacts and metrics.
    It does not write files.
    """
    validate_config(config)

    bridge_type = bridge_type.upper()

    if bridge_type not in BRIDGE_TYPES:
        raise ValueError(
            "bridge_type must be BOX or TRUSS"
        )

    data = load_validated_data(
        bridge_type
    )

    classes = config["class_order"]

    train_data, validation_data = (
        validate_training_splits(
            data,
            bridge_type,
            classes,
        )
    )

    class_to_index = {
        label: index
        for index, label in enumerate(classes)
    }

    comparison = []
    class_rows = []
    matrices = {}
    artifacts = {}

    for model_name in MODEL_NAMES:
        features = cumulative_features(
            config["models"],
            model_name,
        )

        x_train = prepare_features(
            train_data,
            features,
        )

        x_validation = prepare_features(
            validation_data,
            features,
        )

        y_train = (
            train_data["target"]
            .map(class_to_index)
        )

        if y_train.isna().any():
            raise ValueError(
                f"{bridge_type} Model {model_name}: "
                "target encoding failed"
            )

        preprocessor = make_preprocessor(
            features
        )

        classifier = XGBClassifier(
            **config["xgboost_parameters"],
            random_state=config["random_state"],
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        started = time.perf_counter()

        pipeline.fit(
            x_train,
            y_train,
        )

        elapsed = (
            time.perf_counter() - started
        )

        probabilities = (
            pipeline.predict_proba(
                x_validation
            )
        )

        if probabilities.shape[1] != len(classes):
            raise ValueError(
                f"{bridge_type} Model {model_name}: "
                "unexpected probability dimensions"
            )

        validation = evaluate_probabilities(
            validation_data["target"],
            probabilities,
            validation_data["experiment_id"],
            classes,
        )

        artifact = {
            "pipeline": pipeline,
            "class_order": classes,
            "raw_features": features,
            "bridge_type": bridge_type,
            "model_name": model_name,
            "data_source": "MEASURED",
            "validation_status": "PENDING_VALIDATION",
            "training_split": "train",
            "selection_split": "val",
            "test_evaluated": False,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "note": (
                "Model E includes decision_ratio, "
                "an AI1-derived score requiring "
                "leakage review. "
                "This candidate is not approved "
                "for real-time inference."
            ),
        }

        artifacts[model_name] = artifact

        comparison.append({
            "bridge_type": bridge_type,
            "model": model_name,
            "feature_count": len(features),
            "training_seconds": elapsed,
            "validation_window_accuracy": (
                validation["window"]["accuracy"]
            ),
            "validation_window_macro_f1": (
                validation["window"]["macro_f1"]
            ),
            "validation_experiment_accuracy": (
                validation["experiment"]["accuracy"]
            ),
            "validation_experiment_macro_f1": (
                validation["experiment"]["macro_f1"]
            ),
        })

        matrices[
            f"{bridge_type}_Model_{model_name}"
        ] = {
            "class_order": classes,
            "window": (
                validation["window"]["confusion_matrix"]
            ),
            "experiment_probability_mean": (
                validation["experiment"]["confusion_matrix"]
            ),
        }

        for unit in (
            "window",
            "experiment",
        ):
            for row in (
                validation[unit]["class_metrics"]
            ):
                class_rows.append({
                    "bridge_type": bridge_type,
                    "model": model_name,
                    "unit": unit,
                    **row,
                })

    ranked = sorted(
        comparison,
        key=lambda row: (
            -row["validation_window_macro_f1"],
            row["feature_count"],
        ),
    )

    selected_model = ranked[0]["model"]

    for row in comparison:
        row["selected_by_validation_macro_f1"] = (
            row["model"] == selected_model
        )

    return (
        comparison,
        class_rows,
        matrices,
        artifacts,
    )


def create_run_id():
    """Create a unique identifier for a training run."""
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    return (
        f"{timestamp}_"
        f"{uuid.uuid4().hex[:12]}"
    )


def save_training_run(
    run_id,
    config,
    bridge_results,
):
    """Save a complete training run without overwriting old models."""
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError(
            "run_id must be a non-empty single path component"
        )

    if not isinstance(config, dict):
        raise ValueError(
            "Training configuration must be an object"
        )

    if config.get("class_order") != list(EXPECTED_CLASSES):
        raise ValueError(
            "class_order must be "
            "['NORMAL', 'D1', 'D2', 'D3']"
        )

    if not isinstance(bridge_results, dict):
        raise ValueError(
            "bridge_results must be an object"
        )

    missing_bridges = [
        bridge_type
        for bridge_type in BRIDGE_TYPES
        if bridge_type not in bridge_results
    ]

    if missing_bridges:
        raise ValueError(
            "Training results must include BOX and TRUSS; "
            f"missing {missing_bridges}"
        )

    validated_results = {}

    for bridge_type in BRIDGE_TYPES:
        bridge_result = bridge_results[bridge_type]

        if (
            not isinstance(bridge_result, tuple)
            or len(bridge_result) != 4
        ):
            raise ValueError(
                f"{bridge_type}: invalid training result"
            )

        (
            comparison,
            class_rows,
            matrices,
            artifacts,
        ) = bridge_result

        if not isinstance(artifacts, dict):
            raise ValueError(
                f"{bridge_type}: model artifacts must be an object"
            )

        missing_models = [
            model_name
            for model_name in MODEL_NAMES
            if model_name not in artifacts
        ]

        if missing_models:
            raise ValueError(
                f"{bridge_type}: missing model artifacts "
                f"{missing_models}"
            )

        artifacts_to_save = {}

        for model_name in MODEL_NAMES:
            artifact = artifacts[model_name]

            if not isinstance(artifact, dict):
                raise ValueError(
                    f"{bridge_type} Model {model_name}: "
                    "artifact must be an object"
                )

            if artifact.get("data_source") != "MEASURED":
                raise ValueError(
                    f"{bridge_type} Model {model_name}: "
                    "data_source must be MEASURED"
                )

            if (
                artifact.get("validation_status")
                != "PENDING_VALIDATION"
            ):
                raise ValueError(
                    f"{bridge_type} Model {model_name}: "
                    "validation_status must be "
                    "PENDING_VALIDATION"
                )

            if artifact.get("test_evaluated") is not False:
                raise ValueError(
                    f"{bridge_type} Model {model_name}: "
                    "test_evaluated must be False"
                )

            existing_run_id = artifact.get("training_run_id")

            if (
                existing_run_id is not None
                and existing_run_id != run_id
            ):
                raise ValueError(
                    f"{bridge_type} Model {model_name}: "
                    "conflicting training_run_id"
                )

            # Do not mutate the in-memory candidate artifact.  The saved
            # copy is the immutable record associated with this run.
            artifacts_to_save[model_name] = {
                **artifact,
                "training_run_id": run_id,
            }

        validated_results[bridge_type] = (
            comparison,
            class_rows,
            matrices,
            artifacts_to_save,
        )

    model_run_dir = (
        MODELS_DIR / run_id
    )

    result_run_dir = (
        RESULTS_DIR / run_id
    )

    if (
        model_run_dir.exists()
        or result_run_dir.exists()
    ):
        raise FileExistsError(
            f"Training run already exists: {run_id}"
        )

    model_run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    result_run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    all_comparison = []
    all_class_rows = []
    all_matrices = {}

    for bridge_type in BRIDGE_TYPES:
        (
            comparison,
            class_rows,
            matrices,
            artifacts,
        ) = validated_results[bridge_type]

        bridge_dir = (
            model_run_dir
            / bridge_type.lower()
        )

        bridge_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        for model_name in MODEL_NAMES:
            artifact_path = (
                bridge_dir
                / f"model_{model_name.lower()}.joblib"
            )

            if artifact_path.exists():
                raise FileExistsError(
                    f"Model already exists: {artifact_path}"
                )

            joblib.dump(
                artifacts[model_name],
                artifact_path,
            )

        all_comparison.extend(
            comparison
        )

        all_class_rows.extend(
            class_rows
        )

        all_matrices.update(
            matrices
        )

    pd.DataFrame(
        all_comparison
    ).to_csv(
        result_run_dir
        / "validation_model_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        all_class_rows
    ).to_csv(
        result_run_dir
        / "validation_class_metrics.csv",
        index=False,
    )

    with (
        result_run_dir
        / "validation_confusion_matrices.json"
    ).open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            all_matrices,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    manifest = {
        "run_id": run_id,
        "data_source": "MEASURED",
        "validation_status": "PENDING_VALIDATION",
        "bridge_types": list(
            BRIDGE_TYPES
        ),
        "models_per_bridge": list(
            MODEL_NAMES
        ),
        "class_order": config["class_order"],
        "training_split": "train",
        "selection_split": "val",
        "test_evaluated": False,
        "note": (
            "Training and Validation results "
            "do not constitute final model approval. "
            "Independent Test evaluation, "
            "data provenance audit, and "
            "feature leakage review are required."
        ),
    }

    with (
        result_run_dir
        / "training_manifest.json"
    ).open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    return (
        model_run_dir,
        result_run_dir,
        pd.DataFrame(
            all_comparison
        ),
    )


def main():
    """Train both bridge types before writing any model artifacts."""
    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as handle:
        config = json.load(
            handle
        )

    validate_config(
        config
    )

    bridge_results = {}

    for bridge_type in BRIDGE_TYPES:
        bridge_results[bridge_type] = (
            train_bridge(
                bridge_type,
                config,
            )
        )

    run_id = create_run_id()

    (
        model_run_dir,
        result_run_dir,
        comparison,
    ) = save_training_run(
        run_id,
        config,
        bridge_results,
    )

    print(
        comparison.sort_values(
            ["bridge_type", "model"]
        ).to_string(index=False)
    )

    print()
    print(
        "Training run:",
        run_id,
    )
    print(
        "Model artifacts:",
        model_run_dir,
    )
    print(
        "Validation results:",
        result_run_dir,
    )
    print(
        "Validation status: PENDING_VALIDATION"
    )
    print(
        "Real-time inference: NOT AUTHORIZED"
    )


if __name__ == "__main__":
    main()
