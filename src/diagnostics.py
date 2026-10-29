
"""Validation-only diagnostics for measured AI2 Model D candidates.

Only measured-data candidates from an explicitly selected training run
are accepted. This script loads only run-scoped pending-validation models.

Diagnostics use AI2 Train and Validation data only. Test rows are excluded.

A PENDING_VALIDATION candidate may be inspected for diagnostics, but
diagnostic results do not authorize real-time inference or RiskScore use.

Outputs are stored in a unique run directory without overwriting
legacy diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_validated_data
from src.evaluate import evaluate_probabilities
from src.preprocessing import (
    make_preprocessor,
    prepare_features,
)
from src.train import (
    CONFIG_PATH,
    EXPECTED_CLASSES,
    MODELS_DIR,
    cumulative_features,
    validate_config,
    validate_training_splits,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "results"
    / "measured"
    / "diagnostics"
)

CLASS_ORDER = list(EXPECTED_CLASSES)

SHORTCUT_FEATURES = [
    "load_level_kg",
    "load_position",
    "temperature_c",
]

LOAD_CONDITIONS = [
    "0kg-NONE",
    "1kg-C",
    "1kg-E",
    "2kg-C",
    "2kg-E",
]


def load_config():
    """Read and validate the shared training configuration."""
    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)

    validate_config(config)
    return config


def validate_run_id(run_id: str) -> str:
    """Reject paths and invalid training-run identifiers."""
    if (
        not isinstance(run_id, str)
        or not run_id
        or "/" in run_id
        or "\\" in run_id
        or run_id in {".", ".."}
        or not all(
            char.isalnum() or char in "-_"
            for char in run_id
        )
    ):
        raise ValueError(
            "run_id must contain only letters, "
            "numbers, hyphens, or underscores"
        )

    return run_id


def model_path(
    run_id: str,
    bridge_type: str,
) -> Path:
    """Resolve a measured Model D candidate path."""
    run_id = validate_run_id(run_id)

    if bridge_type not in {"BOX", "TRUSS"}:
        raise ValueError(
            "bridge_type must be BOX or TRUSS"
        )

    return (
        MODELS_DIR
        / run_id
        / bridge_type.lower()
        / "model_d.joblib"
    )


def load_measured_artifact(
    run_id: str,
    bridge_type: str,
    expected_features: list[str],
):
    """Validate metadata before accepting a diagnostic candidate.

    joblib files must come from a trusted local training run.
    Never load untrusted joblib files.
    """
    path = model_path(
        run_id,
        bridge_type,
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Measured Model D is missing: {path}"
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"{bridge_type}: model artifact must be a dictionary"
        )

    required_metadata = {
        "data_source": "MEASURED",
        "validation_status": "PENDING_VALIDATION",
        "bridge_type": bridge_type,
        "model_name": "D",
        "training_split": "train",
        "selection_split": "val",
        "test_evaluated": False,
    }

    for field, expected in required_metadata.items():
        if artifact.get(field) != expected:
            raise ValueError(
                f"{bridge_type}: invalid artifact metadata "
                f"{field}; expected {expected!r}"
            )

    if artifact.get("class_order") != CLASS_ORDER:
        raise ValueError(
            f"{bridge_type}: class order mismatch"
        )

    if artifact.get("raw_features") != expected_features:
        raise ValueError(
            f"{bridge_type}: Model D feature set mismatch"
        )

    pipeline = artifact.get("pipeline")

    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            f"{bridge_type}: invalid model pipeline"
        )

    if not all(
        name in pipeline.named_steps
        for name in ("preprocessor", "classifier")
    ):
        raise ValueError(
            f"{bridge_type}: incomplete model pipeline"
        )

    return artifact


def load_condition(
    frame: pd.DataFrame,
) -> pd.Series:
    """Create normalized load-condition keys."""
    if (
        "load_level_kg" not in frame.columns
        or "load_position" not in frame.columns
    ):
        raise ValueError(
            "Load condition columns are missing"
        )

    loads = pd.to_numeric(
        frame["load_level_kg"],
        errors="coerce",
    )

    if not np.isfinite(
        loads.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Invalid load_level_kg values"
        )

    position = (
        frame["load_position"]
        .fillna("NONE")
        .replace("", "NONE")
        .astype(str)
        .str.upper()
    )

    if (
        (loads != 0)
        & position.eq("NONE")
    ).any():
        raise ValueError(
            "Nonzero load requires load_position"
        )

    position = position.where(
        loads != 0,
        "NONE",
    )

    return (
        loads.map(
            lambda value: f"{value:g}kg"
        )
        + "-"
        + position
    )


def feature_importance(
    artifact,
    validation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate associative Model D feature importance."""
    pipeline = artifact["pipeline"]
    raw_features = artifact["raw_features"]

    x_validation = prepare_features(
        validation_data,
        raw_features,
    )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    booster = classifier.get_booster()

    gain = booster.get_score(
        importance_type="gain"
    )

    weight = booster.get_score(
        importance_type="weight"
    )

    encoded_target = (
        validation_data["target"]
        .map({
            label: index
            for index, label in enumerate(
                artifact["class_order"]
            )
        })
    )

    if encoded_target.isna().any():
        raise ValueError(
            "Validation target encoding failed"
        )

    permutation = permutation_importance(
        pipeline,
        x_validation,
        encoded_target,
        scoring="f1_macro",
        n_repeats=10,
        random_state=20261014,
        n_jobs=1,
    )

    permutation_scores = {
        feature: (
            float(mean),
            float(std),
        )
        for feature, mean, std in zip(
            raw_features,
            permutation.importances_mean,
            permutation.importances_std,
        )
    }

    rows = []

    for index, name in enumerate(
        transformed_names
    ):
        raw_name = (
            "load_position"
            if name.startswith(
                "load_position_"
            )
            else name
        )

        if raw_name not in permutation_scores:
            raise ValueError(
                f"Unknown transformed feature: {name}"
            )

        perm_mean, perm_std = (
            permutation_scores[raw_name]
        )

        rows.append({
            "xgboost_input_feature": name,
            "raw_feature": raw_name,
            "xgboost_gain": float(
                gain.get(
                    f"f{index}",
                    0.0,
                )
            ),
            "xgboost_weight": int(
                weight.get(
                    f"f{index}",
                    0,
                )
            ),
            "permutation_macro_f1_mean_decrease": (
                perm_mean
            ),
            "permutation_macro_f1_std": (
                perm_std
            ),
            "importance_note": (
                "Associative diagnostic only; "
                "not a causal effect or RiskScore."
            ),
        })

    return pd.DataFrame(
        rows
    ).sort_values(
        "xgboost_gain",
        ascending=False,
        ignore_index=True,
    )


def metric_rows(
    bridge_type,
    scope,
    data,
    probabilities,
    include_confusion=False,
):
    """Build Window and experiment-level metric rows."""
    evaluation = evaluate_probabilities(
        data["target"],
        probabilities,
        data["experiment_id"],
        CLASS_ORDER,
    )

    rows = []

    for unit in (
        "window",
        "experiment",
    ):
        metrics = evaluation[unit]

        row = {
            "bridge_type": bridge_type,
            "scope": scope,
            "unit": unit,
            "metric_type": "summary",
            "class": "ALL",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "recall": np.nan,
            "support": int(
                len(data)
                if unit == "window"
                else data["experiment_id"].nunique()
            ),
        }

        if include_confusion:
            row["confusion_matrix"] = (
                json.dumps(
                    metrics["confusion_matrix"]
                )
            )

        rows.append(row)

        for class_metric in metrics[
            "class_metrics"
        ]:
            rows.append({
                "bridge_type": bridge_type,
                "scope": scope,
                "unit": unit,
                "metric_type": "class_recall",
                "class": class_metric["class"],
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "recall": class_metric["recall"],
                "support": class_metric["support"],
                **(
                    {"confusion_matrix": ""}
                    if include_confusion
                    else {}
                ),
            })

    return rows


def load_condition_metrics(
    bridge_type,
    validation_data,
    probabilities,
):
    """Evaluate Model D by measured load condition."""
    data = validation_data.copy()

    data["load_condition"] = (
        load_condition(data)
    )

    rows = []

    for condition in LOAD_CONDITIONS:
        subset = data.loc[
            data["load_condition"]
            == condition
        ]

        if subset.empty:
            continue

        probability_subset = probabilities[
            subset[
                "validation_position"
            ].to_numpy(dtype=int)
        ]

        predicted = np.asarray(
            CLASS_ORDER
        )[
            probability_subset.argmax(
                axis=1
            )
        ]

        (
            precision,
            recall,
            f1,
            support,
        ) = precision_recall_fscore_support(
            subset["target"],
            predicted,
            labels=CLASS_ORDER,
            zero_division=0,
        )

        missing_classes = [
            label
            for label, count in zip(
                CLASS_ORDER,
                support,
            )
            if count == 0
        ]

        for label, value, count in zip(
            CLASS_ORDER,
            recall,
            support,
        ):
            rows.append({
                "bridge_type": bridge_type,
                "load_condition": condition,
                "window_count": int(
                    len(subset)
                ),
                "experiment_count": int(
                    subset["experiment_id"].nunique()
                ),
                "accuracy": float(
                    accuracy_score(
                        subset["target"],
                        predicted,
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        subset["target"],
                        predicted,
                        labels=CLASS_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "class": label,
                "class_present": bool(
                    count
                ),
                "recall": float(
                    value
                ),
                "missing_classes": ";".join(
                    missing_classes
                ),
            })

    return rows


def cramers_v(
    frame: pd.DataFrame,
) -> float:
    """Measure association between damage class and load condition."""
    table = pd.crosstab(
        frame["target"],
        frame["load_condition"],
    )

    observed = table.to_numpy(
        dtype=float
    )

    if observed.size == 0:
        return 0.0

    total = observed.sum()

    if total <= 0:
        return 0.0

    expected = (
        observed.sum(
            axis=1,
            keepdims=True,
        )
        * observed.sum(
            axis=0,
            keepdims=True,
        )
        / total
    )

    chi2 = (
        (
            (observed - expected) ** 2
        )
        / np.where(
            expected == 0,
            1,
            expected,
        )
    ).sum()

    denominator = (
        total
        * min(
            observed.shape[0] - 1,
            observed.shape[1] - 1,
        )
    )

    return (
        float(
            np.sqrt(
                chi2 / denominator
            )
        )
        if denominator
        else 0.0
    )


def crosstab_rows(
    bridge_type,
    data,
):
    """Check damage/load associations in Train and Validation."""
    output = []

    for split_name in (
        "train",
        "val",
    ):
        subset = data.loc[
            data["ai2_split"]
            == split_name
        ].copy()

        if subset.empty:
            continue

        subset["load_condition"] = (
            load_condition(subset)
        )

        experiments = (
            subset[
                [
                    "experiment_id",
                    "target",
                    "load_condition",
                ]
            ]
            .drop_duplicates()
        )

        association = cramers_v(
            experiments
        )

        grouped = (
            experiments.groupby(
                [
                    "target",
                    "load_condition",
                ]
            )
            .size()
        )

        for (
            damage_class,
            condition,
        ), count in grouped.items():
            output.append({
                "bridge_type": bridge_type,
                "split": split_name,
                "damage_class": damage_class,
                "load_condition": condition,
                "experiment_count": int(
                    count
                ),
                "cramers_v_damage_vs_load": (
                    association
                ),
            })

    return output


def diagnose_bridge(
    bridge_type,
    run_id,
    config,
):
    """Run diagnostics without writing output files."""
    data = load_validated_data(
        bridge_type
    )

    train, validation = (
        validate_training_splits(
            data,
            bridge_type,
            CLASS_ORDER,
        )
    )

    validation = (
        validation.reset_index(
            drop=True
        )
    )

    validation["validation_position"] = (
        np.arange(
            len(validation)
        )
    )

    model_d_features = (
        cumulative_features(
            config["models"],
            "D",
        )
    )

    artifact = load_measured_artifact(
        run_id,
        bridge_type,
        model_d_features,
    )

    probabilities = (
        artifact["pipeline"]
        .predict_proba(
            prepare_features(
                validation,
                model_d_features,
            )
        )
    )

    if probabilities.shape != (
        len(validation),
        len(CLASS_ORDER),
    ):
        raise ValueError(
            f"{bridge_type}: unexpected "
            "probability dimensions"
        )

    importance = feature_importance(
        artifact,
        validation,
    )

    condition_rows = (
        load_condition_metrics(
            bridge_type,
            validation,
            probabilities,
        )
    )

    experiment_rows = metric_rows(
        bridge_type,
        "model_d",
        validation,
        probabilities,
        include_confusion=True,
    )

    shortcut = Pipeline([
        (
            "preprocessor",
            make_preprocessor(
                SHORTCUT_FEATURES
            ),
        ),
        (
            "classifier",
            XGBClassifier(
                **config[
                    "xgboost_parameters"
                ],
                random_state=config[
                    "random_state"
                ],
            ),
        ),
    ])

    class_to_index = {
        label: index
        for index, label in enumerate(
            CLASS_ORDER
        )
    }

    shortcut.fit(
        prepare_features(
            train,
            SHORTCUT_FEATURES,
        ),
        train["target"].map(
            class_to_index
        ),
    )

    shortcut_probabilities = (
        shortcut.predict_proba(
            prepare_features(
                validation,
                SHORTCUT_FEATURES,
            )
        )
    )

    shortcut_rows = metric_rows(
        bridge_type,
        "load_temperature_only",
        validation,
        shortcut_probabilities,
    )

    crosstab = crosstab_rows(
        bridge_type,
        data,
    )

    return {
        "importance": importance,
        "conditions": condition_rows,
        "experiments": experiment_rows,
        "shortcut": shortcut_rows,
        "crosstab": crosstab,
    }


def create_diagnostic_id():
    """Create a unique diagnostics execution ID."""
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    return (
        f"{timestamp}_"
        f"{uuid.uuid4().hex[:12]}"
    )


def save_diagnostics(
    run_id,
    diagnostic_id,
    bridge_results,
):
    """Save measured diagnostics without overwriting old results."""
    output_path = (
        OUTPUT_DIR
        / run_id
        / diagnostic_id
    )

    if output_path.exists():
        raise FileExistsError(
            f"Diagnostic output already exists: {output_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    condition_rows = []
    experiment_rows = []
    shortcut_rows = []
    crosstab = []

    for bridge_type in (
        "BOX",
        "TRUSS",
    ):
        result = bridge_results[
            bridge_type
        ]

        result["importance"].to_csv(
            output_path
            / (
                "feature_importance_"
                f"{bridge_type.lower()}.csv"
            ),
            index=False,
        )

        condition_rows.extend(
            result["conditions"]
        )

        experiment_rows.extend(
            result["experiments"]
        )

        shortcut_rows.extend(
            result["shortcut"]
        )

        crosstab.extend(
            result["crosstab"]
        )

    pd.DataFrame(
        condition_rows
    ).to_csv(
        output_path
        / "load_condition_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        experiment_rows
    ).to_csv(
        output_path
        / "experiment_level_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        shortcut_rows
    ).to_csv(
        output_path
        / "shortcut_diagnostic.csv",
        index=False,
    )

    pd.DataFrame(
        crosstab
    ).to_csv(
        output_path
        / "damage_load_crosstab.csv",
        index=False,
    )

    manifest = {
        "training_run_id": run_id,
        "diagnostic_id": diagnostic_id,
        "data_source": "MEASURED",
        "validation_status": "PENDING_VALIDATION",
        "data_split": [
            "train",
            "val",
        ],
        "test_evaluated": False,
        "model": "D",
        "bridge_types": [
            "BOX",
            "TRUSS",
        ],
        "note": (
            "Validation-only diagnostics. "
            "Feature importance is associative, "
            "not causal. Results do not establish "
            "structural safety or RiskScore."
        ),
    }

    with (
        output_path
        / "diagnostic_manifest.json"
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

    return output_path


def main():
    """Run diagnostics for an explicitly selected measured training run."""
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only diagnostics "
            "for measured AI2 Model D"
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help=(
            "Measured training run ID "
            "created by src.train"
        ),
    )

    args = parser.parse_args()

    run_id = validate_run_id(
        args.run_id
    )

    config = load_config()

    bridge_results = {}

    for bridge_type in (
        "BOX",
        "TRUSS",
    ):
        bridge_results[
            bridge_type
        ] = diagnose_bridge(
            bridge_type,
            run_id,
            config,
        )

    diagnostic_id = (
        create_diagnostic_id()
    )

    output_path = save_diagnostics(
        run_id,
        diagnostic_id,
        bridge_results,
    )

    print(
        "Measured diagnostics saved to:",
        output_path,
    )

    print(
        "Validation status: PENDING_VALIDATION"
    )

    print(
        "Test data: NOT USED"
    )

    print(
        "Real-time inference: NOT AUTHORIZED"
    )


if __name__ == "__main__":
    main()
