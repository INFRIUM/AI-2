
"""Analyze measured AI2 RiskScore candidate inputs.

This module does not calculate RiskScore or risk levels.

Train data:
- Vibration feature distributions
- Changes against approved measured NORMAL baselines
- AI1 anomaly-related output distributions

Validation data:
- Model C class probability distributions

Test data is excluded from all analyses.

Only explicitly selected measured model candidates are accepted.
Existing dummy artifacts and legacy risk-design results are not used.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.baseline_manager import FEATURES
from src.data_loader import load_validated_data
from src.feature_change import (
    calculate_feature_changes,
    load_baseline_lookup,
)
from src.preprocessing import prepare_features
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
    / "risk_design"
)

CLASS_ORDER = list(EXPECTED_CLASSES)
BRIDGE_TYPES = ("BOX", "TRUSS")


def validate_run_id(run_id: str) -> str:
    """Accept only simple training-run identifiers."""
    if (
        not isinstance(run_id, str)
        or not run_id
        or not all(
            char.isalnum() or char in "-_"
            for char in run_id
        )
    ):
        raise ValueError(
            "run_id must contain only letters, "
            "numbers, hyphens, and underscores"
        )

    return run_id


def load_config() -> dict:
    """Read the shared AI2 feature configuration."""
    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as handle:
        config = json.load(handle)

    validate_config(config)

    return config


def load_measured_model_c(
    run_id: str,
    bridge_type: str,
    expected_features: list[str],
) -> dict:
    """Load an explicitly selected measured Model C candidate.

    joblib uses pickle internally. Only trusted locally generated
    model artifacts must be loaded.
    """
    run_id = validate_run_id(run_id)

    if bridge_type not in BRIDGE_TYPES:
        raise ValueError(
            "bridge_type must be BOX or TRUSS"
        )

    path = (
        MODELS_DIR
        / run_id
        / bridge_type.lower()
        / "model_c.joblib"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Measured Model C is missing: {path}"
        )

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"{bridge_type}: invalid Model C artifact"
        )

    required = {
        "data_source": "MEASURED",
        "validation_status": "PENDING_VALIDATION",
        "bridge_type": bridge_type,
        "model_name": "C",
        "training_split": "train",
        "selection_split": "val",
        "test_evaluated": False,
    }

    for key, expected in required.items():
        if artifact.get(key) != expected:
            raise ValueError(
                f"{bridge_type}: invalid Model C "
                f"metadata for {key}"
            )

    if artifact.get("class_order") != CLASS_ORDER:
        raise ValueError(
            f"{bridge_type}: Model C class order mismatch"
        )

    if artifact.get("raw_features") != expected_features:
        raise ValueError(
            f"{bridge_type}: Model C feature mismatch"
        )

    pipeline = artifact.get("pipeline")

    if (
        pipeline is None
        or not hasattr(pipeline, "predict_proba")
    ):
        raise ValueError(
            f"{bridge_type}: Model C pipeline is invalid"
        )

    return artifact


def distribution(values: pd.Series) -> dict:
    """Return descriptive statistics without fabricating missing data."""
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "iqr": np.nan,
        }

    q1, q3 = numeric.quantile(
        [0.25, 0.75]
    )

    return {
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "median": float(numeric.median()),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "iqr": float(q3 - q1),
    }


def vibration_feature_rows(
    data: pd.DataFrame,
    bridge_type: str,
) -> list[dict]:
    """Compare measured vibration features using Train rows only."""
    rows = []

    experiment_means = (
        data.groupby(
            ["experiment_id", "target"]
        )[FEATURES]
        .mean()
        .reset_index()
    )

    for unit, source in (
        ("window", data),
        ("experiment_mean", experiment_means),
    ):
        normal = source.loc[
            source["target"] == "NORMAL"
        ]

        if normal.empty:
            raise ValueError(
                f"{bridge_type}: NORMAL Train data is missing"
            )

        for feature in FEATURES:
            normal_values = normal[feature]

            normal_mean = (
                normal_values.mean()
            )

            normal_std = (
                normal_values.std()
            )

            for damage_class, subset in (
                source.groupby("target")
            ):
                values = subset[feature]

                difference = (
                    values.mean()
                    - normal_mean
                )

                pooled = np.sqrt(
                    (
                        values.std() ** 2
                        + normal_std ** 2
                    )
                    / 2
                )

                standardized = (
                    float(difference / pooled)
                    if (
                        np.isfinite(pooled)
                        and pooled > 0
                    )
                    else np.nan
                )

                rows.append({
                    "bridge_type": bridge_type,
                    "source_split": "train",
                    "unit": unit,
                    "damage_class": damage_class,
                    "feature": feature,
                    "experiment_count": int(
                        subset["experiment_id"].nunique()
                    ),
                    "window_count": int(
                        len(subset)
                    ),
                    **distribution(values),
                    "normal_mean": float(
                        normal_mean
                    ),
                    "mean_difference_vs_normal": (
                        float(difference)
                    ),
                    "standardized_mean_difference_vs_normal": (
                        standardized
                    ),
                    "note": (
                        "Windows within an experiment "
                        "are correlated. Experiment means "
                        "provide a separate aggregation "
                        "but do not establish causality."
                    ),
                })

    return rows


def payload_from_row(
    row: pd.Series,
    bridge_type: str,
) -> dict:
    """Build a feature-change request from a measured Train row."""
    position = (
        None
        if pd.isna(row["load_position"])
        else row["load_position"]
    )

    return {
        "schema_version": "risk-design-analysis-v1",
        "bridge_type": bridge_type,
        "window_id": row["window_id"],
        "timestamp": str(
            row["window_start"]
        ),
        "features": {
            feature: float(row[feature])
            for feature in FEATURES
        },
        "environment": {
            "load_level_kg": float(
                row["load_level_kg"]
            ),
            "load_position": position,
            "temperature_c": float(
                row["temperature_c"]
            ),
        },
        "ai1_result": {},
    }


def baseline_change_rows(
    data: pd.DataFrame,
    bridge_type: str,
    lookup: dict,
) -> list[dict]:
    """Analyze changes against an approved measured baseline."""
    long_rows = []

    for _, row in data.iterrows():
        change = calculate_feature_changes(
            payload_from_row(
                row,
                bridge_type,
            ),
            lookup,
        )

        for feature, result in (
            change["feature_changes"].items()
        ):
            long_rows.append({
                "experiment_id": row["experiment_id"],
                "damage_class": row["target"],
                "feature": feature,
                "absolute_difference": (
                    result["absolute_difference"]
                ),
                "absolute_relative_change_percent": (
                    result[
                        "absolute_relative_change_percent"
                    ]
                ),
                "baseline_status": (
                    change["baseline_status"]
                ),
                "baseline_experiment_count": (
                    change["baseline_experiment_count"]
                ),
                "baseline_window_count": (
                    change["baseline_window_count"]
                ),
            })

    if not long_rows:
        raise ValueError(
            f"{bridge_type}: no baseline change rows"
        )

    changes = pd.DataFrame(
        long_rows
    )

    experiment_means = (
        changes.groupby(
            [
                "experiment_id",
                "damage_class",
                "feature",
                "baseline_status",
                "baseline_experiment_count",
                "baseline_window_count",
            ],
            dropna=False,
        )[
            [
                "absolute_difference",
                "absolute_relative_change_percent",
            ]
        ]
        .mean()
        .reset_index()
    )

    rows = []

    for unit, source in (
        ("window", changes),
        ("experiment_mean", experiment_means),
    ):
        grouped = source.groupby(
            [
                "damage_class",
                "feature",
                "baseline_status",
            ],
            dropna=False,
        )

        for (
            damage_class,
            feature,
            status,
        ), subset in grouped:
            abs_diff = subset[
                "absolute_difference"
            ]

            abs_relative = (
                subset[
                    "absolute_relative_change_percent"
                ]
                .dropna()
            )

            rows.append({
                "bridge_type": bridge_type,
                "source_split": "train",
                "unit": unit,
                "damage_class": damage_class,
                "feature": feature,
                "baseline_status": status,
                "experiment_count": int(
                    subset["experiment_id"].nunique()
                ),
                "window_count": int(
                    len(subset)
                ),
                "mean_absolute_difference": float(
                    abs_diff.mean()
                ),
                "median_absolute_difference": float(
                    abs_diff.median()
                ),
                "mean_absolute_relative_change_percent": (
                    float(abs_relative.mean())
                    if not abs_relative.empty
                    else np.nan
                ),
                "median_absolute_relative_change_percent": (
                    float(abs_relative.median())
                    if not abs_relative.empty
                    else np.nan
                ),
                "normal_comparison_note": (
                    "NORMAL Train rows may contribute "
                    "to the same baseline. This is not "
                    "an independent validation comparison."
                    if damage_class == "NORMAL"
                    else
                    "Damage Train rows are compared "
                    "with approved measured NORMAL "
                    "baselines under matching conditions."
                ),
            })

    return rows


def ai1_score_rows(
    data: pd.DataFrame,
    bridge_type: str,
) -> list[dict]:
    """Summarize available measured AI1 output columns."""
    definitions = {
        "anomaly_score": (
            "AI1 anomaly score. Exact scale and "
            "calibration require AI1 contract verification.",
            "AI1 model output",
        ),
        "decision_ratio": (
            "AI1 derived decision statistic. "
            "Threshold semantics require AI1 "
            "contract verification.",
            "AI1 derived output",
        ),
        "is_anomaly": (
            "AI1 anomaly flag, if supplied.",
            "AI1 output",
        ),
        "raw_prediction": (
            "AI1 per-window decision, if supplied.",
            "AI1 decision output",
        ),
        "prediction": (
            "AI1 post-processed decision, if supplied.",
            "AI1 post-processing output",
        ),
    }

    rows = []

    for column, (
        definition,
        provenance,
    ) in definitions.items():
        available = (
            column in data.columns
        )

        if not available:
            rows.append({
                "bridge_type": bridge_type,
                "source_split": "train",
                "column": column,
                "availability": "ABSENT",
                "definition": definition,
                "provenance": provenance,
            })

            continue

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            for damage_class, subset in (
                data.groupby("target")
            ):
                rows.append({
                    "bridge_type": bridge_type,
                    "source_split": "train",
                    "column": column,
                    "availability": "PRESENT",
                    "damage_class": damage_class,
                    "window_count": int(
                        len(subset)
                    ),
                    **distribution(
                        subset[column]
                    ),
                    "missing_count": int(
                        subset[column].isna().sum()
                    ),
                    "definition": definition,
                    "provenance": provenance,
                })
        else:
            for damage_class, subset in (
                data.groupby("target")
            ):
                counts = (
                    subset[column]
                    .fillna("<MISSING>")
                    .value_counts()
                )

                for value, count in (
                    counts.items()
                ):
                    rows.append({
                        "bridge_type": bridge_type,
                        "source_split": "train",
                        "column": column,
                        "availability": "PRESENT",
                        "damage_class": damage_class,
                        "category": str(value),
                        "window_count": int(
                            count
                        ),
                        "missing_count": int(
                            subset[column].isna().sum()
                        ),
                        "definition": definition,
                        "provenance": provenance,
                    })

    return rows


def probability_rows(
    data: pd.DataFrame,
    bridge_type: str,
    artifact: dict,
    model_c_features: list[str],
) -> list[dict]:
    """Summarize Model C Validation probabilities, not risk severity."""
    pipeline = artifact["pipeline"]

    x_validation = prepare_features(
        data,
        model_c_features,
    )

    probabilities = pipeline.predict_proba(
        x_validation
    )

    if probabilities.shape != (
        len(data),
        len(CLASS_ORDER),
    ):
        raise ValueError(
            f"{bridge_type}: invalid Model C "
            "probability shape"
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"{bridge_type}: non-finite probabilities"
        )

    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        )
    ):
        raise ValueError(
            f"{bridge_type}: invalid probability values"
        )

    output = data[
        ["experiment_id", "target"]
    ].copy()

    output["predicted_class"] = (
        np.asarray(CLASS_ORDER)[
            probabilities.argmax(
                axis=1
            )
        ]
    )

    output["max_probability"] = (
        probabilities.max(
            axis=1
        )
    )

    output["normal_probability"] = (
        probabilities[:, 0]
    )

    output["damage_probability_sum"] = (
        probabilities[:, 1:].sum(
            axis=1
        )
    )

    for index, label in enumerate(
        CLASS_ORDER
    ):
        output[f"prob_{label}"] = (
            probabilities[:, index]
        )

    rows = []

    for actual_class, subset in (
        output.groupby("target")
    ):
        for probability_class in (
            CLASS_ORDER
        ):
            values = subset[
                f"prob_{probability_class}"
            ]

            rows.append({
                "bridge_type": bridge_type,
                "source_split": "validation",
                "row_type": "actual_class_probability",
                "actual_class": actual_class,
                "probability_class": probability_class,
                "window_count": int(
                    len(subset)
                ),
                **distribution(values),
                "note": (
                    "Model C probability is a "
                    "classification output, not "
                    "structural risk severity."
                ),
            })

    grouped = output.groupby(
        ["target", "predicted_class"]
    )

    for (
        actual_class,
        predicted_class,
    ), subset in grouped:
        rows.append({
            "bridge_type": bridge_type,
            "source_split": "validation",
            "row_type": "prediction_outcome",
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "is_correct": bool(
                actual_class
                == predicted_class
            ),
            "window_count": int(
                len(subset)
            ),
            "experiment_count": int(
                subset["experiment_id"].nunique()
            ),
            "mean_max_probability": float(
                subset["max_probability"].mean()
            ),
            "median_max_probability": float(
                subset["max_probability"].median()
            ),
            "mean_normal_probability": float(
                subset["normal_probability"].mean()
            ),
            "mean_damage_probability_sum": float(
                subset["damage_probability_sum"].mean()
            ),
            "note": (
                "Model C probability is not "
                "RiskScore or damage severity."
            ),
        })

    return rows


def assessment_markdown(
    run_id: str,
) -> str:
    """Build a design assessment from measured-data evidence only."""
    return f"""# RiskScore 입력 설계 검토

- 실측 모델 학습 실행 ID: `{run_id}`
- 데이터 출처: `MEASURED`
- 분석 상태: `PENDING_VALIDATION`
- 분석 데이터: AI② Train 및 Validation
- 독립 Test 평가: 미실시

이 문서는 RiskScore 계산식, 가중치 또는 위험도 임계값을 정의하지 않는다.

D1/D2/D3는 **손상 유형**이며 위험도 순서가 아니다.

## A. 정상 기준 대비 진동 변화량

- 입력 후보: 22개 진동 Feature의 절대 차이와 상대 변화율
- 기준: 출처가 검증되고 별도 승인된 실측 NORMAL Baseline
- 정상 기준의 실험 수와 Window 수를 함께 검토한다.
- 동일 Train 데이터가 정상 기준 생성에 사용됐다면 독립 검증으로 해석하지 않는다.
- 센서 교정, 측정 단위, 하중 조건, 온도, 결측 처리를 확인해야 한다.
- 기준 상태가 충분하지 않은 조건은 RiskScore 산정 근거로 사용하지 않는다.

## B. XGBoost 구조 상태 분류 확률

- 입력 후보: Model C의 NORMAL/D1/D2/D3 확률
- 분석 대상: AI② Validation
- 모델 상태: PENDING_VALIDATION
- 분류 확률은 손상 유형에 대한 모델 출력이지 구조적 위험 심각도가 아니다.
- 확률 보정(calibration), 데이터 누출, 실험 간 독립성을 검증해야 한다.
- 독립 Test 평가와 모델 승인 전에는 운영 예측에 사용하지 않는다.

## C. AI① 이상 탐지 결과

- 입력 후보: anomaly_score, decision_ratio, raw_prediction, prediction
- 각 항목의 존재 여부와 통계는 실제 입력 데이터에 따라 산출한다.
- 점수 범위, 임계값, 시간 집계 규칙은 AI① 계약과 대조해야 한다.
- AI① 모델 학습 데이터와 AI② 평가 데이터의 누출 여부를 확인해야 한다.
- AI① 출력은 독립적인 물리적 위험도 측정값으로 간주하지 않는다.

## 필수 선행 검증

1. 실제 센서 원본 기록과 실험 로그의 출처를 확인한다.
2. BOX/TRUSS별 독립 NORMAL 반복실험과 조건별 Baseline을 검증한다.
3. Train/Validation/Test 실험 단위 분리를 확인한다.
4. Model C의 독립 Test 성능과 확률 보정을 검토한다.
5. 손상 유형, 하중, 온도 및 센서 위치에 따른 편향을 검토한다.
6. RiskScore 목적, 허용 오경보율, 미탐 대응 및 사람 검토 절차를 합의한다.
7. 검증과 승인 후에만 위험도 계산식과 운영 임계값을 확정한다.

## 해석 제한

본 분석은 입력 후보의 분포와 관계를 검토하기 위한 것이다.

분석 결과만으로 실제 교량의 구조 안전성, 손상 위치,
손상 심각도 또는 위험도를 확정할 수 없다.
"""


def create_analysis_id() -> str:
    """Create a unique ID for a risk-design analysis."""
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    return (
        f"{timestamp}_"
        f"{uuid.uuid4().hex[:12]}"
    )


def analyze_bridge(
    bridge_type: str,
    run_id: str,
    config: dict,
    lookup: dict,
) -> dict:
    """Analyze one bridge without writing output files."""
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

    model_c_features = (
        cumulative_features(
            config["models"],
            "C",
        )
    )

    artifact = load_measured_model_c(
        run_id,
        bridge_type,
        model_c_features,
    )

    return {
        "vibration": vibration_feature_rows(
            train,
            bridge_type,
        ),
        "changes": baseline_change_rows(
            train,
            bridge_type,
            lookup,
        ),
        "ai1_scores": ai1_score_rows(
            train,
            bridge_type,
        ),
        "probabilities": probability_rows(
            validation,
            bridge_type,
            artifact,
            model_c_features,
        ),
    }


def save_analysis(
    run_id: str,
    analysis_id: str,
    bridge_results: dict,
) -> Path:
    """Save results without overwriting existing risk-design outputs."""
    output_path = (
        OUTPUT_DIR
        / run_id
        / analysis_id
    )

    if output_path.exists():
        raise FileExistsError(
            f"Analysis directory already exists: {output_path}"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    vibration = []
    changes = []
    ai1_scores = []
    probabilities = []

    for bridge_type in BRIDGE_TYPES:
        result = bridge_results[
            bridge_type
        ]

        vibration.extend(
            result["vibration"]
        )

        changes.extend(
            result["changes"]
        )

        ai1_scores.extend(
            result["ai1_scores"]
        )

        probabilities.extend(
            result["probabilities"]
        )

    pd.DataFrame(
        vibration
    ).to_csv(
        output_path
        / "vibration_feature_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        changes
    ).to_csv(
        output_path
        / "baseline_change_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        ai1_scores
    ).to_csv(
        output_path
        / "ai1_anomaly_score_analysis.csv",
        index=False,
    )

    pd.DataFrame(
        probabilities
    ).to_csv(
        output_path
        / "xgboost_probability_analysis.csv",
        index=False,
    )

    (
        output_path
        / "risk_input_assessment.md"
    ).write_text(
        assessment_markdown(
            run_id
        ),
        encoding="utf-8",
    )

    manifest = {
        "training_run_id": run_id,
        "analysis_id": analysis_id,
        "data_source": "MEASURED",
        "validation_status": "PENDING_VALIDATION",
        "bridge_types": list(
            BRIDGE_TYPES
        ),
        "model": "C",
        "vibration_split": "train",
        "baseline_change_split": "train",
        "ai1_score_split": "train",
        "probability_split": "val",
        "test_evaluated": False,
        "risk_score_calculated": False,
        "note": (
            "RiskScore input analysis only. "
            "No operational risk assessment "
            "or model approval is implied."
        ),
    }

    with (
        output_path
        / "analysis_manifest.json"
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
    """Analyze explicitly selected measured training candidates."""
    parser = argparse.ArgumentParser(
        description=(
            "Measured AI2 RiskScore "
            "input candidate analysis"
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

    # Baseline must already have independent approval.
    # A newly generated PENDING_VALIDATION baseline is rejected.
    lookup = load_baseline_lookup()

    bridge_results = {}

    for bridge_type in BRIDGE_TYPES:
        bridge_results[
            bridge_type
        ] = analyze_bridge(
            bridge_type,
            run_id,
            config,
            lookup,
        )

    analysis_id = create_analysis_id()

    output_path = save_analysis(
        run_id,
        analysis_id,
        bridge_results,
    )

    print(
        "Measured risk-design analysis saved to:",
        output_path,
    )

    print(
        "Model validation status: PENDING_VALIDATION"
    )

    print(
        "RiskScore calculated: NO"
    )

    print(
        "Test data: NOT USED"
    )

    print(
        "Real-time inference: NOT AUTHORIZED"
    )


if __name__ == "__main__":
    main()
