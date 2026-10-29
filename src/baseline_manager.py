
"""Build measured-data NORMAL vibration baselines for AI2.

Only provenance-validated measured data from the AI2 Train split
is eligible. Baselines are descriptive statistics, not RiskScores.

Generated baselines remain PENDING_VALIDATION until independently
reviewed. This module never approves its own outputs.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_validated_data


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results" / "baselines" / "measured"

MIN_EXPERIMENTS_PER_CONDITION = 3
NEAR_ZERO_STD = 1e-12

FEATURES = [
    "acc01_rms_x",
    "acc01_rms_y",
    "acc01_rms_z",
    "acc02_rms_x",
    "acc02_rms_y",
    "acc02_rms_z",
    "acc01_p2p_x",
    "acc01_p2p_y",
    "acc01_p2p_z",
    "acc02_p2p_x",
    "acc02_p2p_y",
    "acc02_p2p_z",
    "acc01_dom_freq_x",
    "acc01_dom_freq_y",
    "acc01_dom_freq_z",
    "acc02_dom_freq_x",
    "acc02_dom_freq_y",
    "acc02_dom_freq_z",
    "acc01_damping_z",
    "acc02_damping_z",
    "cross_amp_ratio_L4_L2_z",
    "cross_torsion_y_z_L2",
]

LOAD_ORDER = [
    "0kg-NONE",
    "1kg-C",
    "1kg-E",
    "2kg-C",
    "2kg-E",
]


def load_condition(frame: pd.DataFrame) -> pd.Series:
    """Build normalized load-condition keys."""
    position = (
        frame["load_position"]
        .fillna("NONE")
        .replace("", "NONE")
    )

    invalid = (
        frame["load_level_kg"].ne(0)
        & position.eq("NONE")
    )

    if invalid.any():
        raise ValueError(
            "Nonzero load has no load_position; "
            "baseline cannot be trusted"
        )

    return (
        frame["load_level_kg"].map(
            lambda value: f"{value:g}kg"
        )
        + "-"
        + position
    )


def validate_normal_data(
    frame: pd.DataFrame,
    bridge_type: str,
) -> None:
    """Reject incomplete or nonfinite baseline source features."""
    if frame.empty:
        raise ValueError(
            f"{bridge_type}: no AI2 Train NORMAL data"
        )

    required = {
        "experiment_id",
        "load_level_kg",
        "load_position",
        *FEATURES,
    }

    missing = sorted(required - set(frame.columns))

    if missing:
        raise ValueError(
            f"{bridge_type}: missing baseline columns: {missing}"
        )

    if frame["experiment_id"].isna().any():
        raise ValueError(
            f"{bridge_type}: missing experiment_id"
        )

    numeric_columns = [
        "load_level_kg",
        *FEATURES,
    ]

    for column in numeric_columns:
        numeric = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{bridge_type}: invalid numeric values "
                f"in {column}"
            )


def feature_statistics(
    frame: pd.DataFrame,
    bridge_type: str,
    scope: str,
    condition: str | None = None,
) -> list[dict]:
    """Calculate descriptive feature statistics."""
    experiments = int(
        frame["experiment_id"].nunique()
    )
    windows = int(len(frame))

    experiment_means = (
        frame.groupby("experiment_id")[FEATURES]
        .mean()
    )

    output = []

    for feature in FEATURES:
        values = frame[feature]

        q1, q3 = values.quantile(
            [0.25, 0.75]
        )

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        std = values.std()

        output.append({
            "bridge_type": bridge_type,
            "scope": scope,
            "load_condition": condition or "ALL",
            "feature": feature,
            "mean": float(values.mean()),
            "std": float(std),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
            "iqr": float(iqr),
            "experiment_count": experiments,
            "window_count": windows,
            "inter_experiment_mean_std": (
                float(
                    experiment_means[feature].std()
                )
                if experiments > 1
                else None
            ),
            "outlier_window_count": int(
                (
                    (values < lower)
                    | (values > upper)
                ).sum()
            ),
            "near_zero_std": (
                bool(std <= NEAR_ZERO_STD)
                if pd.notna(std)
                else None
            ),
        })

    return output


def coverage_row(
    frame: pd.DataFrame,
    bridge_type: str,
    condition: str,
) -> dict:
    """Summarize independent experiment coverage."""
    experiment_means = (
        frame.groupby("experiment_id")[FEATURES]
        .mean()
    )

    feature_std = frame[FEATURES].std()

    outlier_windows = pd.Series(
        False,
        index=frame.index,
    )

    for feature in FEATURES:
        q1, q3 = frame[feature].quantile(
            [0.25, 0.75]
        )

        iqr = q3 - q1

        outlier_windows |= (
            (frame[feature] < q1 - 1.5 * iqr)
            | (frame[feature] > q3 + 1.5 * iqr)
        )

    if len(experiment_means) > 1:
        inter_std = experiment_means.std()
    else:
        inter_std = pd.Series(dtype=float)

    experiment_count = int(
        frame["experiment_id"].nunique()
    )

    return {
        "bridge_type": bridge_type,
        "load_condition": condition,
        "normal_experiment_count": experiment_count,
        "normal_window_count": int(len(frame)),
        "minimum_experiment_requirement": (
            MIN_EXPERIMENTS_PER_CONDITION
        ),
        "coverage_status": (
            "SUFFICIENT"
            if experiment_count
            >= MIN_EXPERIMENTS_PER_CONDITION
            else "INSUFFICIENT_EXPERIMENTS"
        ),
        "outlier_window_count_any_feature": int(
            outlier_windows.sum()
        ),
        "near_zero_std_feature_count": int(
            (feature_std <= NEAR_ZERO_STD).sum()
        ),
        "inter_experiment_mean_std_median": (
            float(inter_std.median())
            if not inter_std.empty
            else None
        ),
        "inter_experiment_mean_std_max": (
            float(inter_std.max())
            if not inter_std.empty
            else None
        ),
        "notes": (
            "Window count does not replace "
            "independent experiment count."
        ),
    }


def nested_lookup(
    by_load: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict:
    """Build a lookup that cannot be used before approval."""
    lookup = {}

    for bridge_type in ("BOX", "TRUSS"):
        lookup[bridge_type] = {}

        bridge_conditions = by_load.loc[
            by_load["bridge_type"] == bridge_type,
            "load_condition",
        ].unique()

        for condition in bridge_conditions:
            stats = by_load.loc[
                (by_load["bridge_type"] == bridge_type)
                & (
                    by_load["load_condition"]
                    == condition
                )
            ]

            cover = coverage.loc[
                (coverage["bridge_type"] == bridge_type)
                & (
                    coverage["load_condition"]
                    == condition
                )
            ].iloc[0].to_dict()

            lookup[bridge_type][condition] = {
                "coverage": cover,
                "features": {
                    row.feature: {
                        key: row[key]
                        for key in (
                            "mean",
                            "std",
                            "median",
                            "min",
                            "max",
                            "iqr",
                            "inter_experiment_mean_std",
                        )
                    }
                    for _, row in stats.iterrows()
                },
            }

    return {
        "schema_version": "measured-baseline-v1",
        "data_source": "MEASURED",
        "validation_status": "PENDING_VALIDATION",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "schema_note": (
            "Descriptive NORMAL baselines only. "
            "No RiskScore or risk thresholds."
        ),
        "baseline_source": (
            "AI2 Train NORMAL from provenance-validated "
            "measured AI1 outputs."
        ),
        "minimum_experiment_requirement": (
            MIN_EXPERIMENTS_PER_CONDITION
        ),
        "baselines": lookup,
    }


def json_safe(value):
    """Convert missing or nonfinite values to JSON null."""
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return None

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    if pd.isna(value):
        return None

    return value


def main():
    """Generate new, unapproved measured baseline artifacts."""
    summary_rows = []
    by_load_rows = []
    coverage_rows = []

    for bridge_type in ("BOX", "TRUSS"):
        data = load_validated_data(
            bridge_type
        )

        normal_train = data.loc[
            (data["ai2_split"] == "train")
            & (data["target"] == "NORMAL")
        ].copy()

        validate_normal_data(
            normal_train,
            bridge_type,
        )

        normal_train["load_condition"] = (
            load_condition(normal_train)
        )

        summary_rows.extend(
            feature_statistics(
                normal_train,
                bridge_type,
                "normal_train_all_conditions",
            )
        )

        for condition in LOAD_ORDER:
            subset = normal_train.loc[
                normal_train["load_condition"]
                == condition
            ]

            if subset.empty:
                continue

            by_load_rows.extend(
                feature_statistics(
                    subset,
                    bridge_type,
                    "normal_train_by_load",
                    condition,
                )
            )

            coverage_rows.append(
                coverage_row(
                    subset,
                    bridge_type,
                    condition,
                )
            )

    summary = pd.DataFrame(
        summary_rows
    )
    by_load = pd.DataFrame(
        by_load_rows
    )
    coverage = pd.DataFrame(
        coverage_rows
    )

    lookup = json_safe(
        nested_lookup(
            by_load,
            coverage,
        )
    )

    # Do not create output files until every bridge is validated.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "summary": (
            OUTPUT_DIR
            / "normal_baseline_summary.csv"
        ),
        "by_load": (
            OUTPUT_DIR
            / "normal_baseline_by_load.csv"
        ),
        "coverage": (
            OUTPUT_DIR
            / "baseline_coverage_report.csv"
        ),
        "lookup": (
            OUTPUT_DIR
            / "measured_baseline_lookup.json"
        ),
    }

    existing = [
        str(path)
        for path in output_files.values()
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing measured "
            f"baseline artifacts: {existing}"
        )

    summary.to_csv(
        output_files["summary"],
        index=False,
    )

    by_load.to_csv(
        output_files["by_load"],
        index=False,
    )

    coverage.to_csv(
        output_files["coverage"],
        index=False,
    )

    with output_files["lookup"].open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            lookup,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print(
        "Unapproved measured baselines saved to",
        OUTPUT_DIR,
    )
    print(
        "Validation status: PENDING_VALIDATION"
    )


if __name__ == "__main__":
    main()
