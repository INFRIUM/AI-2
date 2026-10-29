
"""Feature-change unit tests without stored dummy baseline dependencies.

All numeric values in this file are local test fixtures.
They are not hardware measurements, training data, or performance evidence.

A synthetic lookup marked as MEASURED/VALIDATED is used only to exercise
the arithmetic branch of the feature-change implementation. Such metadata
does not establish real-world measurement provenance.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.baseline_manager import FEATURES
from src.feature_change import (
    FeatureChangeInputError,
    calculate_feature_changes,
    load_baseline_lookup,
    safe_calculate_feature_changes,
)


def make_test_lookup() -> dict:
    """Build an in-memory baseline for arithmetic tests only."""
    conditions = {
        "BOX": ("0kg-NONE", "1kg-C"),
        "TRUSS": ("1kg-E", "2kg-E"),
    }

    baselines = {}

    for bridge_type, bridge_conditions in conditions.items():
        baselines[bridge_type] = {}

        for condition in bridge_conditions:
            baselines[bridge_type][condition] = {
                "coverage": {
                    "coverage_status": "TEST_ONLY",
                    "normal_experiment_count": 2,
                    "normal_window_count": 10,
                },
                "features": {
                    feature: {
                        "mean": 2.0,
                        "median": 2.0,
                    }
                    for feature in FEATURES
                },
            }

    return {
        "data_source": "MEASURED",
        "validation_status": "VALIDATED",
        "baselines": baselines,
    }


def fixture(bridge_type: str, condition: str) -> dict:
    """Create a standalone input Window with deterministic test values."""
    load_text, position = condition.split("-", 1)

    return {
        "schema_version": "feature-change-unit-test-v1",
        "bridge_type": bridge_type,
        "window_id": f"TEST_{bridge_type}_{condition}",
        "timestamp": "2026-10-15T12:30:00+09:00",
        "features": {
            feature: 2.0
            for feature in FEATURES
        },
        "environment": {
            "load_level_kg": float(
                load_text.removesuffix("kg")
            ),
            "load_position": position,
            "temperature_c": 24.5,
        },
        "ai1_result": {
            "test_fixture": True,
        },
    }


class FeatureChangeTests(unittest.TestCase):

    def setUp(self):
        self.lookup = make_test_lookup()

    def test_box_normal_input(self):
        result = calculate_feature_changes(
            fixture("BOX", "1kg-C"),
            self.lookup,
        )

        self.assertEqual(
            result["bridge_type"],
            "BOX",
        )

        self.assertEqual(
            result["baseline_status"],
            "TEST_ONLY",
        )

        self.assertEqual(
            result["baseline_source"],
            "MEASURED",
        )

        self.assertEqual(
            result["feature_changes"]["acc01_rms_x"][
                "relative_change_percent"
            ],
            0.0,
        )

    def test_truss_normal_input(self):
        result = calculate_feature_changes(
            fixture("TRUSS", "2kg-E"),
            self.lookup,
        )

        self.assertEqual(
            result["load_condition"],
            "2kg-E",
        )

        self.assertEqual(
            result["bridge_type"],
            "TRUSS",
        )

    def test_zero_load_empty_position_normalizes_to_none(self):
        payload = fixture("BOX", "0kg-NONE")
        payload["environment"]["load_position"] = ""

        result = calculate_feature_changes(
            payload,
            self.lookup,
        )

        self.assertEqual(
            result["load_condition"],
            "0kg-NONE",
        )

    def test_missing_feature(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"].pop("acc01_rms_x")

        result = safe_calculate_feature_changes(
            payload,
            self.lookup,
        )

        self.assertEqual(
            result["error"]["code"],
            "FEATURE_MISSING",
        )

    def test_non_numeric_feature(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"]["acc01_rms_x"] = "not-a-number"

        result = safe_calculate_feature_changes(
            payload,
            self.lookup,
        )

        self.assertEqual(
            result["error"]["code"],
            "FEATURE_NON_NUMERIC",
        )

    def test_non_finite_feature(self):
        payload = fixture("BOX", "1kg-C")
        payload["features"]["acc01_rms_x"] = math.nan

        result = safe_calculate_feature_changes(
            payload,
            self.lookup,
        )

        self.assertEqual(
            result["error"]["code"],
            "FEATURE_NON_FINITE",
        )

    def test_missing_baseline_condition_has_no_fallback(self):
        payload = fixture("BOX", "1kg-C")
        payload["environment"]["load_level_kg"] = 3.0

        result = safe_calculate_feature_changes(
            payload,
            self.lookup,
        )

        self.assertEqual(
            result["error"]["code"],
            "BASELINE_NOT_FOUND",
        )

    def test_zero_baseline_mean_returns_null_relative_change(self):
        test_lookup = copy.deepcopy(self.lookup)

        test_lookup["baselines"]["BOX"]["1kg-C"][
            "features"
        ]["acc01_rms_x"]["mean"] = 0.0

        payload = fixture("BOX", "1kg-C")
        payload["features"]["acc01_rms_x"] = 1.0

        change = calculate_feature_changes(
            payload,
            test_lookup,
        )["feature_changes"]["acc01_rms_x"]

        self.assertIsNone(
            change["relative_change_percent"],
        )

        self.assertIsNone(
            change["absolute_relative_change_percent"],
        )

        self.assertEqual(
            change["absolute_difference"],
            1.0,
        )

    def test_output_is_json_serializable(self):
        result = calculate_feature_changes(
            fixture("TRUSS", "1kg-E"),
            self.lookup,
        )

        serialized = json.dumps(
            result,
            allow_nan=False,
        )

        self.assertIsInstance(
            serialized,
            str,
        )

    def test_missing_baseline_is_not_ready(self):
        payload = fixture("BOX", "1kg-C")

        result = safe_calculate_feature_changes(
            payload,
            baseline_lookup=None,
        )

        self.assertEqual(
            result["error"]["code"],
            "BASELINE_NOT_READY",
        )

    def test_unvalidated_baseline_is_rejected(self):
        payload = fixture("BOX", "1kg-C")
        lookup = copy.deepcopy(self.lookup)

        lookup["data_source"] = "DUMMY"
        lookup["validation_status"] = "NOT_VALIDATED"

        result = safe_calculate_feature_changes(
            payload,
            lookup,
        )

        self.assertEqual(
            result["error"]["code"],
            "BASELINE_NOT_VALIDATED",
        )

    def test_invalid_baseline_median_is_rejected(self):
        payload = fixture("BOX", "1kg-C")
        lookup = copy.deepcopy(self.lookup)

        lookup["baselines"]["BOX"]["1kg-C"][
            "features"
        ]["acc01_rms_x"]["median"] = math.nan

        result = safe_calculate_feature_changes(
            payload,
            lookup,
        )

        self.assertEqual(
            result["error"]["code"],
            "BASELINE_MEDIAN_INVALID",
        )

    def test_explicit_baseline_file_validation(self):
        """Reject a file without verified measurement metadata."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test_baseline.json"

            lookup = make_test_lookup()
            lookup["data_source"] = "DUMMY"
            lookup["validation_status"] = "NOT_VALIDATED"

            path.write_text(
                json.dumps(lookup),
                encoding="utf-8",
            )

            with self.assertRaises(
                FeatureChangeInputError
            ) as context:
                load_baseline_lookup(path)

            self.assertEqual(
                context.exception.code,
                "BASELINE_NOT_VALIDATED",
            )


if __name__ == "__main__":
    unittest.main()
