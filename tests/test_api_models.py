
"""Validation tests for the AI1 HTTP envelope.

All numeric values are deterministic interface-test fixtures.
They are not measured sensor values, model training data, or
predictive-performance evidence.

No dummy baseline or trained model artifact is loaded.
"""

from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from src.api_models import Ai1WindowRequest
from src.baseline_manager import FEATURES


NOW_US = int(
    datetime.now(timezone.utc).timestamp() * 1_000_000
)


def request_payload(
    bridge_type: str = "BOX",
    *,
    experiment_id: str = "SIM_BOX_001",
    window_id: str = "W001",
) -> dict:
    """Build an AI1 request for schema and API behavior tests only."""
    return {
        "schema_version": "1.0",
        "message_type": "ai1_window",
        "bridge_type": bridge_type,
        "experiment_id": experiment_id,
        "window_id": window_id,
        "window_start": "2026-10-25 10:30:27.670000",
        "window_start_us": NOW_US - 2_000_000,
        "window_end_us": NOW_US,
        "raw": {
            "start_us": NOW_US - 2_000_000,
            "end_us": NOW_US,
            "sensors": {
                "acc01": None,
            },
        },
        "ai1": {
            "prediction": "Normal",
            "warming_up": False,
            "anomaly_score": 0.1,
            "condition": {
                "condition": "1kg-C",
                "load_level_kg": 1.0,
                "load_position": "C",
                "temperature_c": 24.5,
            },
            "features": {
                feature: 1.0
                for feature in FEATURES
            },
        },
    }


class ApiModelTests(unittest.TestCase):

    def test_box_and_truss_are_accepted(self):
        box = Ai1WindowRequest.model_validate(
            request_payload()
        )

        truss = Ai1WindowRequest.model_validate(
            request_payload(
                "TRUSS",
                experiment_id="SIM_TRUSS_001",
            )
        )

        self.assertEqual(
            box.bridge_type,
            "BOX",
        )

        self.assertEqual(
            truss.bridge_type,
            "TRUSS",
        )

    def test_required_features_are_exactly_validated_by_name(self):
        payload = request_payload()

        self.assertEqual(
            list(payload["ai1"]["features"]),
            FEATURES,
        )

        payload["ai1"]["features"].pop(
            FEATURES[0]
        )

        with self.assertRaises(ValidationError):
            Ai1WindowRequest.model_validate(
                payload
            )

    def test_null_nan_and_infinity_features_are_rejected(self):
        invalid_values = (
            None,
            float("nan"),
            float("inf"),
        )

        for invalid in invalid_values:
            payload = request_payload()

            payload["ai1"]["features"][
                FEATURES[0]
            ] = invalid

            with self.subTest(
                invalid=repr(invalid)
            ):
                with self.assertRaises(
                    ValidationError
                ):
                    Ai1WindowRequest.model_validate(
                        payload
                    )

    def test_null_raw_sensor_array_is_allowed_and_n_is_checked(self):
        payload = request_payload()

        payload["raw"]["sensors"] = {
            "acc01": {
                "sample_rate_hz": 100.0,
                "unit": "m/s2",
                "t0_us": NOW_US,
                "n": 3,
                "axes": {
                    "x": None,
                },
            }
        }

        Ai1WindowRequest.model_validate(
            payload
        )

        invalid = copy.deepcopy(
            payload
        )

        invalid["raw"]["sensors"]["acc01"][
            "axes"
        ]["x"] = [1.0, 2.0]

        with self.assertRaises(
            ValidationError
        ):
            Ai1WindowRequest.model_validate(
                invalid
            )

    def test_type_interval_and_message_constraints(self):
        mutations = (
            lambda p: p.update({
                "message_type": "unknown"
            }),
            lambda p: p.update({
                "bridge_type": "ARCH"
            }),
            lambda p: p.update({
                "experiment_id": " "
            }),
            lambda p: p.update({
                "window_start_us": p["window_end_us"]
            }),
        )

        for mutate in mutations:
            payload = request_payload()
            mutate(payload)

            with self.subTest(
                payload=payload
            ):
                with self.assertRaises(
                    ValidationError
                ):
                    Ai1WindowRequest.model_validate(
                        payload
                    )


if __name__ == "__main__":
    unittest.main()
