
"""JSON contract tests without dummy-trained model dependencies.

Example payloads are used only for schema, validation, and transport tests.
They are not used for model training or predictive performance evaluation.
"""

from __future__ import annotations

import copy
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.baseline_manager import FEATURES
from src.realtime_processor import RealtimeRiskProcessor
from src.risk_evaluator import CLASS_ORDER


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

MISSING_MODEL_ROOT = ROOT / "tests" / "not_a_model_root"

NOW = datetime(
    2026, 10, 15, 3, 30,
    tzinfo=timezone.utc,
)


def example(name: str) -> dict:
    return json.loads(
        (EXAMPLES / name).read_text(
            encoding="utf-8"
        )
    )


class JsonContractTests(unittest.TestCase):

    def setUp(self):
        self.processor = RealtimeRiskProcessor(
            max_age_seconds=3600,
            max_future_seconds=3600,
            baseline_lookup=None,
            model_root=MISSING_MODEL_ROOT,
        )

    def test_box_input_contract(self):
        payload = example("ai1_to_ai2_box.json")

        self.assertEqual(
            payload["bridge_type"],
            "BOX",
        )
        self.assertEqual(
            len(payload["features"]),
            len(FEATURES),
        )
        self.assertEqual(
            set(payload["features"]),
            set(FEATURES),
        )

    def test_truss_input_contract(self):
        payload = example("ai1_to_ai2_truss.json")

        self.assertEqual(
            payload["bridge_type"],
            "TRUSS",
        )
        self.assertEqual(
            len(payload["features"]),
            len(FEATURES),
        )
        self.assertEqual(
            set(payload["features"]),
            set(FEATURES),
        )

    def test_feature_names_and_model_c_order(self):
        """Verify declared contracts without loading trained artifacts."""
        self.assertEqual(
            len(FEATURES),
            22,
        )

        self.assertEqual(
            len(FEATURES),
            len(set(FEATURES)),
        )

        self.assertEqual(
            CLASS_ORDER,
            ["NORMAL", "D1", "D2", "D3"],
        )

        for bridge in ("BOX", "TRUSS"):
            name = (
                "ai1_to_ai2_box.json"
                if bridge == "BOX"
                else "ai1_to_ai2_truss.json"
            )

            payload = example(name)

            self.assertEqual(
                payload["bridge_type"],
                bridge,
            )
            self.assertEqual(
                set(payload["features"]),
                set(FEATURES),
            )

    def test_missing_required_field(self):
        payload = example("ai1_to_ai2_box.json")
        payload.pop("features")

        result = self.processor.process(
            payload,
            now=NOW,
        )

        self.assertEqual(
            result["risk_status"],
            "EVALUATION_ERROR",
        )
        self.assertEqual(
            result["evidence"]["error"]["code"],
            "FIELD_MISSING",
        )

    def test_invalid_bridge_type(self):
        payload = example("ai1_to_ai2_box.json")
        payload["bridge_type"] = "ARCH"

        result = self.processor.process(
            payload,
            now=NOW,
        )

        self.assertEqual(
            result["evidence"]["error"]["code"],
            "BRIDGE_TYPE_INVALID",
        )

    def test_invalid_timestamp(self):
        payload = example("ai1_to_ai2_box.json")
        payload["timestamp"] = "2026-10-15T12:30:00"

        result = self.processor.process(
            payload,
            now=NOW,
        )

        self.assertEqual(
            result["evidence"]["error"]["code"],
            "TIMESTAMP_INVALID",
        )

    def test_output_contract_and_null_risk(self):
        payload = example("ai1_to_ai2_box.json")

        result = self.processor.process(
            payload,
            now=NOW,
        )

        # Response must remain strict JSON-serializable.
        json.dumps(
            result,
            allow_nan=False,
        )

        required_fields = (
            "schema_version",
            "bridge_type",
            "window_id",
            "timestamp",
            "damage_class",
            "damage_probabilities",
            "risk_score",
            "risk_level",
            "risk_status",
            "data_quality",
            "evidence",
            "processing_status",
            "duplicate",
            "out_of_order",
            "received_at",
            "message_id",
        )

        for field in required_fields:
            with self.subTest(field=field):
                self.assertIn(
                    field,
                    result,
                )

        # No validated measured-data model is available.
        self.assertIsNone(
            result["damage_class"],
        )
        self.assertIsNone(
            result["damage_probabilities"],
        )
        self.assertIsNone(
            result["risk_score"],
        )
        self.assertIsNone(
            result["risk_level"],
        )

        self.assertEqual(
            result["risk_status"],
            "EVALUATION_ERROR",
        )

        self.assertEqual(
            result["evidence"]["error"]["code"],
            "MODEL_NOT_READY",
        )

        self.assertIn(
            "statuses",
            result["data_quality"],
        )

    def test_duplicate_window_policy(self):
        payload = example("ai1_to_ai2_truss.json")

        first = self.processor.process(
            copy.deepcopy(payload),
            now=NOW,
        )

        duplicate = self.processor.process(
            copy.deepcopy(payload),
            now=NOW,
        )

        self.assertFalse(
            first["duplicate"],
        )
        self.assertTrue(
            duplicate["duplicate"],
        )

        self.assertEqual(
            duplicate["processing_status"],
            "DUPLICATE",
        )

    def test_ai1_optional_fields(self):
        payload = example("ai1_to_ai2_box.json")
        payload["window_id"] = "SYNTHETIC_BOX_OPTIONAL"
        payload.pop("ai1_result", None)

        result = self.processor.process(
            payload,
            now=NOW,
        )

        # Missing optional AI1 information must not be fabricated.
        # Model readiness is checked independently.
        self.assertEqual(
            result["evidence"]["error"]["code"],
            "MODEL_NOT_READY",
        )

        self.assertIsNone(
            result["damage_class"],
        )
        self.assertIsNone(
            result["damage_probabilities"],
        )
        self.assertIsNone(
            result["risk_score"],
        )
        self.assertIsNone(
            result["risk_level"],
        )


if __name__ == "__main__":
    unittest.main()
