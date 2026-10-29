
"""Risk evaluator contract and model approval security tests.

Synthetic values are used only for software contract testing.
No dummy-trained model or dummy baseline is loaded.
No model training, performance evaluation, or approval is performed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.baseline_manager import FEATURES
from src.risk_evaluator import (
    CLASS_ORDER,
    RiskEvaluationError,
    _load_model,
    _verify_model_approval,
    evaluate_risk_evidence_safe,
)


def fixture(
    bridge_type: str,
    *,
    ai1: bool = True,
) -> dict:
    """Schema-level input only; not a measured sensor sample."""
    payload = {
        "schema_version": "risk-contract-test-v1",
        "bridge_type": bridge_type,
        "window_id": f"CONTRACT_{bridge_type}_001",
        "timestamp": "2026-10-15T12:30:00+09:00",
        "features": {
            feature: 1.0
            for feature in FEATURES
        },
        "environment": {
            "load_level_kg": 1.0,
            "load_position": "C",
            "temperature_c": 24.5,
        },
    }

    if ai1:
        payload["ai1_result"] = {
            "anomaly_score": 0.42,
            "decision_ratio": 0.8,
            "raw_prediction": "Normal",
            "prediction": "Normal",
        }

    return payload


class RiskEvaluatorTests(unittest.TestCase):
    """Existing public response contract tests."""

    def setUp(self):
        self.model_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.model_directory.cleanup)
        self.model_root = Path(self.model_directory.name)

    def evaluate(self, payload: dict) -> dict:
        return evaluate_risk_evidence_safe(
            payload,
            baseline_lookup=None,
            model_root=self.model_root,
        )

    def assert_model_not_ready(self, result, bridge_type):
        self.assertEqual(
            result["bridge_type"],
            bridge_type,
        )

        self.assertEqual(
            result["risk_status"],
            "EVALUATION_ERROR",
        )

        self.assertEqual(
            result["evidence"]["error"]["code"],
            "MODEL_NOT_READY",
        )

        for field in (
            "damage_class",
            "damage_probabilities",
            "risk_score",
            "risk_level",
        ):
            with self.subTest(field=field):
                self.assertIsNone(result[field])

    def test_box_input(self):
        result = self.evaluate(fixture("BOX"))
        self.assert_model_not_ready(result, "BOX")

    def test_truss_input(self):
        result = self.evaluate(fixture("TRUSS"))
        self.assert_model_not_ready(result, "TRUSS")

    def test_baseline_shortage_is_preserved(self):
        result = self.evaluate(fixture("BOX"))

        self.assert_model_not_ready(result, "BOX")

        self.assertNotIn(
            "baseline",
            result.get("evidence", {}),
        )

    def test_missing_feature_is_not_normal(self):
        payload = fixture("BOX")
        payload["features"].pop("acc01_rms_x")

        result = self.evaluate(payload)

        self.assertEqual(
            result["risk_status"],
            "EVALUATION_ERROR",
        )

        for field in (
            "damage_class",
            "damage_probabilities",
            "risk_score",
            "risk_level",
        ):
            self.assertIsNone(result[field])

        self.assertNotEqual(
            result.get("damage_class"),
            "NORMAL",
        )

    def test_missing_model_file(self):
        result = self.evaluate(fixture("BOX"))

        self.assertEqual(
            result["evidence"]["error"]["code"],
            "MODEL_NOT_READY",
        )

    def test_invalid_class_order(self):
        """Unapproved artifacts must be blocked before loading."""
        path = self.model_root / "box"
        path.mkdir(parents=True, exist_ok=True)

        # Arbitrary bytes are sufficient because no manifest exists.
        (path / "model_c.joblib").write_bytes(
            b"unapproved-test-file"
        )

        result = self.evaluate(fixture("BOX"))

        self.assert_model_not_ready(result, "BOX")

    def test_missing_ai1_is_marked_optional(self):
        result = self.evaluate(
            fixture("BOX", ai1=False)
        )

        self.assert_model_not_ready(result, "BOX")

        self.assertNotIn(
            "ai1_result",
            result.get("evidence", {}),
        )

    def test_json_serialization(self):
        result = self.evaluate(fixture("TRUSS"))

        serialized = json.dumps(
            result,
            allow_nan=False,
        )

        self.assertIsInstance(serialized, str)

        restored = json.loads(serialized)

        self.assertEqual(
            restored["evidence"]["error"]["code"],
            "MODEL_NOT_READY",
        )

    def test_score_and_level_are_null(self):
        for bridge_type in ("BOX", "TRUSS"):
            with self.subTest(bridge_type=bridge_type):
                result = self.evaluate(
                    fixture(bridge_type)
                )

                for field in (
                    "risk_score",
                    "risk_level",
                    "damage_class",
                    "damage_probabilities",
                ):
                    self.assertIsNone(result[field])

    def test_class_order_contract(self):
        """Damage types are not ordinal risk levels."""
        self.assertEqual(
            CLASS_ORDER,
            ["NORMAL", "D1", "D2", "D3"],
        )


class ModelApprovalTests(unittest.TestCase):
    """Approval gate tests using temporary files, never real models."""

    RUN_ID = "contract_test_run"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.model_root = Path(temporary.name)

    def create_model_file(
        self,
        bridge_type: str = "BOX",
        content: bytes = b"test-model-placeholder",
    ) -> tuple[Path, str]:
        """Create non-executable placeholder bytes, not a model."""
        relative = (
            Path(bridge_type.lower()) / "model_c.joblib"
        )

        path = self.model_root / relative
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(content)

        return path, hashlib.sha256(content).hexdigest()

    def create_manifest(
        self,
        bridge_type: str = "BOX",
        *,
        sha256: str | None = None,
    ) -> dict:
        if sha256 is None:
            _, sha256 = self.create_model_file(
                bridge_type
            )

        manifest = {
            "schema_version": "ai2-model-approval-v1",
            "data_source": "MEASURED",
            "validation_status": "VALIDATED",
            "training_run_id": self.RUN_ID,
            "approved_by": "test-reviewer",
            "approved_at": "2026-10-29T12:00:00+09:00",
            "test_evaluation_reference": "test-only-reference",
            "source_record_reference": "test-only-source",
            "models": {
                bridge_type: {
                    "bridge_type": bridge_type,
                    "model_name": "C",
                    "relative_path": (
                        f"{bridge_type.lower()}/model_c.joblib"
                    ),
                    "data_source": "MEASURED",
                    "validation_status": "VALIDATED",
                    "sha256": sha256,
                }
            },
        }

        self.write_manifest(manifest)
        return manifest

    def write_manifest(self, manifest: dict):
        path = (
            self.model_root / "approval_manifest.json"
        )

        path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def assert_not_ready(self, bridge_type="BOX"):
        with self.assertRaises(
            RiskEvaluationError
        ) as context:
            _verify_model_approval(
                bridge_type,
                self.model_root,
            )

        self.assertEqual(
            context.exception.code,
            "MODEL_NOT_READY",
        )

    def test_missing_approval_manifest(self):
        self.create_model_file("BOX")

        self.assert_not_ready("BOX")

    def test_invalid_approval_status(self):
        manifest = self.create_manifest()

        manifest["validation_status"] = (
            "PENDING_VALIDATION"
        )

        self.write_manifest(manifest)
        self.assert_not_ready()

    def test_invalid_data_source(self):
        manifest = self.create_manifest()

        manifest["data_source"] = "SYNTHETIC"

        self.write_manifest(manifest)
        self.assert_not_ready()

    def test_missing_approval_identity(self):
        manifest = self.create_manifest()

        manifest["approved_by"] = ""

        self.write_manifest(manifest)
        self.assert_not_ready()

    def test_missing_test_evaluation_reference(self):
        manifest = self.create_manifest()

        manifest["test_evaluation_reference"] = ""

        self.write_manifest(manifest)
        self.assert_not_ready()

    def test_sha256_mismatch(self):
        manifest = self.create_manifest()

        # File changed after the manifest was written.
        model_path = (
            self.model_root / "box" / "model_c.joblib"
        )
        model_path.write_bytes(
            b"modified-after-approval"
        )

        self.assert_not_ready()

    def test_invalid_sha256_format(self):
        manifest = self.create_manifest()

        manifest["models"]["BOX"]["sha256"] = (
            "not-a-valid-sha256"
        )

        self.write_manifest(manifest)
        self.assert_not_ready()

    def test_bridge_type_mismatch(self):
        manifest = self.create_manifest("BOX")

        manifest["models"]["BOX"]["bridge_type"] = (
            "TRUSS"
        )

        self.write_manifest(manifest)
        self.assert_not_ready("BOX")

    def test_model_path_mismatch(self):
        manifest = self.create_manifest("BOX")

        manifest["models"]["BOX"]["relative_path"] = (
            "truss/model_c.joblib"
        )

        self.write_manifest(manifest)
        self.assert_not_ready("BOX")

    def test_missing_truss_approval(self):
        self.create_manifest("BOX")

        self.assert_not_ready("TRUSS")

    def test_approved_metadata_and_digest_gate(self):
        """The verification gate may pass without loading a pickle."""
        self.create_manifest("BOX")

        with patch(
            "src.risk_evaluator.joblib.load"
        ) as mocked_load:
            path, manifest = _verify_model_approval(
                "BOX",
                self.model_root,
            )

            self.assertEqual(
                path,
                self.model_root
                / "box"
                / "model_c.joblib",
            )

            self.assertEqual(
                manifest["training_run_id"],
                self.RUN_ID,
            )

            mocked_load.assert_not_called()

    def test_unapproved_model_never_deserializes(self):
        """No joblib.load call if the digest does not match."""
        self.create_manifest("BOX")

        model_path = (
            self.model_root / "box" / "model_c.joblib"
        )
        model_path.write_bytes(b"tampered-content")

        with patch(
            "src.risk_evaluator.joblib.load"
        ) as mocked_load:
            with self.assertRaises(
                RiskEvaluationError
            ) as context:
                _load_model(
                    "BOX",
                    self.model_root,
                )

            self.assertEqual(
                context.exception.code,
                "MODEL_NOT_READY",
            )

            mocked_load.assert_not_called()

    def test_model_training_run_mismatch(self):
        """Approved file must also match the training run."""
        self.create_manifest("BOX")

        artifact = {
            "data_source": "MEASURED",
            "validation_status": "VALIDATED",
            "class_order": CLASS_ORDER,
            "raw_features": FEATURES,
            "bridge_type": "BOX",
            "model_name": "C",
            "training_run_id": "different_run",
            "pipeline": unittest.mock.Mock(),
        }

        with patch(
            "src.risk_evaluator.joblib.load",
            return_value=artifact,
        ) as mocked_load:
            with self.assertRaises(
                RiskEvaluationError
            ) as context:
                _load_model(
                    "BOX",
                    self.model_root,
                )

            self.assertEqual(
                context.exception.code,
                "MODEL_ARTIFACT_INVALID",
            )

            mocked_load.assert_called_once()

    def test_model_class_order_mismatch(self):
        """Damage-class order must match the public contract."""
        self.create_manifest("BOX")

        artifact = {
            "data_source": "MEASURED",
            "validation_status": "VALIDATED",
            "class_order": [
                "D1", "NORMAL", "D2", "D3"
            ],
            "raw_features": FEATURES,
            "bridge_type": "BOX",
            "model_name": "C",
            "training_run_id": self.RUN_ID,
            "pipeline": unittest.mock.Mock(),
        }

        with patch(
            "src.risk_evaluator.joblib.load",
            return_value=artifact,
        ):
            with self.assertRaises(
                RiskEvaluationError
            ) as context:
                _load_model(
                    "BOX",
                    self.model_root,
                )

            self.assertEqual(
                context.exception.code,
                "MODEL_CLASS_ORDER_INVALID",
            )

    def test_symlink_model_file_is_rejected(self):
        """Do not follow a symlink at the model file location."""
        self.create_manifest("BOX")

        original = (
            self.model_root / "box" / "model_c.joblib"
        )

        target = self.model_root / "other_file"
        target.write_bytes(
            original.read_bytes()
        )

        original.unlink()

        try:
            original.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest(
                "Symlink creation is unavailable."
            )

        self.assert_not_ready()


if __name__ == "__main__":
    unittest.main()
