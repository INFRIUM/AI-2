
"""Test AI2 training artifact persistence without training models."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from src import train


class TrainingArtifactTests(unittest.TestCase):
    RUN_ID = "test_training_run_001"

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name)

        self.model_dir = self.root / "models"
        self.result_dir = self.root / "results"

        model_patch = patch.object(
            train, "MODELS_DIR", self.model_dir
        )
        result_patch = patch.object(
            train, "RESULTS_DIR", self.result_dir
        )

        model_patch.start()
        result_patch.start()

        self.addCleanup(model_patch.stop)
        self.addCleanup(result_patch.stop)

        self.config = {
            "class_order": ["NORMAL", "D1", "D2", "D3"]
        }

    def make_artifact(self, bridge_type, model_name):
        """Metadata-only placeholder, not a trained model."""
        return {
            "bridge_type": bridge_type,
            "model_name": model_name,
            "data_source": "MEASURED",
            "validation_status": "PENDING_VALIDATION",
            "test_evaluated": False,
            "class_order": self.config["class_order"],
            "raw_features": ["test_feature"],
            "pipeline": None,
        }

    def make_bridge_results(self):
        results = {}

        for bridge_type in train.BRIDGE_TYPES:
            artifacts = {
                model_name: self.make_artifact(
                    bridge_type, model_name
                )
                for model_name in train.MODEL_NAMES
            }

            results[bridge_type] = (
                [],
                [],
                {},
                artifacts,
            )

        return results

    def test_training_run_id_is_saved(self):
        results = self.make_bridge_results()

        train.save_training_run(
            self.RUN_ID, self.config, results
        )

        for bridge_type in train.BRIDGE_TYPES:
            for model_name in train.MODEL_NAMES:
                with self.subTest(
                    bridge=bridge_type,
                    model=model_name,
                ):
                    path = (
                        self.model_dir
                        / self.RUN_ID
                        / bridge_type.lower()
                        / f"model_{model_name.lower()}.joblib"
                    )

                    artifact = joblib.load(path)

                    self.assertEqual(
                        artifact["training_run_id"],
                        self.RUN_ID,
                    )
                    self.assertEqual(
                        artifact["validation_status"],
                        "PENDING_VALIDATION",
                    )
                    self.assertFalse(
                        artifact["test_evaluated"]
                    )

    def test_original_artifacts_not_mutated(self):
        results = self.make_bridge_results()
        original = results["BOX"][3]["C"]

        self.assertNotIn("training_run_id", original)

        train.save_training_run(
            self.RUN_ID, self.config, results
        )

        self.assertNotIn("training_run_id", original)

    def test_training_manifest_is_pending(self):
        results = self.make_bridge_results()

        train.save_training_run(
            self.RUN_ID, self.config, results
        )

        path = (
            self.result_dir
            / self.RUN_ID
            / "training_manifest.json"
        )

        manifest = json.loads(
            path.read_text(encoding="utf-8")
        )

        self.assertEqual(
            manifest["run_id"], self.RUN_ID
        )
        self.assertEqual(
            manifest["data_source"], "MEASURED"
        )
        self.assertEqual(
            manifest["validation_status"],
            "PENDING_VALIDATION",
        )
        self.assertFalse(
            manifest["test_evaluated"]
        )

    def test_existing_run_is_not_overwritten(self):
        results = self.make_bridge_results()

        train.save_training_run(
            self.RUN_ID, self.config, results
        )

        with self.assertRaises(FileExistsError):
            train.save_training_run(
                self.RUN_ID, self.config, results
            )

    def test_missing_bridge_is_rejected(self):
        results = self.make_bridge_results()
        del results["TRUSS"]

        with self.assertRaises(ValueError):
            train.save_training_run(
                self.RUN_ID, self.config, results
            )

        self.assertFalse(
            (self.model_dir / self.RUN_ID).exists()
        )

    def test_invalid_candidate_status_is_rejected(self):
        results = self.make_bridge_results()

        results["BOX"][3]["C"][
            "validation_status"
        ] = "VALIDATED"

        with self.assertRaises(ValueError):
            train.save_training_run(
                self.RUN_ID, self.config, results
            )

        self.assertFalse(
            (self.model_dir / self.RUN_ID).exists()
        )

    def test_conflicting_run_id_is_rejected(self):
        results = self.make_bridge_results()

        results["BOX"][3]["C"][
            "training_run_id"
        ] = "another_run"

        with self.assertRaises(ValueError):
            train.save_training_run(
                self.RUN_ID, self.config, results
            )

        self.assertFalse(
            (self.model_dir / self.RUN_ID).exists()
        )

    def test_invalid_run_path_is_rejected(self):
        results = self.make_bridge_results()

        for run_id in (
            "",
            ".",
            "..",
            "../outside",
            "nested/run",
            "nested\\run",
        ):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    train.save_training_run(
                        run_id, self.config, results
                    )


if __name__ == "__main__":
    unittest.main()
