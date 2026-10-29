
"""FastAPI ingress and Window-processing tests without dummy-trained models.

Synthetic request fixtures are used only for transport, validation,
deduplication, ordering, and error-handling tests.
No predictive performance is measured or claimed.
"""

from __future__ import annotations

import copy
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from src.ai1_adapter import to_ai2_payload
from src.api import create_app
from src.api_models import Ai1WindowRequest
from src.realtime_processor import RealtimeRiskProcessor
from tests.test_api_models import request_payload


MISSING_MODEL_ROOT = (
    Path(__file__).resolve().parent / "not_a_model_root"
)


def client_for(
    processor: RealtimeRiskProcessor | None = None,
    *,
    limit: int = 1_048_576,
) -> TestClient:
    return TestClient(
        create_app(
            processor=processor,
            max_request_bytes=limit,
        ),
        raise_server_exceptions=False,
    )


def canonical_payload(payload: dict) -> dict:
    """Convert an HTTP request into the AI2 processor contract."""
    message = Ai1WindowRequest.model_validate(payload)
    return to_ai2_payload(message)


class ApiWindowTests(unittest.TestCase):

    def setUp(self):
        self.processor = RealtimeRiskProcessor(
            max_age_seconds=300,
            max_future_seconds=30,
            baseline_lookup=None,
            model_root=MISSING_MODEL_ROOT,
        )

    def assert_model_not_ready(self, response):
        self.assertEqual(response.status_code, 503)

        body = response.json()
        self.assertEqual(
            body["detail"]["code"],
            "MODEL_NOT_READY",
        )
        self.assertNotIn(
            str(MISSING_MODEL_ROOT),
            response.text,
        )

    def test_box_normal_and_truss_anomaly_requests(self):
        with client_for(self.processor) as client:
            normal = client.post(
                "/api/ai1/windows",
                json=request_payload("BOX"),
            )

            anomaly_payload = request_payload(
                "TRUSS",
                experiment_id="SIM_TRUSS_001",
            )
            anomaly_payload["ai1"]["prediction"] = "Anomaly"

            anomaly = client.post(
                "/api/ai1/windows",
                json=anomaly_payload,
            )

        self.assert_model_not_ready(normal)
        self.assert_model_not_ready(anomaly)

        self.assertEqual(
            self.processor.bridge_snapshot("BOX")[
                "processed_window_count"
            ],
            1,
        )
        self.assertEqual(
            self.processor.bridge_snapshot("TRUSS")[
                "processed_window_count"
            ],
            1,
        )

    def test_warming_up_is_explicit_without_validated_model(self):
        payload = request_payload()
        payload["ai1"]["warming_up"] = True
        payload["ai1"]["prediction"] = "Normal"

        with client_for(self.processor) as client:
            response = client.post(
                "/api/ai1/windows",
                json=payload,
            )

        self.assert_model_not_ready(response)

        self.assertEqual(
            self.processor.bridge_snapshot("BOX")[
                "processed_window_count"
            ],
            1,
        )

    def test_validation_and_raw_size_cases(self):
        missing = request_payload()
        missing["ai1"]["features"].pop("acc01_rms_x")

        too_big = request_payload(
            experiment_id="SIM_SIZE",
        )
        too_big["raw"]["padding"] = "x" * 2_000

        with client_for(self.processor) as client:
            invalid_response = client.post(
                "/api/ai1/windows",
                json=missing,
            )

        self.assertEqual(
            invalid_response.status_code,
            422,
        )

        with client_for(self.processor, limit=1_000) as client:
            oversized_response = client.post(
                "/api/ai1/windows",
                json=too_big,
            )

        self.assertEqual(
            oversized_response.status_code,
            413,
        )

    def test_100_kib_raw_request_is_accepted_under_default_limit(self):
        payload = request_payload(
            experiment_id="SIM_RAW_SIZE",
        )
        payload["raw"]["padding"] = "x" * (100 * 1024)

        with client_for(self.processor) as client:
            response = client.post(
                "/api/ai1/windows",
                json=payload,
            )

        # The request passes size validation, then reaches
        # the expected model-unavailable response.
        self.assert_model_not_ready(response)

        self.assertEqual(
            self.processor.bridge_snapshot("BOX")[
                "processed_window_count"
            ],
            1,
        )

    def test_deduplication_conflict_and_different_experiment_window_ids(self):
        payload = request_payload(
            experiment_id="SIM_A",
            window_id="W001",
        )

        another_experiment = request_payload(
            experiment_id="SIM_B",
            window_id="W001",
        )

        conflict = copy.deepcopy(payload)
        conflict["ai1"]["prediction"] = "Anomaly"

        first = self.processor.process(
            canonical_payload(payload)
        )
        duplicate = self.processor.process(
            canonical_payload(payload)
        )
        different = self.processor.process(
            canonical_payload(another_experiment)
        )
        conflicting = self.processor.process(
            canonical_payload(conflict)
        )

        self.assertFalse(first["duplicate"])
        self.assertTrue(duplicate["duplicate"])
        self.assertFalse(different["duplicate"])

        self.assertEqual(
            duplicate["processing_status"],
            "DUPLICATE",
        )
        self.assertEqual(
            conflicting["evidence"]["error"]["code"],
            "WINDOW_CONFLICT",
        )

        self.assertEqual(
            self.processor.bridge_snapshot("BOX")[
                "processed_window_count"
            ],
            2,
        )

    def test_bridge_conflict_for_same_experiment_window_is_not_silent(self):
        box = request_payload(
            experiment_id="SHARED",
            window_id="W001",
        )

        truss = request_payload(
            "TRUSS",
            experiment_id="SHARED",
            window_id="W001",
        )

        with client_for(self.processor) as client:
            first = client.post(
                "/api/ai1/windows",
                json=box,
            )

            conflicting = client.post(
                "/api/ai1/windows",
                json=truss,
            )

        self.assert_model_not_ready(first)

        self.assertEqual(
            conflicting.status_code,
            409,
        )
        self.assertEqual(
            conflicting.json()["detail"]["code"],
            "EXPERIMENT_WINDOW_BRIDGE_CONFLICT",
        )

    def test_stale_future_and_out_of_order_are_successful_analysis_results(self):
        stale = request_payload(
            experiment_id="SIM_TIME",
            window_id="W001",
        )
        stale["window_end_us"] -= 600_000_000
        stale["window_start_us"] -= 600_000_000

        future = request_payload(
            experiment_id="SIM_TIME",
            window_id="W002",
        )
        future["window_end_us"] += 600_000_000
        future["window_start_us"] += 600_000_000

        latest = request_payload(
            experiment_id="SIM_ORDER",
            window_id="W003",
        )

        earlier = request_payload(
            experiment_id="SIM_ORDER",
            window_id="W002",
        )
        earlier["window_end_us"] -= 1_000_000
        earlier["window_start_us"] -= 1_000_000

        stale_result = self.processor.process(
            canonical_payload(stale)
        )
        future_result = self.processor.process(
            canonical_payload(future)
        )
        latest_result = self.processor.process(
            canonical_payload(latest)
        )
        earlier_result = self.processor.process(
            canonical_payload(earlier)
        )

        self.assertEqual(
            stale_result["processing_status"],
            "STALE_DATA",
        )
        self.assertEqual(
            future_result["processing_status"],
            "FUTURE_TIMESTAMP",
        )
        self.assertEqual(
            latest_result["processing_status"],
            "PROCESSED",
        )
        self.assertEqual(
            earlier_result["processing_status"],
            "OUT_OF_ORDER",
        )

    def test_model_missing_returns_sanitized_503(self):
        with client_for(self.processor) as client:
            response = client.post(
                "/api/ai1/windows",
                json=request_payload(),
            )

        self.assert_model_not_ready(response)

    def test_unexpected_analysis_exception_is_500_without_traceback(self):
        class ExplodingProcessor:
            def process(self, _: dict):
                raise RuntimeError(
                    "synthetic internal failure"
                )

        with client_for(ExplodingProcessor()) as client:
            response = client.post(
                "/api/ai1/windows",
                json=request_payload(),
            )

        self.assertEqual(
            response.status_code,
            500,
        )
        self.assertNotIn(
            "synthetic internal failure",
            response.text,
        )

    def test_concurrent_duplicate_requests_do_not_analyze_twice(self):
        payload = canonical_payload(
            request_payload(
                experiment_id="SIM_CONCURRENT",
            )
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda _: self.processor.process(
                        copy.deepcopy(payload)
                    ),
                    range(2),
                )
            )

        self.assertEqual(
            sum(
                bool(result["duplicate"])
                for result in results
            ),
            1,
        )

        self.assertEqual(
            self.processor.bridge_snapshot("BOX")[
                "processed_window_count"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
