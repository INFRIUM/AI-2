"""FastAPI ingress for AI① Window messages.

This is a development-server ingress only. It does not send to Django, persist
raw samples, calculate RiskScore, or expose the service without authentication.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.ai1_adapter import to_ai2_payload
from src.api_models import Ai1WindowRequest
from src.feature_change import load_baseline_lookup
from src.realtime_processor import RealtimeRiskProcessor


DEFAULT_MAX_REQUEST_BYTES = 1 * 1024 * 1024
MODEL_UNAVAILABLE_CODES = {
    "MODEL_ARTIFACT_MISSING",
    "MODEL_ARTIFACT_INVALID",
    "MODEL_CLASS_ORDER_INVALID",
    "MODEL_FEATURE_MISMATCH",
    "MODEL_INFERENCE_FAILED",
    "MODEL_INFERENCE_INVALID",
    "MODEL_NOT_READY"
}
CONFLICT_CODES = {"WINDOW_CONFLICT", "EXPERIMENT_WINDOW_BRIDGE_CONFLICT"}


def configured_max_request_bytes() -> int:
    """Read a bounded development ingress limit without mixing it with risk rules."""
    raw = os.getenv("AI2_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES
    return value if value > 0 else DEFAULT_MAX_REQUEST_BYTES


def _apply_warmup_status(result: dict[str, Any], warming_up: bool) -> dict[str, Any]:
    """Keep XGBoost evidence while making AI① warm-up non-decisive."""
    result["ai1_status"] = "PENDING_WARMUP" if warming_up else "AVAILABLE"
    if warming_up:
        quality = result.setdefault("data_quality", {})
        statuses = quality.setdefault("statuses", [])
        if "AI1_WARMING_UP" not in statuses:
            statuses.append("AI1_WARMING_UP")
        not_evaluated = quality.setdefault("not_evaluated", [])
        if "AI1_WARMING_UP" in not_evaluated:
            not_evaluated.remove("AI1_WARMING_UP")
        result.setdefault("evidence", {}).setdefault("ai1_result", {})["warmup_note"] = (
            "AI① warming_up=true: its Normal/Anomaly prediction is not interpreted as a final state. "
            "Model C evidence is retained when features are valid."
        )
    return result


def _analysis_http_exception(result: dict[str, Any]) -> HTTPException | None:
    error = result.get("evidence", {}).get("error", {})
    code = error.get("code")
    if code in CONFLICT_CODES:
        return HTTPException(status_code=409, detail={"code": code, "message": "Window key conflicts with prior content."})
    if code in MODEL_UNAVAILABLE_CODES:
        # Do not expose local artifact paths or exception traces to AI①.
        return HTTPException(
            status_code=503,
            detail={"code": code, "message": "AI② analysis is temporarily unavailable."},
        )
    return None


def create_app(*, processor: RealtimeRiskProcessor | None = None, max_request_bytes: int | None = None) -> FastAPI:
    """Create an injectable app for production wiring and deterministic tests."""
    limit = configured_max_request_bytes() if max_request_bytes is None else max_request_bytes

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Start the API even when measured-data artifacts are not ready."""
        if not hasattr(app.state, "processor"):
            try:
                baseline_lookup = load_baseline_lookup()
            except FeatureChangeInputError as error:
                if error.code != "BASELINE_LOOKUP_MISSING":
                    raise
                baseline_lookup = {}

            app.state.processor = RealtimeRiskProcessor(
                baseline_lookup=baseline_lookup
            )
        
        yield

    app = FastAPI(title="INFRIUM AI② ingress", version="0.1.0", lifespan=lifespan)
    if processor is not None:
        app.state.processor = processor

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, __: Exception) -> JSONResponse:
        """Return a non-sensitive failure response; deployment logging is configured separately."""
        return JSONResponse(status_code=500, content={"detail": "Unexpected AI② server error."})

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next: Any):
        if request.url.path == "/api/ai1/windows":
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    if int(content_length) > limit:
                        return JSONResponse(status_code=413, content={"detail": "Request body exceeds configured size limit."})
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header."})
            body = await request.body()
            if len(body) > limit:
                return JSONResponse(status_code=413, content={"detail": "Request body exceeds configured size limit."})
        return await call_next(request)

    @app.post("/api/ai1/windows")
    async def receive_ai1_window(message: Ai1WindowRequest) -> dict[str, Any]:
        canonical = to_ai2_payload(message)
        result = app.state.processor.process(canonical)
        error = _analysis_http_exception(result)
        if error is not None:
            raise error
        return _apply_warmup_status(result, message.ai1.warming_up)

    return app


app = create_app()
