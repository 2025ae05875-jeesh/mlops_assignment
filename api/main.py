"""FastAPI inference service for the Heart Disease classifier.

Endpoints
  GET  /health      liveness  - process is up
  GET  /ready       readiness - model artifact loaded (used by K8s probe)
  GET  /model-info  metadata of the served model (no patient data)
  POST /predict     JSON patient features -> class + probability + confidence
  GET  /metrics     Prometheus metrics

Logging policy: one JSON line per request with request_id, method, path,
status and latency. Request bodies (patient data) are NEVER logged; for
validation errors only the offending field NAMES are logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from heartml.predict import (
    DEFAULT_MODEL_PATH,
    load_metadata,
    load_model,
    predict_records,
)
from heartml.schema import PatientFeatures, PredictionResponse

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ---------------- structured logging ----------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("heart_api")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(JsonFormatter())
    logger.addHandler(_h)
logger.setLevel(LOG_LEVEL)
logger.propagate = False


def log(level: int, msg: str, **fields) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


# ---------------- Prometheus metrics ----------------
REQUESTS = Counter("heart_api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("heart_api_request_duration_seconds", "Request latency",
                    ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("heart_api_predictions_total", "Predictions by class",
                      ["predicted_class"])
PROBABILITY = Histogram("heart_api_prediction_probability",
                        "Predicted probability of disease",
                        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
VALIDATION_ERRORS = Counter("heart_api_validation_errors_total",
                            "Rejected requests (HTTP 422)")
APP_ERRORS = Counter("heart_api_errors_total", "Unhandled errors (HTTP 5xx)")
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model is loaded")


def create_app(model_path: str | None = None) -> FastAPI:
    """App factory: lets tests build an app against a temporary model file."""
    model_path = model_path or os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model, app.state.meta = None, {}
        try:
            app.state.model = load_model(model_path)
            app.state.meta = load_metadata(model_path)
            MODEL_LOADED.set(1)
            log(logging.INFO, "model loaded", model_path=model_path,
                model_name=app.state.meta.get("model_name", "unknown"),
                model_version=app.state.meta.get("model_version", "unknown"))
        except Exception as exc:  # service stays up; /ready reports 503
            MODEL_LOADED.set(0)
            log(logging.ERROR, "model load failed", model_path=model_path,
                error=str(exc))
        yield
        log(logging.INFO, "shutting down")

    app = FastAPI(title="Heart Disease Risk API", version="1.0.0",
                  lifespan=lifespan,
                  description="Educational MLOps demo. Not for clinical use.")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:
            APP_ERRORS.inc()
            log(logging.ERROR, "unhandled error", request_id=request_id,
                path=request.url.path, error=type(exc).__name__)
            response = JSONResponse(status_code=500,
                                    content={"detail": "internal server error",
                                             "request_id": request_id})
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")  # bounded label values
        if endpoint != "/metrics":
            REQUESTS.labels(request.method, endpoint, str(status)).inc()
            LATENCY.labels(endpoint).observe(duration)
            log(logging.INFO, "request", request_id=request_id,
                method=request.method, path=endpoint, status=status,
                duration_ms=round(duration * 1000, 2),
                client=request.client.host if request.client else None)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        VALIDATION_ERRORS.inc()
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body",
                   "error": e["msg"]} for e in exc.errors()]
        log(logging.WARNING, "validation failed", path=request.url.path,
            fields=[e["field"] for e in errors])
        return JSONResponse(status_code=422,
                            content={"detail": "invalid input", "errors": errors})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        if request.app.state.model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ready",
                "model_name": request.app.state.meta.get("model_name")}

    @app.get("/model-info")
    def model_info(request: Request):
        meta = request.app.state.meta
        keys = ["model_name", "model_version", "selection_rule", "metrics",
                "features", "decision_threshold", "library_versions",
                "trained_utc"]
        return {k: meta.get(k) for k in keys}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientFeatures, request: Request):
        model, meta = request.app.state.model, request.app.state.meta
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        result = predict_records(model, [patient.model_dump()])[0]
        PREDICTIONS.labels(str(result["prediction"])).inc()
        PROBABILITY.observe(result["probability_disease"])
        return PredictionResponse(
            **result,
            model_name=meta.get("model_name", "unknown"),
            model_version=meta.get("model_version", "unknown"),
        )

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
