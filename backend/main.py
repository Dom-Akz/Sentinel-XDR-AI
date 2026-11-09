"""FastAPI application: the detection engine.

Run from the project root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Interactive API docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from .model import IDSModel, ModelNotAvailable
from .preprocessing import InputValidationError
from .schemas import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_ROWS = int(os.environ.get("IDS_MAX_ROWS", "200000"))
MAX_UPLOAD_BYTES = int(os.environ.get("IDS_MAX_UPLOAD_MB", "100")) * 1024 * 1024


def model_dir() -> Path:
    return Path(os.environ.get("IDS_MODEL_DIR", PROJECT_ROOT / "models"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.load_error = None
    try:
        app.state.model = IDSModel.load(model_dir())
        app.state.model.warm_up()
    except ModelNotAvailable as exc:
        app.state.load_error = str(exc)
    yield


def get_model(request: Request) -> IDSModel:
    model = request.app.state.model
    if model is None:
        raise ModelNotAvailable(request.app.state.load_error or "Model not loaded.")
    return model


def create_app() -> FastAPI:
    app = FastAPI(
        title="Network IDS API",
        version="1.0.0",
        description="Classifies network flows as Normal or Attack, with the attack category, "
                    "a confidence score and a SHAP-based explanation.",
        lifespan=lifespan,
    )

    @app.exception_handler(InputValidationError)
    async def _validation_handler(_: Request, exc: InputValidationError):
        return JSONResponse(status_code=422, content={"detail": {"message": exc.message, **exc.details}})

    @app.exception_handler(ModelNotAvailable)
    async def _model_handler(_: Request, exc: ModelNotAvailable):
        return JSONResponse(status_code=503, content={"detail": {"message": str(exc)}})

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health(request: Request):
        model = request.app.state.model
        if model is None:
            return HealthResponse(status="degraded", model_loaded=False, detail=request.app.state.load_error)
        md = model.metadata
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model=model.name,
            trained_on=md.get("dataset_name"),
            synthetic_data=bool(md.get("is_synthetic", False)),
        )

    @app.get("/model/info", tags=["system"])
    def model_info(request: Request):
        return get_model(request).info()

    @app.post("/predict", response_model=PredictResponse, tags=["detection"])
    def predict(req: PredictRequest, request: Request):
        """Classify flows sent as JSON records (column name -> value)."""
        model = get_model(request)
        df = pd.DataFrame.from_records(req.flows)
        return model.predict(df, req.threshold, req.explain_limit)

    @app.post("/predict/csv", response_model=PredictResponse, tags=["detection"])
    def predict_csv(
        request: Request,
        file: UploadFile = File(..., description="CSV file with one flow per row"),
        threshold: float | None = Query(None, ge=0, le=1),
        explain_limit: int = Query(25, ge=0, le=500),
    ):
        """Classify every flow of an uploaded CSV file."""
        model = get_model(request)
        content = file.file.read()
        if not content.strip():
            raise HTTPException(status_code=400, detail={"message": "The uploaded file is empty."})
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"message": f"File too large (limit {MAX_UPLOAD_BYTES // 1024 // 1024} MB)."},
            )
        try:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise HTTPException(
                status_code=400, detail={"message": f"The file is not a readable CSV: {exc}"}
            ) from exc
        if len(df) > MAX_ROWS:
            raise HTTPException(
                status_code=413,
                detail={"message": f"Too many rows ({len(df)}). Limit is {MAX_ROWS}; split the file."},
            )
        return model.predict(df, threshold, explain_limit)

    @app.post("/explain", response_model=ExplainResponse, tags=["detection"])
    def explain(req: ExplainRequest, request: Request):
        """Explain one flow: SHAP contributions for the predicted (or a chosen) class."""
        model = get_model(request)
        df = pd.DataFrame.from_records([req.flow])
        return model.explain(df, req.top_k, req.target_class, req.threshold)

    return app


app = create_app()
