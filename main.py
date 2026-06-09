"""FastAPI service for the Healthy Illusion final model."""

from __future__ import annotations

import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from src.features import RAW_INPUT_FEATURES, prepare_model_input

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/final_model.joblib"))
artifact = joblib.load(MODEL_PATH)
model = artifact["pipeline"]
threshold = float(artifact["threshold"])


class NutritionInput(BaseModel):
    """Raw product information required for a prediction."""

    sugars_100g: float = Field(..., ge=0, le=100)
    fat_100g: float = Field(..., ge=0, le=100)
    saturated_fat_100g: float = Field(..., ge=0, le=100)
    salt_100g: float = Field(..., ge=0, le=100)
    fiber_100g: float = Field(..., ge=0, le=100)
    proteins_100g: float = Field(..., ge=0, le=100)
    energy_kcal_100g: float = Field(..., ge=0, le=1000)
    additives_count: int = Field(..., ge=0, le=100)
    main_category: str = Field(..., min_length=1)
    country: str = Field(..., min_length=1)
    has_labels: Literal[0, 1]
    image_saine: Literal[0, 1]

    @field_validator("saturated_fat_100g")
    @classmethod
    def validate_saturated_fat(cls, value, info):
        fat = info.data.get("fat_100g")
        if fat is not None and value > fat:
            raise ValueError("saturated_fat_100g cannot exceed fat_100g")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {
                "sugars_100g": 25.0,
                "fat_100g": 12.0,
                "saturated_fat_100g": 5.0,
                "salt_100g": 0.8,
                "fiber_100g": 1.5,
                "proteins_100g": 4.0,
                "energy_kcal_100g": 450.0,
                "additives_count": 3,
                "main_category": "snacks",
                "country": "france",
                "has_labels": 0,
                "image_saine": 1,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float
    confidence: Literal["high", "medium", "low"]
    risk_level: str


app = FastAPI(
    title="Healthy Illusion - API nutritionnelle",
    description="Detection de produits a mauvaise qualite nutritionnelle.",
    version=artifact.get("model_version", "2.0.0"),
    contact={"name": "Equipe ML - ENSA Tetouan"},
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    logger.info(
        "%s %s -> %s in %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        perf_counter() - started,
    )
    return response


def get_confidence(probability: float, decision_threshold: float) -> str:
    distance = abs(probability - decision_threshold)
    if distance >= 0.30:
        return "high"
    if distance >= 0.15:
        return "medium"
    return "low"


def predict_frame(raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    numeric_bounds = {
        "sugars_100g": 100,
        "fat_100g": 100,
        "saturated_fat_100g": 100,
        "salt_100g": 100,
        "fiber_100g": 100,
        "proteins_100g": 100,
        "energy_kcal_100g": 1000,
        "additives_count": 100,
    }
    for column, upper_bound in numeric_bounds.items():
        values = pd.to_numeric(raw_data[column], errors="coerce")
        invalid_conversion = raw_data[column].notna() & values.isna()
        if invalid_conversion.any() or ((values < 0) | (values > upper_bound)).any():
            raise ValueError(f"{column} doit etre compris entre 0 et {upper_bound}")
    if (
        pd.to_numeric(raw_data["saturated_fat_100g"])
        > pd.to_numeric(raw_data["fat_100g"])
    ).any():
        raise ValueError("saturated_fat_100g ne peut pas depasser fat_100g")
    for column in ("has_labels", "image_saine"):
        values = pd.to_numeric(raw_data[column], errors="coerce")
        invalid_conversion = raw_data[column].notna() & values.isna()
        if invalid_conversion.any() or not set(values.dropna().unique()) <= {0, 1}:
            raise ValueError(f"{column} doit valoir 0 ou 1")
    features = prepare_model_input(raw_data)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return probabilities, predictions


def predict_single(data: dict) -> dict:
    probabilities, predictions = predict_frame(pd.DataFrame([data]))
    probability = float(probabilities[0])
    prediction = int(predictions[0])
    return {
        "prediction": "mauvaise_nutrition" if prediction else "bonne_nutrition",
        "probability": round(probability, 4),
        "threshold": threshold,
        "confidence": get_confidence(probability, threshold),
        "risk_level": "RISQUE ELEVE" if prediction else "RISQUE FAIBLE",
    }


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Bienvenue sur l'API Healthy Illusion",
        "version": artifact.get("model_version", "2.0.0"),
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }


@app.get("/health", tags=["General"])
def health():
    return {
        "status": "ok",
        "model": "loaded",
        "threshold": threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info", tags=["Modele"])
def model_info():
    return {
        "model_type": "XGBoost (pipeline sklearn)",
        "version": artifact.get("model_version", "2.0.0"),
        "trained_at": artifact.get("trained_at"),
        "threshold": threshold,
        "cost_config": artifact["cost_config"],
        "metrics_default": artifact["metrics_default_threshold"],
        "metrics_optimal": artifact["metrics_optimal_threshold"],
        "cv_results": artifact.get("cv_results", {}),
        "best_params": artifact.get("best_params", {}),
        "imbalance_strategy": artifact.get("imbalance_strategy"),
        "raw_input_features": RAW_INPUT_FEATURES,
        "allowed_categories": artifact.get("allowed_categories", {}),
        "target": "bad_nutrition (0=bonne, 1=mauvaise)",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(input_data: NutritionInput):
    try:
        return predict_single(input_data.model_dump())
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(file: UploadFile = File(..., description="CSV de produits")):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit etre un CSV.")
    try:
        dataframe = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"CSV illisible: {error}") from error

    missing = [column for column in RAW_INPUT_FEATURES if column not in dataframe.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes: {missing}")
    try:
        probabilities, predictions = predict_frame(dataframe[RAW_INPUT_FEATURES])
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Donnees invalides: {error}") from error

    dataframe["probability"] = np.round(probabilities, 4)
    dataframe["prediction"] = np.where(
        predictions == 1, "mauvaise_nutrition", "bonne_nutrition"
    )
    dataframe["risk_level"] = np.where(
        predictions == 1, "RISQUE ELEVE", "RISQUE FAIBLE"
    )
    dataframe["confidence"] = [
        get_confidence(float(probability), threshold) for probability in probabilities
    ]
    output = io.StringIO()
    dataframe.to_csv(output, index=False)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"},
    )
