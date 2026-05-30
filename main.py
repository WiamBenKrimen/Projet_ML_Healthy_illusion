"""
API FastAPI — Projet ML Healthy Illusion
Détection de produits à mauvaise nutrition
Phase 4 — Déploiement
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import numpy as np
import io
import logging
from datetime import datetime
from pathlib import Path

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Chargement du modèle UNE SEULE FOIS au démarrage ───────────────────────
MODEL_PATH = Path("models/final_model.joblib")

try:
    artifact  = joblib.load(MODEL_PATH)
    model     = artifact["pipeline"]
    threshold = artifact["threshold"]
    cost_cfg  = artifact["cost_config"]
    metrics   = artifact["metrics_optimal_threshold"]
    logger.info(f"Modèle chargé — seuil={threshold} | F1={metrics['f1']}")
except Exception as e:
    logger.error(f"Erreur chargement modèle : {e}")
    raise RuntimeError(f"Impossible de charger {MODEL_PATH} : {e}")

# Colonnes exactes attendues par le pipeline (dans l'ordre du ColumnTransformer)
NUMERIC_COLS = [
    "sugars_100g", "fat_100g", "saturated_fat_100g", "salt_100g",
    "fiber_100g", "proteins_100g", "energy_kcal_100g", "additives_count",
    "sugar_energy_ratio", "sat_fat_ratio", "salt_sugar_interaction",
    "nutrition_risk_score"
]
CATEGORICAL_COLS = ["main_category", "country", "has_labels", "image_saine"]
ALL_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# ─── Application FastAPI ─────────────────────────────────────────────────────
app = FastAPI(
    title="Healthy Illusion — API de détection nutritionnelle",
    description=(
        "Prédit si un produit alimentaire présente une **mauvaise nutrition** "
        "à partir de ses valeurs nutritionnelles. "
        "Modèle : XGBoost tuné | Métrique principale : F1-score."
    ),
    version="1.0.0",
    contact={"name": "Équipe ML — ENSA Tétouan"},
)

# ─── Schémas Pydantic ────────────────────────────────────────────────────────

class NutritionInput(BaseModel):
    """Features d'un produit alimentaire — colonnes exactes du pipeline."""
    # Numériques
    sugars_100g:             float = Field(..., ge=0, le=100,  description="Sucres (g/100g)")
    fat_100g:                float = Field(..., ge=0, le=100,  description="Graisses totales (g/100g)")
    saturated_fat_100g:      float = Field(..., ge=0, le=100,  description="Graisses saturées (g/100g)")
    salt_100g:               float = Field(..., ge=0, le=100,  description="Sel (g/100g)")
    fiber_100g:              float = Field(..., ge=0, le=100,  description="Fibres (g/100g)")
    proteins_100g:           float = Field(..., ge=0, le=100,  description="Protéines (g/100g)")
    energy_kcal_100g:        float = Field(..., ge=0, le=1000, description="Énergie (kcal/100g)")
    additives_count:         int   = Field(..., ge=0, le=50,   description="Nombre d'additifs")
    sugar_energy_ratio:      float = Field(..., ge=0,          description="Ratio sucres/énergie (feature engineered)")
    sat_fat_ratio:           float = Field(..., ge=0,          description="Ratio graisses saturées/graisses totales")
    salt_sugar_interaction:  float = Field(..., ge=0,          description="Interaction sel × sucres")
    nutrition_risk_score:    float = Field(...,                description="Score de risque nutritionnel global")
    # Catégorielles
    main_category:           str   = Field(...,                description="Catégorie principale du produit")
    country:                 str   = Field(...,                description="Pays d'origine")
    has_labels:              str   = Field(...,                description="Présence de labels (ex: 'yes'/'no')")
    image_saine:             str   = Field(...,                description="Image saine (ex: 'yes'/'no')")

    @field_validator("saturated_fat_100g")
    @classmethod
    def sat_fat_lte_fat(cls, v, info):
        fat = info.data.get("fat_100g")
        if fat is not None and v > fat:
            raise ValueError("saturated_fat_100g ne peut pas dépasser fat_100g")
        return v

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
                "sugar_energy_ratio": 0.22,
                "sat_fat_ratio": 0.42,
                "salt_sugar_interaction": 20.0,
                "nutrition_risk_score": 3.5,
                "main_category": "Snacks",
                "country": "France",
                "has_labels": "no",
                "image_saine": "no"
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction:  str   = Field(..., description="'mauvaise_nutrition' ou 'bonne_nutrition'")
    probability: float = Field(..., description="Probabilité de mauvaise nutrition")
    threshold:   float = Field(..., description="Seuil de décision appliqué")
    confidence:  str   = Field(..., description="'high' | 'medium' | 'low'")
    risk_level:  str   = Field(..., description="Niveau de risque lisible")


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def get_confidence(prob: float, thresh: float) -> str:
    distance = abs(prob - thresh)
    if distance >= 0.3:  return "high"
    if distance >= 0.15: return "medium"
    return "low"


def predict_single(data: dict) -> dict:
    df   = pd.DataFrame([data])[ALL_COLS]
    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= threshold)
    return {
        "prediction":  "mauvaise_nutrition" if pred == 1 else "bonne_nutrition",
        "probability": round(prob, 4),
        "threshold":   threshold,
        "confidence":  get_confidence(prob, threshold),
        "risk_level":  "RISQUE ÉLEVÉ 🔴" if pred == 1 else "RISQUE FAIBLE 🟢",
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Général"])
def root():
    """Page d'accueil — informations sur l'API."""
    return {
        "message":    "Bienvenue sur l'API Healthy Illusion",
        "version":    "1.0.0",
        "docs":       "/docs",
        "health":     "/health",
        "model_info": "/model/info",
    }


@app.get("/health", tags=["Général"])
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    return {
        "status":    "ok",
        "model":     "loaded",
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/info", tags=["Modèle"])
def model_info():
    """Métadonnées du modèle : type, métriques, seuil, coût."""
    return {
        "model_type":         "XGBoost (pipeline sklearn)",
        "version":            "1.0.0",
        "threshold":          threshold,
        "cost_config":        cost_cfg,
        "metrics_default":    artifact.get("metrics_default_threshold", {}),
        "metrics_optimal":    metrics,
        "features_numeric":   NUMERIC_COLS,
        "features_categorical": CATEGORICAL_COLS,
        "target":             "bad_nutrition (0=bonne, 1=mauvaise)",
        "imbalance_strategy": "scale_pos_weight=4.71",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict(input_data: NutritionInput):
    """
    Prédiction unitaire : reçoit les features d'un produit,
    retourne la classe + probabilité + niveau de risque.
    """
    logger.info(f"Prédiction unitaire — category={input_data.main_category}")
    try:
        result = predict_single(input_data.model_dump())
        logger.info(f"Résultat : {result['prediction']} (prob={result['probability']})")
        return result
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.post("/predict/batch", tags=["Prédiction"])
def predict_batch(file: UploadFile = File(..., description="Fichier CSV avec les colonnes features")):
    """
    Prédiction par lot : reçoit un fichier CSV,
    retourne un CSV enrichi avec prediction, probability, risk_level.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un CSV (.csv)")

    logger.info(f"Batch reçu : {file.filename}")

    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lecture CSV : {str(e)}")

    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Colonnes manquantes : {missing}")

    try:
        probs = model.predict_proba(df[ALL_COLS])[:, 1]
        preds = (probs >= threshold).astype(int)
        df["probability"] = np.round(probs, 4)
        df["prediction"]  = ["mauvaise_nutrition" if p == 1 else "bonne_nutrition" for p in preds]
        df["risk_level"]  = ["RISQUE ÉLEVÉ" if p == 1 else "RISQUE FAIBLE" for p in preds]
        df["confidence"]  = [get_confidence(p, threshold) for p in probs]
    except Exception as e:
        logger.error(f"Erreur batch : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur prédiction : {str(e)}")

    logger.info(f"Batch OK : {len(df)} lignes, {preds.sum()} mauvaise_nutrition")

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"}
    )