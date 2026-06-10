"""Shared feature contract used during training and inference."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

RAW_NUMERIC_FEATURES = [
    "sugars_100g",
    "fat_100g",
    "saturated_fat_100g",
    "salt_100g",
    "fiber_100g",
    "proteins_100g",
    "energy_kcal_100g",
    "additives_count",
]

ENGINEERED_FEATURES = [
    "sugar_energy_ratio",
    "sat_fat_ratio",
    "salt_sugar_interaction",
    "nutrition_risk_score",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["main_category", "country", "has_labels", "image_saine"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_INPUT_FEATURES = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def normalize_category(value: object) -> str:
    """Normalize UI/API category values to the training representation."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the four business features used by the model."""
    result = data.copy()
    epsilon = 1e-6

    result["sugar_energy_ratio"] = (
        result["sugars_100g"] / (result["energy_kcal_100g"] + epsilon)
    )
    result["sat_fat_ratio"] = (
        result["saturated_fat_100g"] / (result["fat_100g"] + epsilon)
    )
    result["salt_sugar_interaction"] = result["salt_100g"] * result["sugars_100g"]
    result["nutrition_risk_score"] = (
        result["sugars_100g"]
        + result["saturated_fat_100g"]
        + result["salt_100g"] * 10
        + result["energy_kcal_100g"] / 100
        - result["fiber_100g"]
    )

    result[ENGINEERED_FEATURES] = result[ENGINEERED_FEATURES].replace(
        [np.inf, -np.inf], np.nan
    )
    return result


def prepare_model_input(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw input and derive all columns expected by the pipeline."""
    result = data.copy()
    for column in ("main_category", "country"):
        result[column] = result[column].map(normalize_category)
    for column in ("has_labels", "image_saine"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    return add_engineered_features(result)[MODEL_FEATURES]
