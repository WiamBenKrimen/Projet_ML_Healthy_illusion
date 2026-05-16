#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_collection.py
Projet Machine Learning — Healthy Illusion

Objectif :
- Collecter des produits alimentaires depuis l'API Open Food Facts.
- Construire un dataset tabulaire pour une classification supervisée binaire.
- Créer la cible principale bad_nutrition à partir du Nutri-Score.
- Créer image_saine et healthy_illusion pour l'analyse métier.
- Exporter uniquement les fichiers demandés :
  data/raw/*.json
  data/dataset.csv
  data/sample.csv
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ==========================================================
# 1. Configuration générale
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = DATA_DIR / "dataset.csv"
SAMPLE_PATH = DATA_DIR / "sample.csv"

BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"

HEADERS = {
    "User-Agent": "HealthyIllusionMLProject/1.0 (student project)",
    "Accept": "application/json",
}

PAGE_SIZE = 200
SLEEP_BETWEEN_REQUESTS = 1.5
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 10

MIN_FINAL_ROWS = 10_000
TARGET_RAW_PRODUCTS = 40_000
MAX_PAGES_PER_CATEGORY = 50

MIN_MINORITY_RATIO = 5.0
MAX_MINORITY_RATIO = 25.0

COLLECTION_CATEGORIES = [
    # Produits emballés avec souvent un Nutri-Score acceptable
    "waters",
    "unsweetened-beverages",
    "milk",
    "plain-yogurts",
    "plant-based-beverages",
    "plant-based-foods",
    "wholemeal-breads",
    "soups",
    "vegetable-soups",
    "canned-vegetables",
    "frozen-vegetables",
    "fruit-compotes",
    "pastas",
    "rice",
    "couscous",
    "oatmeal",
    "rolled-oats",
    "canned-legumes",
    "canned-beans",
    "lentils",
    "canned-fish",
    "tuna",
    "tomato-sauces",

    # Produits associés à une image nutritionnelle positive
    "breakfast-cereals",
    "muesli",
    "granola",
    "yogurts",
    "fermented-milk-products",
    "fruit-juices",
    "smoothies",
    "cereal-bars",
    "protein-bars",
    "energy-bars",
    "sports-nutrition",
    "organic-foods",

    # Produits de contraste
    "biscuits",
    "chocolates",
]

FIELDS = [
    "code",
    "product_name",
    "brands",
    "categories_tags",
    "labels_tags",
    "countries_tags",
    "nutriscore_grade",
    "nutrition_grades",
    "nutriments",
    "additives_n",
    "additives_tags",
]

BASE_PARAMS = {
    "action": "process",
    "json": 1,
    "page_size": PAGE_SIZE,
    "fields": ",".join(FIELDS),
    "sort_by": "unique_scans_n",
}

HEALTHY_IMAGE_CATEGORY_KEYWORDS = {
    "muesli",
    "granola",
    "breakfast-cereals",
    "cereals",
    "yogurts",
    "fermented-milk-products",
    "smoothies",
    "fruit-juices",
    "protein-bars",
    "energy-bars",
    "cereal-bars",
    "sports-nutrition",
    "diet-products",
    "light-products",
    "plant-based-foods",
    "organic-foods",
}

HEALTHY_IMAGE_LABEL_KEYWORDS = {
    "organic",
    "bio",
    "no-added-sugar",
    "low-fat",
    "reduced-fat",
    "high-protein",
    "source-of-fibre",
    "rich-in-fibre",
    "natural",
}

VALID_NUTRISCORES = {"a", "b", "c", "d", "e"}


# ============================================================
# 2. Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ============================================================
# 3. Fonctions utilitaires
# ============================================================

def safe_get_list(product: dict[str, Any], key: str) -> list[str]:
    value = product.get(key, [])
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).lower().strip() for item in value if item is not None]
    if isinstance(value, str):
        return [value.lower().strip()]
    return []


def normalize_tag(tag: str) -> str:
    tag = str(tag).lower().strip()
    if ":" in tag:
        tag = tag.split(":", 1)[1]
    return tag


def contains_keyword(tags: list[str], keywords: set[str]) -> bool:
    normalized_tags = [normalize_tag(tag) for tag in tags]
    return any(keyword in tag for tag in normalized_tags for keyword in keywords)


def first_normalized_tag(tags: list[str], default: str = "unknown") -> str:
    if not tags:
        return default
    return normalize_tag(tags[0])


def get_nutrient(nutriments: dict[str, Any], key: str) -> float | None:
    value = nutriments.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_nutriscore(product: dict[str, Any]) -> str:
    grade = product.get("nutriscore_grade") or product.get("nutrition_grades") or ""
    return str(grade).lower().strip()


def build_bad_nutrition(nutriscore_grade: str) -> int | None:
    if nutriscore_grade not in VALID_NUTRISCORES:
        return None
    return int(nutriscore_grade in {"d", "e"})


def build_image_saine(categories_tags: list[str], labels_tags: list[str]) -> int:
    has_healthy_category = contains_keyword(categories_tags, HEALTHY_IMAGE_CATEGORY_KEYWORDS)
    has_healthy_label = contains_keyword(labels_tags, HEALTHY_IMAGE_LABEL_KEYWORDS)
    return int(has_healthy_category or has_healthy_label)


def get_additives_count(product: dict[str, Any]) -> int:
    additives_n = product.get("additives_n")
    try:
        if additives_n is not None:
            return int(additives_n)
    except (TypeError, ValueError):
        pass
    return len(safe_get_list(product, "additives_tags"))


def product_matches_category(product: dict[str, Any], category: str) -> bool:
    categories_tags = safe_get_list(product, "categories_tags")
    normalized_categories = [normalize_tag(tag) for tag in categories_tags]
    return any(category in tag for tag in normalized_categories)


# ============================================================
# 4. Collecte depuis l'API
# ============================================================

def fetch_page(category: str, page: int) -> dict[str, Any] | None:
    params = {
        **BASE_PARAMS,
        "page": page,
        "tagtype_0": "categories",
        "tag_contains_0": "contains",
        "tag_0": category,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=40)

            logger.info(
                "API | category=%s | page=%s | attempt=%s/%s | status=%s",
                category,
                page,
                attempt,
                MAX_RETRIES,
                response.status_code,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 403:
                logger.warning("Accès refusé pour category=%s, page=%s.", category, page)
                return None

            if response.status_code in {429, 500, 502, 503, 504}:
                logger.warning(
                    "Erreur temporaire %s | pause %ss avant nouvelle tentative.",
                    response.status_code,
                    RETRY_SLEEP_SECONDS,
                )
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            response.raise_for_status()

        except requests.RequestException as error:
            logger.warning(
                "Erreur réseau | category=%s | page=%s | attempt=%s/%s | %s",
                category,
                page,
                attempt,
                MAX_RETRIES,
                error,
            )
            time.sleep(RETRY_SLEEP_SECONDS)

        except json.JSONDecodeError as error:
            logger.warning("Réponse JSON invalide | category=%s | page=%s | %s", category, page, error)
            return None

    return None


def save_raw_response(category: str, page: int, data: dict[str, Any]) -> None:
    output_path = RAW_DIR / f"{category.replace('/', '_')}_page_{page}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def collect_products() -> list[dict[str, Any]]:
    all_products: list[dict[str, Any]] = []
    seen_codes: set[str] = set()

    logger.info("=" * 70)
    logger.info("Début de collecte Open Food Facts")
    logger.info("Objectif brut : %s produits", TARGET_RAW_PRODUCTS)
    logger.info("Nombre de catégories : %s", len(COLLECTION_CATEGORIES))
    logger.info("=" * 70)

    for category in COLLECTION_CATEGORIES:
        logger.info("Catégorie : %s", category)

        for page in range(1, MAX_PAGES_PER_CATEGORY + 1):
            if len(all_products) >= TARGET_RAW_PRODUCTS:
                logger.info("Objectif brut atteint.")
                return all_products

            data = fetch_page(category, page)
            if data is None:
                logger.warning("Page ignorée | category=%s | page=%s", category, page)
                break

            products = data.get("products", [])
            if not products:
                logger.info("Aucun produit restant | category=%s | page=%s", category, page)
                break

            matching_products = [
                product for product in products
                if product_matches_category(product, category)
            ]

            if not matching_products:
                logger.warning("Aucun produit cohérent avec la catégorie | category=%s | page=%s", category, page)
                break

            data_to_save = dict(data)
            data_to_save["products"] = matching_products
            save_raw_response(category, page, data_to_save)

            added = 0
            for product in matching_products:
                code = str(product.get("code", "")).strip()
                if not code or code in seen_codes:
                    continue

                seen_codes.add(code)
                all_products.append(product)
                added += 1

            logger.info(
                "Progression | total=%s | category=%s | page=%s | ajoutés=%s",
                len(all_products),
                category,
                page,
                added,
            )

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    return all_products


# ============================================================
# 5. Transformation en dataset tabulaire
# ============================================================

def product_to_row(product: dict[str, Any]) -> dict[str, Any] | None:
    nutriscore_grade = get_nutriscore(product)
    bad_nutrition = build_bad_nutrition(nutriscore_grade)

    if bad_nutrition is None:
        return None

    nutriments = product.get("nutriments", {}) or {}
    if not isinstance(nutriments, dict):
        nutriments = {}

    categories_tags = safe_get_list(product, "categories_tags")
    labels_tags = safe_get_list(product, "labels_tags")
    countries_tags = safe_get_list(product, "countries_tags")

    image_saine = build_image_saine(categories_tags, labels_tags)
    healthy_illusion = int(image_saine == 1 and bad_nutrition == 1)

    return {
        "code": product.get("code"),
        "product_name": product.get("product_name"),
        "brands": product.get("brands"),
        "main_category": first_normalized_tag(categories_tags),
        "country": first_normalized_tag(countries_tags),
        "sugars_100g": get_nutrient(nutriments, "sugars_100g"),
        "fat_100g": get_nutrient(nutriments, "fat_100g"),
        "saturated_fat_100g": get_nutrient(nutriments, "saturated-fat_100g"),
        "salt_100g": get_nutrient(nutriments, "salt_100g"),
        "fiber_100g": get_nutrient(nutriments, "fiber_100g"),
        "proteins_100g": get_nutrient(nutriments, "proteins_100g"),
        "energy_kcal_100g": get_nutrient(nutriments, "energy-kcal_100g"),
        "additives_count": get_additives_count(product),
        "has_labels": int(len(labels_tags) > 0),
        "image_saine": image_saine,
        "nutriscore_grade": nutriscore_grade,
        "bad_nutrition": bad_nutrition,
        "healthy_illusion": healthy_illusion,
    }


def build_dataframe(products: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [row for product in products if (row := product_to_row(product)) is not None]
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("Aucune ligne exploitable après transformation.")

    df = df.drop_duplicates(subset=["code"]).copy()

    numeric_columns = [
        "sugars_100g",
        "fat_100g",
        "saturated_fat_100g",
        "salt_100g",
        "fiber_100g",
        "proteins_100g",
        "energy_kcal_100g",
        "additives_count",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["missing_nutrition_values"] = df[numeric_columns].isna().sum(axis=1)
    df = df[df["missing_nutrition_values"] <= 2].copy()
    df = df.drop(columns=["missing_nutrition_values"])

    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].median())

    df["main_category"] = df["main_category"].fillna("unknown").astype(str)
    df["country"] = df["country"].fillna("unknown").astype(str)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================
# 6. Vérification et export
# ============================================================

def minority_ratio_percent(series: pd.Series) -> float:
    distribution = series.value_counts(normalize=True)
    if len(distribution) < 2:
        return 0.0
    return float(distribution.min() * 100)


def check_conformity(df: pd.DataFrame) -> dict[str, Any]:
    feature_columns = [
        "sugars_100g",
        "fat_100g",
        "saturated_fat_100g",
        "salt_100g",
        "fiber_100g",
        "proteins_100g",
        "energy_kcal_100g",
        "additives_count",
        "main_category",
        "country",
        "has_labels",
        "image_saine",
    ]

    ratio = minority_ratio_percent(df["bad_nutrition"])

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "feature_count": len(feature_columns),
        "minority_ratio": round(ratio, 2),
        "rows_ok": len(df) >= MIN_FINAL_ROWS,
        "features_ok": len(feature_columns) >= 8,
        "ratio_ok": MIN_MINORITY_RATIO <= ratio <= MAX_MINORITY_RATIO,
        "target_ok": "bad_nutrition" in df.columns,
    }


def export_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_final = df.copy()

    df_final.to_csv(DATASET_PATH, index=False, encoding="utf-8")

    sample_size = min(100, len(df_final))
    df_final.sample(n=sample_size, random_state=42).to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8",
    )

    return df_final


def print_summary(df: pd.DataFrame, checks: dict[str, Any]) -> None:
    logger.info("=" * 70)
    logger.info("Résumé du dataset final")
    logger.info("Dimensions : %s lignes × %s colonnes", df.shape[0], df.shape[1])

    logger.info("Distribution bad_nutrition :")
    logger.info("\n%s", df["bad_nutrition"].value_counts())
    logger.info("Distribution bad_nutrition en pourcentage :")
    logger.info("\n%s", (df["bad_nutrition"].value_counts(normalize=True) * 100).round(2))

    logger.info("image_saine=1 : %s", int(df["image_saine"].sum()))
    logger.info("healthy_illusion=1 : %s", int(df["healthy_illusion"].sum()))

    logger.info("=" * 70)
    logger.info("Vérification des contraintes")
    logger.info("Lignes >= 10 000 : %s", "OK" if checks["rows_ok"] else "NON")
    logger.info("Features >= 8 : %s", "OK" if checks["features_ok"] else "NON")
    logger.info("Classe minoritaire entre 5%% et 25%% : %s (%s%%)", "OK" if checks["ratio_ok"] else "NON", checks["minority_ratio"])
    logger.info("Cible bad_nutrition présente : %s", "OK" if checks["target_ok"] else "NON")
    logger.info("=" * 70)


# ============================================================
# 7. Programme principal
# ============================================================

def main() -> None:
    logger.info("Démarrage du script data_collection.py")

    products = collect_products()
    logger.info("Produits bruts uniques collectés : %s", len(products))

    if not products:
        logger.error("Aucun produit collecté.")
        return

    df = build_dataframe(products)
    logger.info("Après nettoyage : %s lignes", len(df))

    df_final = export_dataset(df)
    checks = check_conformity(df_final)
    print_summary(df_final, checks)

    logger.info("Dataset sauvegardé : %s", DATASET_PATH)
    logger.info("Sample sauvegardé : %s", SAMPLE_PATH)
    logger.info("Fin du script.")


if __name__ == "__main__":
    main()
