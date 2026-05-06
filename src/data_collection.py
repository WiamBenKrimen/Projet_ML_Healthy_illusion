#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_collection.py
Projet Machine Learning — Healthy Illusion / Bad Nutrition

Nouvelle conception validée :
1) La cible ML principale est bad_nutrition.
   bad_nutrition = 1 si Nutri-Score D ou E
   bad_nutrition = 0 si Nutri-Score A, B ou C

2) nutriscore_grade vient de l'API Open Food Facts.
   Il sert seulement à créer la cible, puis il ne doit pas être utilisé
   comme feature d'entrée du modèle.

3) image_saine est une variable métier construite à partir des catégories
   et labels du produit. Elle veut dire "produit présenté/perçu comme sain",
   pas "produit réellement sain".

4) healthy_illusion est une conclusion métier :
   healthy_illusion = 1 si image_saine = 1 ET bad_nutrition = 1

Sorties :
- data/raw/*.json
- data/processed/dataset.csv
- data/processed/sample.csv
- data/processed/verification_dataset.csv
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROCESSED_DIR / "dataset.csv"
SAMPLE_PATH = PROCESSED_DIR / "sample.csv"
VERIFICATION_PATH = PROCESSED_DIR / "verification_dataset.csv"

BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"

# Important : mets ton vrai email ou l'email du groupe si possible.
# Open Food Facts demande un User-Agent personnalisé pour les appels API.
HEADERS = {
    "User-Agent": "HealthyIllusionMLProject/1.0 (student project; contact: student@example.com)",
    "From": "bbenkrimen@gmail.com",
    "Accept": "application/json",
}

PAGE_SIZE = 100
SLEEP_BETWEEN_REQUESTS = 5.0
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 30

MIN_FINAL_ROWS = 10_000
TARGET_RAW_PRODUCTS = 35_000
MAX_PAGES_PER_CATEGORY = 80

MIN_MINORITY_RATIO = 5.0
MAX_MINORITY_RATIO = 25.0

# Laisse False au début.
# Si vous décidez que votre population d'étude = produits présentés comme sains,
# vous pouvez tester True, mais il faut le justifier dans le rapport.
KEEP_ONLY_IMAGE_SAINE = False

COLLECTION_CATEGORIES = [
    "breakfast-cereals", "muesli", "granola", "yogurts",
    "fermented-milk-products", "fruit-juices", "smoothies",
    "protein-bars", "energy-bars", "cereal-bars", "sports-nutrition",
    "plant-based-foods", "organic-foods", "milk", "cheeses", "breads",
    "biscuits", "chocolates", "sodas", "beverages", "salty-snacks",
    "ready-meals", "frozen-foods", "canned-foods", "condiments",
    "breakfasts", "desserts", "snacks", "groceries",
    "plant-based-beverages", "fruit-based-beverages", "vegetables", "fruits",
]

FIELDS = [
    "code", "product_name", "brands", "categories_tags", "labels_tags",
    "countries_tags", "nutriscore_grade", "nutrition_grades", "nutriments",
    "additives_n", "additives_tags",
]

BASE_PARAMS = {
    "action": "process",
    "json": 1,
    "page_size": PAGE_SIZE,
    "fields": ",".join(FIELDS),
    "sort_by": "unique_scans_n",
}

HEALTHY_IMAGE_CATEGORY_KEYWORDS = {
    "muesli", "granola", "breakfast-cereals", "cereals", "yogurts",
    "fermented-milk-products", "smoothies", "fruit-juices", "protein-bars",
    "energy-bars", "cereal-bars", "sports-nutrition", "diet-products",
    "light-products", "plant-based-foods", "organic-foods",
}

HEALTHY_IMAGE_LABEL_KEYWORDS = {
    "organic", "bio", "no-added-sugar", "low-fat", "reduced-fat",
    "high-protein", "source-of-fibre", "rich-in-fibre", "natural",
}

VALID_NUTRISCORES = {"a", "b", "c", "d", "e"}


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 3. FONCTIONS UTILITAIRES
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
    for tag in normalized_tags:
        for keyword in keywords:
            if keyword in tag:
                return True
    return False


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
    grade = (
        product.get("nutriscore_grade")
        or product.get("nutrition_grades")
        or product.get("nutrition_grade_fr")
        or ""
    )
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


# ============================================================
# 4. COLLECTE API
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
                "API call | category=%s | page=%s | attempt=%s/%s | status=%s",
                category, page, attempt, MAX_RETRIES, response.status_code,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code in {403, 429, 500, 502, 503, 504}:
                logger.warning("Erreur API %s | pause %ss puis retry", response.status_code, RETRY_SLEEP_SECONDS)
                time.sleep(RETRY_SLEEP_SECONDS)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            logger.warning(
                "Erreur réseau | category=%s | page=%s | attempt=%s/%s | error=%s",
                category, page, attempt, MAX_RETRIES, error,
            )
            time.sleep(RETRY_SLEEP_SECONDS)
        except json.JSONDecodeError as error:
            logger.warning("JSON invalide | category=%s | page=%s | error=%s", category, page, error)
            return None

    logger.warning("Page abandonnée après retries | category=%s | page=%s", category, page)
    return None


def save_raw_response(category: str, page: int, data: dict[str, Any]) -> None:
    output_path = RAW_DIR / f"{category.replace('/', '_')}_page_{page}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def collect_products() -> list[dict[str, Any]]:
    all_products: list[dict[str, Any]] = []
    seen_codes: set[str] = set()

    logger.info("=" * 70)
    logger.info("DÉBUT COLLECTE API")
    logger.info("Objectif brut: %s produits", TARGET_RAW_PRODUCTS)
    logger.info("Catégories: %s", len(COLLECTION_CATEGORIES))
    logger.info("=" * 70)

    for category in COLLECTION_CATEGORIES:
        logger.info("CATÉGORIE: %s", category)

        for page in range(1, MAX_PAGES_PER_CATEGORY + 1):
            if len(all_products) >= TARGET_RAW_PRODUCTS:
                logger.info("Objectif brut atteint.")
                return all_products

            data = fetch_page(category, page)
            if data is None:
                logger.warning("Page ignorée | category=%s | page=%s", category, page)
                continue

            products = data.get("products", [])
            if not products:
                logger.info("Plus de produits | category=%s | page=%s", category, page)
                break

            save_raw_response(category, page, data)

            added = 0
            for product in products:
                code = str(product.get("code", "")).strip()
                if not code or code in seen_codes:
                    continue
                seen_codes.add(code)
                all_products.append(product)
                added += 1

            logger.info(
                "Progression | total_unique=%s | category=%s | page=%s | added=%s",
                len(all_products), category, page, added,
            )
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        logger.info("Fin catégorie: %s | total_unique=%s", category, len(all_products))

    return all_products


# ============================================================
# 5. TRANSFORMATION
# ============================================================

def product_to_row(product: dict[str, Any]) -> dict[str, Any] | None:
    nutriscore_grade = get_nutriscore(product)
    bad_nutrition = build_bad_nutrition(nutriscore_grade)
    if bad_nutrition is None:
        return None

    nutriments = product.get("nutriments", {})
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
        "sugars_100g", "fat_100g", "saturated_fat_100g", "salt_100g",
        "fiber_100g", "proteins_100g", "energy_kcal_100g", "additives_count",
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

    if KEEP_ONLY_IMAGE_SAINE:
        df = df[df["image_saine"] == 1].copy()

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ============================================================
# 6. VÉRIFICATION ET EXPORT
# ============================================================

def minority_ratio_percent(series: pd.Series) -> float:
    distribution = series.value_counts(normalize=True)
    if len(distribution) < 2:
        return 0.0
    return float(distribution.min() * 100)


def reduce_to_target_rows_preserve_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MIN_FINAL_ROWS:
        return df

    sampled_parts = []
    for _, group in df.groupby("bad_nutrition"):
        n = max(1, round(len(group) / len(df) * MIN_FINAL_ROWS))
        sampled_parts.append(group.sample(n=min(n, len(group)), random_state=42))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > MIN_FINAL_ROWS:
        sampled = sampled.sample(n=MIN_FINAL_ROWS, random_state=42)
    return sampled.sample(frac=1, random_state=42).reset_index(drop=True)


def check_conformity(df: pd.DataFrame) -> dict[str, Any]:
    feature_columns = [
        "sugars_100g", "fat_100g", "saturated_fat_100g", "salt_100g",
        "fiber_100g", "proteins_100g", "energy_kcal_100g", "additives_count",
        "has_labels", "image_saine", "main_category", "country",
    ]
    ratio = minority_ratio_percent(df["bad_nutrition"])
    checks = {
        "rows": len(df),
        "columns": len(df.columns),
        "feature_count": len(feature_columns),
        "minority_ratio_bad_nutrition_percent": round(ratio, 2),
        "rows_ok": len(df) >= MIN_FINAL_ROWS,
        "feature_count_ok": len(feature_columns) >= 8,
        "minority_ratio_ok": MIN_MINORITY_RATIO <= ratio <= MAX_MINORITY_RATIO,
        "target_exists": "bad_nutrition" in df.columns,
    }
    checks["overall_ok"] = all([
        checks["rows_ok"], checks["feature_count_ok"],
        checks["minority_ratio_ok"], checks["target_exists"],
    ])
    return checks


def print_summary(df: pd.DataFrame, checks: dict[str, Any]) -> None:
    logger.info("=" * 70)
    logger.info("RÉSUMÉ DATASET FINAL")
    logger.info("=" * 70)
    logger.info("Dimensions: %s lignes × %s colonnes", df.shape[0], df.shape[1])

    logger.info("\nDistribution bad_nutrition:")
    logger.info("\n%s", df["bad_nutrition"].value_counts())
    logger.info("\nDistribution bad_nutrition en pourcentage:")
    logger.info("\n%s", (df["bad_nutrition"].value_counts(normalize=True) * 100).round(2))
    logger.info("\nDistribution image_saine:")
    logger.info("\n%s", df["image_saine"].value_counts())
    logger.info("\nDistribution healthy_illusion:")
    logger.info("\n%s", df["healthy_illusion"].value_counts())

    logger.info("=" * 70)
    logger.info("VÉRIFICATION CONFORMITÉ")
    logger.info("Lignes >= 10 000 : %s", "OK" if checks["rows_ok"] else "NON")
    logger.info("Features >= 8 : %s", "OK" if checks["feature_count_ok"] else "NON")
    logger.info(
        "Ratio minoritaire bad_nutrition entre 5%% et 25%% : %s (%s%%)",
        "OK" if checks["minority_ratio_ok"] else "NON",
        checks["minority_ratio_bad_nutrition_percent"],
    )
    logger.info("Target bad_nutrition existe : %s", "OK" if checks["target_exists"] else "NON")

    if checks["overall_ok"]:
        logger.info("✅ DATASET CONFORME AUX CONTRAINTES PRINCIPALES.")
    else:
        logger.warning("⚠️ DATASET PAS ENCORE CONFORME.")
        logger.warning("Si lignes < 10 000 : augmente TARGET_RAW_PRODUCTS ou MAX_PAGES_PER_CATEGORY.")
        logger.warning("Si ratio non conforme : ajuste COLLECTION_CATEGORIES, sans modifier les labels.")


def export_dataset(df: pd.DataFrame, checks: dict[str, Any]) -> None:
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    df.sample(n=min(100, len(df)), random_state=42).to_csv(SAMPLE_PATH, index=False, encoding="utf-8")
    pd.DataFrame([checks]).to_csv(VERIFICATION_PATH, index=False, encoding="utf-8")
    logger.info("Dataset sauvegardé: %s", DATASET_PATH)
    logger.info("Sample sauvegardé: %s", SAMPLE_PATH)
    logger.info("Vérification sauvegardée: %s", VERIFICATION_PATH)


def main() -> None:
    logger.info("Démarrage script data_collection.py")
    products = collect_products()
    logger.info("Produits bruts uniques collectés: %s", len(products))

    if not products:
        logger.error("Aucun produit collecté. Vérifie User-Agent, connexion ou réessaie plus tard.")
        return

    df = build_dataframe(products)
    df = reduce_to_target_rows_preserve_distribution(df)
    checks = check_conformity(df)
    print_summary(df, checks)
    export_dataset(df, checks)
    logger.info("Fin du script.")


if __name__ == "__main__":
    main()