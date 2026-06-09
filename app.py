"""Streamlit user interface for Healthy Illusion."""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Healthy Illusion",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom CSS
# =============================================================================

st.markdown(
    """
<style>
    html, body, [class*="css"] {
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #102030 0%, #163b5c 55%, #1d5f8a 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 24px rgba(16, 32, 48, 0.18);
    }

    .main-header h1 {
        font-size: 2.3rem;
        font-weight: 750;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-header p {
        font-size: 1rem;
        margin: 0.6rem 0 0 0;
        opacity: 0.88;
        line-height: 1.6;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e7edf3;
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(16, 32, 48, 0.06);
    }

    .info-box {
        background: #f0f6fb;
        border-left: 5px solid #1d5f8a;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        color: #17324d;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .result-danger {
        background: linear-gradient(135deg, #9f1d2f, #c93345);
        color: white;
        padding: 1.4rem 1.7rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 750;
        box-shadow: 0 8px 22px rgba(159, 29, 47, 0.22);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .result-safe {
        background: linear-gradient(135deg, #176b55, #21916f);
        color: white;
        padding: 1.4rem 1.7rem;
        border-radius: 14px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 750;
        box-shadow: 0 8px 22px rgba(23, 107, 85, 0.22);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e7edf3;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
    }

    .metric-card .label {
        color: #60758a;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }

    .metric-card .value {
        color: #102030;
        font-size: 1.35rem;
        font-weight: 750;
    }

    .stButton > button {
        background: linear-gradient(135deg, #163b5c, #1d5f8a) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.4rem !important;
        font-weight: 650 !important;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.92 !important;
    }

    .stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100%;
    }

    div[data-testid="stRadio"] > div {
        gap: 0.8rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Helper functions
# =============================================================================

@st.cache_data(ttl=30)
def get_model_info() -> dict:
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30)
def check_api_health() -> dict:
    response = requests.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    return response.json()


def get_error_message(response: requests.Response) -> str:
    try:
        return response.json().get("detail", "Erreur API")
    except Exception:
        return response.text or "Erreur API"


def metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Session state
# =============================================================================

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "batch_result_df" not in st.session_state:
    st.session_state.batch_result_df = None

if "batch_result_bytes" not in st.session_state:
    st.session_state.batch_result_bytes = None

if "batch_filename" not in st.session_state:
    st.session_state.batch_filename = None


# =============================================================================
# Load API information
# =============================================================================

try:
    health_info = check_api_health()
    model_info = get_model_info()
    api_available = True
except Exception:
    health_info = {}
    model_info = {}
    api_available = False


categories = model_info.get("allowed_categories", {})
main_categories = categories.get(
    "main_category",
    ["snacks", "beverages", "dairies", "plant-based-foods-and-beverages"],
)
countries = categories.get("country", ["france", "morocco", "spain", "other"])


# =============================================================================
# Header
# =============================================================================

st.markdown(
    """
<div class="main-header">
    <h1>Healthy Illusion</h1>
    <p>
        Interface de détection des produits alimentaires présentant un profil nutritionnel risqué,
        même lorsqu'ils possèdent une image saine.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### Statut de l'API")

    if api_available:
        st.success("API connectée")
        st.caption(f"URL : `{API_URL}`")
        st.caption(f"Seuil métier : `{model_info.get('threshold', 'N/A')}`")
    else:
        st.error("API indisponible")
        st.caption("Commande recommandée :")
        st.code("docker compose up --build")

    st.markdown("---")
    st.markdown("### Informations modèle")

    if model_info:
        metrics = model_info.get("metrics_optimal", {})
        st.caption(f"Modèle : `{model_info.get('model_type', 'N/A')}`")
        st.caption(f"Seuil optimal : `{model_info.get('threshold', 'N/A')}`")
        st.caption(f"F1 test : `{metrics.get('f1', 'N/A')}`")
        st.caption(f"Recall test : `{metrics.get('recall', 'N/A')}`")
        st.caption(f"ROC-AUC : `{metrics.get('roc_auc', 'N/A')}`")
        st.caption(f"Date d'entraînement : `{model_info.get('trained_at', 'N/A')}`")
    else:
        st.caption("Informations indisponibles.")

    st.markdown("---")
    st.markdown("### À propos")
    st.caption(
        "Le modèle estime le risque qu'un produit appartienne aux classes de mauvaise nutrition. "
        "Le seuil de décision est ajusté selon une matrice de coût métier."
    )


# =============================================================================
# Navigation
# =============================================================================

page = st.radio(
    "Navigation",
    ["Prédiction unitaire", "Prédiction par lot", "Informations"],
    horizontal=True,
    key="active_page",
    label_visibility="collapsed",
)

st.markdown("---")


# =============================================================================
# Page 1 — Single prediction
# =============================================================================

if page == "Prédiction unitaire":
    st.markdown("### Prédiction unitaire")

    st.markdown(
        """
        <div class="info-box">
            Saisissez les valeurs nutritionnelles du produit. Les variables dérivées
            sont calculées automatiquement par l'API afin de garder la même logique
            entre l'interface, l'API et le modèle final.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("single_prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Sucres et énergie")
            sugars = st.number_input("Sucres (g/100g)", 0.0, 100.0, 10.0, 0.1)
            energy = st.number_input("Énergie (kcal/100g)", 0.0, 1000.0, 250.0, 1.0)

        with col2:
            st.markdown("#### Graisses et sel")
            fat = st.number_input("Graisses totales (g/100g)", 0.0, 100.0, 5.0, 0.1)
            saturated_fat = st.number_input("Graisses saturées (g/100g)", 0.0, 100.0, 2.0, 0.1)
            salt = st.number_input("Sel (g/100g)", 0.0, 100.0, 0.5, 0.01)

        with col3:
            st.markdown("#### Fibres, protéines et additifs")
            fiber = st.number_input("Fibres (g/100g)", 0.0, 100.0, 2.0, 0.1)
            proteins = st.number_input("Protéines (g/100g)", 0.0, 100.0, 5.0, 0.1)
            additives = st.number_input("Nombre d'additifs", 0, 100, 1, 1)

        st.markdown("#### Informations complémentaires")

        col4, col5 = st.columns(2)

        with col4:
            main_category = st.selectbox("Catégorie principale", main_categories)
            country = st.selectbox("Pays", countries)

        with col5:
            has_labels = st.selectbox(
                "Le produit possède-t-il des labels ?",
                [0, 1],
                format_func=lambda x: "Oui" if x else "Non",
            )
            image_saine = st.selectbox(
                "Le produit a-t-il une image saine ?",
                [0, 1],
                format_func=lambda x: "Oui" if x else "Non",
            )

        submitted = st.form_submit_button(
            "Analyser ce produit",
            type="primary",
            disabled=not api_available,
        )

    if submitted:
        if saturated_fat > fat:
            st.warning(
                "Les graisses saturées sont supérieures aux graisses totales. "
                "La valeur sera limitée automatiquement aux graisses totales pour l'analyse."
            )

        payload = {
            "sugars_100g": sugars,
            "fat_100g": fat,
            "saturated_fat_100g": min(saturated_fat, fat),
            "salt_100g": salt,
            "fiber_100g": fiber,
            "proteins_100g": proteins,
            "energy_kcal_100g": energy,
            "additives_count": int(additives),
            "main_category": main_category,
            "country": country,
            "has_labels": has_labels,
            "image_saine": image_saine,
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)

                if response.ok:
                    st.session_state.single_result = response.json()
                else:
                    st.session_state.single_result = None
                    st.error(get_error_message(response))

            except Exception as exc:
                st.session_state.single_result = None
                st.error(f"Impossible de joindre l'API : {exc}")

    if st.session_state.single_result:
        result = st.session_state.single_result

        if result["prediction"] == "mauvaise_nutrition":
            st.markdown(
                '<div class="result-danger">Risque élevé — mauvaise nutrition détectée</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-safe">Risque faible — profil nutritionnel acceptable</div>',
                unsafe_allow_html=True,
            )

        a, b, c = st.columns(3)
        with a:
            metric_card("Probabilité de risque", f"{result['probability']:.1%}")
        with b:
            metric_card("Confiance", str(result["confidence"]).upper())
        with c:
            metric_card("Seuil appliqué", str(result["threshold"]))


# =============================================================================
# Page 2 — Batch prediction
# =============================================================================

elif page == "Prédiction par lot":
    st.markdown("### Prédiction par lot")

    st.markdown(
        """
        <div class="info-box">
            Importez un fichier CSV contenant les colonnes attendues par l'API.
            Le résultat sera affiché dans cette page et restera visible après l'analyse.
        </div>
        """,
        unsafe_allow_html=True,
    )

    template = pd.DataFrame(
        [
            {
                "sugars_100g": 25.0,
                "fat_100g": 12.0,
                "saturated_fat_100g": 5.0,
                "salt_100g": 0.8,
                "fiber_100g": 1.5,
                "proteins_100g": 4.0,
                "energy_kcal_100g": 450.0,
                "additives_count": 3,
                "main_category": main_categories[0],
                "country": countries[0],
                "has_labels": 0,
                "image_saine": 1,
            }
        ]
    )

    st.download_button(
        "Télécharger le template CSV",
        template.to_csv(index=False).encode("utf-8"),
        "template_nutrition.csv",
        "text/csv",
    )

    uploaded = st.file_uploader(
        "Choisir un fichier CSV",
        type="csv",
        key="batch_uploader",
    )

    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()

        try:
            preview = pd.read_csv(io.BytesIO(uploaded_bytes))
            st.markdown(f"#### Aperçu du fichier : {len(preview)} ligne(s)")
            st.dataframe(preview.head(10), use_container_width=True)

            missing_columns = [col for col in template.columns if col not in preview.columns]
            if missing_columns:
                st.warning(
                    "Colonnes manquantes dans le CSV : "
                    + ", ".join(missing_columns)
                )

            analyze_batch = st.button(
                "Analyser le fichier",
                type="primary",
                disabled=not api_available or bool(missing_columns),
            )

            if analyze_batch:
                with st.spinner(f"Analyse de {len(preview)} produit(s)..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            files={
                                "file": (
                                    uploaded.name,
                                    uploaded_bytes,
                                    "text/csv",
                                )
                            },
                            timeout=60,
                        )

                        if response.ok:
                            result_bytes = response.content
                            result_df = pd.read_csv(
                                io.StringIO(result_bytes.decode("utf-8"))
                            )

                            st.session_state.batch_result_df = result_df
                            st.session_state.batch_result_bytes = result_bytes
                            st.session_state.batch_filename = uploaded.name
                        else:
                            st.session_state.batch_result_df = None
                            st.session_state.batch_result_bytes = None
                            st.error(get_error_message(response))

                    except Exception as exc:
                        st.session_state.batch_result_df = None
                        st.session_state.batch_result_bytes = None
                        st.error(f"Impossible de joindre l'API : {exc}")

        except Exception as exc:
            st.error(f"Impossible de lire le fichier CSV : {exc}")

    if st.session_state.batch_result_df is not None:
        results = st.session_state.batch_result_df

        st.markdown("### Résultats de l'analyse")

        n_total = len(results)
        n_risk = int((results["prediction"] == "mauvaise_nutrition").sum())
        n_safe = int(n_total - n_risk)

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total analysé", str(n_total))
        with c2:
            metric_card("Mauvaise nutrition", str(n_risk))
        with c3:
            metric_card("Profil acceptable", str(n_safe))

        st.dataframe(results, use_container_width=True)

        filename = st.session_state.batch_filename or "resultats.csv"

        st.download_button(
            "Télécharger les résultats",
            data=st.session_state.batch_result_bytes,
            file_name=f"predictions_{filename}",
            mime="text/csv",
        )


# =============================================================================
# Page 3 — Information
# =============================================================================

else:
    st.markdown("### Informations sur le modèle")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="section-card">
                <h4>Objectif métier</h4>
                <p>
                    Le projet vise à détecter les produits alimentaires qui présentent
                    un profil nutritionnel risqué, notamment lorsque leur image marketing
                    peut donner une impression plus saine que leur composition réelle.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h4>Modèle utilisé</h4>
                <p>
                    Le modèle final est basé sur XGBoost. Il utilise les variables
                    nutritionnelles, les informations de catégorie, le pays, les labels
                    et l'indicateur d'image saine.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="section-card">
                <h4>Décision métier</h4>
                <p>
                    Le seuil de décision est optimisé pour limiter les faux négatifs,
                    car ne pas détecter un produit réellement risqué est considéré comme
                    plus coûteux qu'une alerte excessive.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-card">
                <h4>Limites</h4>
                <p>
                    Les performances dépendent de la qualité des données Open Food Facts
                    et des valeurs saisies par l'utilisateur. Le modèle ne remplace pas
                    une expertise nutritionnelle professionnelle.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if model_info:
        st.markdown("### Métadonnées API")

        display_info = {
            "model_type": model_info.get("model_type"),
            "threshold": model_info.get("threshold"),
            "metrics_optimal": model_info.get("metrics_optimal"),
            "cost_config": model_info.get("cost_config"),
            "trained_at": model_info.get("trained_at"),
        }

        st.json(display_info)

    st.markdown("### Endpoints disponibles")

    endpoints = pd.DataFrame(
        {
            "Endpoint": [
                "GET /",
                "GET /health",
                "GET /model/info",
                "POST /predict",
                "POST /predict/batch",
            ],
            "Description": [
                "Page d'accueil de l'API",
                "Vérification de l'état de l'API et du modèle",
                "Métadonnées du modèle final",
                "Prédiction unitaire",
                "Prédiction par lot à partir d'un fichier CSV",
            ],
        }
    )

    st.dataframe(endpoints, use_container_width=True, hide_index=True)