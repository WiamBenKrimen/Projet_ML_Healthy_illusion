"""Streamlit user interface for Healthy Illusion."""

from __future__ import annotations

import io
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Healthy Illusion", page_icon="🥗", layout="wide")
st.title("🥗 Healthy Illusion")
st.caption(
    "Detecter les produits dont le profil nutritionnel est risque, "
    "meme lorsqu'ils possedent une image saine."
)


@st.cache_data(ttl=30)
def get_model_info():
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    response.raise_for_status()
    return response.json()


try:
    model_info = get_model_info()
    api_available = True
except Exception:
    model_info = {}
    api_available = False

categories = model_info.get("allowed_categories", {})
main_categories = categories.get(
    "main_category", ["snacks", "beverages", "dairies", "plant-based-foods-and-beverages"]
)
countries = categories.get("country", ["france", "morocco", "spain", "other"])

with st.sidebar:
    st.subheader("Statut")
    if api_available:
        st.success("API connectee")
        st.metric("Seuil metier", model_info["threshold"])
        metrics = model_info.get("metrics_optimal", {})
        st.metric("F1 test", metrics.get("f1", "N/A"))
        st.metric("Recall test", metrics.get("recall", "N/A"))
        st.caption(f"Modele entraine le {model_info.get('trained_at', 'N/A')}")
    else:
        st.error("API indisponible")
        st.code("docker compose up --build")

tab_single, tab_batch, tab_info = st.tabs(
    ["Prediction unitaire", "Prediction par lot", "Informations"]
)

with tab_single:
    st.subheader("Informations nutritionnelles du produit")
    col1, col2, col3 = st.columns(3)
    with col1:
        sugars = st.number_input("Sucres (g/100g)", 0.0, 100.0, 10.0, 0.1)
        fat = st.number_input("Graisses totales (g/100g)", 0.0, 100.0, 5.0, 0.1)
        saturated_fat = st.number_input(
            "Graisses saturees (g/100g)", 0.0, float(fat), min(2.0, fat), 0.1
        )
    with col2:
        salt = st.number_input("Sel (g/100g)", 0.0, 100.0, 0.5, 0.01)
        fiber = st.number_input("Fibres (g/100g)", 0.0, 100.0, 2.0, 0.1)
        proteins = st.number_input("Proteines (g/100g)", 0.0, 100.0, 5.0, 0.1)
    with col3:
        energy = st.number_input("Energie (kcal/100g)", 0.0, 1000.0, 250.0, 1.0)
        additives = st.number_input("Nombre d'additifs", 0, 100, 1, 1)
        main_category = st.selectbox("Categorie principale", main_categories)
        country = st.selectbox("Pays", countries)

    col4, col5 = st.columns(2)
    with col4:
        has_labels = st.selectbox(
            "Le produit possede-t-il des labels ?", [0, 1], format_func=lambda x: "Oui" if x else "Non"
        )
    with col5:
        image_saine = st.selectbox(
            "Le produit a-t-il une image saine ?", [0, 1], format_func=lambda x: "Oui" if x else "Non"
        )

    if st.button("Analyser ce produit", type="primary", disabled=not api_available):
        payload = {
            "sugars_100g": sugars,
            "fat_100g": fat,
            "saturated_fat_100g": saturated_fat,
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
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        if response.ok:
            result = response.json()
            if result["prediction"] == "mauvaise_nutrition":
                st.error("RISQUE ELEVE - mauvaise nutrition detectee")
            else:
                st.success("RISQUE FAIBLE - profil nutritionnel acceptable")
            a, b, c = st.columns(3)
            a.metric("Probabilite de risque", f"{result['probability']:.1%}")
            b.metric("Confiance", result["confidence"])
            c.metric("Seuil applique", result["threshold"])
        else:
            st.error(response.json().get("detail", "Erreur API"))

with tab_batch:
    st.subheader("Analyser un fichier CSV")
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
        "Telecharger le template CSV",
        template.to_csv(index=False).encode("utf-8"),
        "template_nutrition.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Choisir un CSV", type="csv")
    if uploaded:
        preview = pd.read_csv(uploaded)
        st.dataframe(preview.head(), use_container_width=True)
        if st.button("Analyser le fichier", disabled=not api_available):
            uploaded.seek(0)
            response = requests.post(
                f"{API_URL}/predict/batch",
                files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                timeout=60,
            )
            if response.ok:
                results = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Telecharger les resultats",
                    response.content,
                    f"predictions_{uploaded.name}",
                    "text/csv",
                )
            else:
                st.error(response.json().get("detail", "Erreur API"))

with tab_info:
    st.subheader("A propos")
    st.write(
        "Le modele XGBoost predit si un produit appartient aux Nutri-Scores D ou E. "
        "Les variables derivees sont calculees automatiquement par l'API."
    )
    if model_info:
        st.json(
            {
                "performance": model_info.get("metrics_optimal"),
                "cout_metier": model_info.get("cost_config"),
                "limites": [
                    "Qualite variable des donnees collaboratives Open Food Facts.",
                    "Le modele ne remplace pas l'avis d'un professionnel de sante.",
                    "La prediction depend de valeurs nutritionnelles correctement saisies.",
                ],
            }
        )
