"""
Interface Streamlit — Projet ML Healthy Illusion
Détection de produits à mauvaise nutrition
Phase 4 — Déploiement
"""

import streamlit as st
import requests
import pandas as pd
import io

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthy Illusion",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personnalisé ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin: 0.5rem 0 0; }

    .result-danger {
        background: linear-gradient(135deg, #ff4444, #cc0000);
        color: white; padding: 1.5rem 2rem;
        border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: 700;
        box-shadow: 0 4px 20px rgba(255,68,68,0.4);
    }
    .result-safe {
        background: linear-gradient(135deg, #00b09b, #096b57);
        color: white; padding: 1.5rem 2rem;
        border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: 700;
        box-shadow: 0 4px 20px rgba(0,176,155,0.4);
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #0f3460;
    }
    .info-box {
        background: #e8f4fd; border-left: 4px solid #0f3460;
        padding: 1rem 1.5rem; border-radius: 8px;
        margin-bottom: 1rem; font-size: 0.9rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a6ebd) !important;
        color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.6rem 2rem !important;
        font-weight: 600 !important; width: 100%;
    }
    .stButton > button:hover { opacity: 0.9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🥗 Healthy Illusion</h1>
    <p>Détection automatique de produits alimentaires à mauvaise nutrition — Modèle XGBoost (F1 = 0.85)</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar — statut API ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Statut de l'API")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success("🟢 API connectée")
            st.caption(f"Seuil : `{data['threshold']}`")
        else:
            st.error("🔴 API non disponible")
    except:
        st.error("🔴 API non disponible")
        st.caption("Lancez : `py -m uvicorn main:app --reload`")

    st.markdown("---")
    st.markdown("### 📊 Infos modèle")
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.caption(f"**Modèle :** {info['model_type']}")
            st.caption(f"**Seuil optimal :** `{info['threshold']}`")
            m = info.get("metrics_optimal", {})
            st.caption(f"**F1 :** `{m.get('f1','N/A')}`")
            st.caption(f"**Recall :** `{m.get('recall','N/A')}`")
            st.caption(f"**ROC-AUC :** `{m.get('roc_auc','N/A')}`")
    except:
        st.caption("Infos indisponibles")

    st.markdown("---")
    st.markdown("### 📖 À propos")
    st.caption(
        "Ce modèle détecte les produits alimentaires présentant "
        "un profil nutritionnel risqué (sucres, graisses, sel élevés). "
        "Un FN (danger non détecté) coûte 10× plus qu'un FP."
    )

# ─── Onglets ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prédiction unitaire", "📂 Prédiction par lot (CSV)", "ℹ️ Informations"])

# ══════════════════════════════════════════════════════
# TAB 1 — Prédiction unitaire
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Saisissez les valeurs nutritionnelles du produit")

    st.markdown('<div class="info-box">💡 Remplissez tous les champs puis cliquez sur <strong>Analyser</strong>.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🍬 Sucres & Énergie**")
        sugars         = st.number_input("Sucres (g/100g)",          0.0, 100.0, 10.0, 0.1)
        energy         = st.number_input("Énergie (kcal/100g)",      0.0, 1000.0, 250.0, 1.0)
        sugar_energy   = st.number_input("Ratio sucres/énergie",     0.0, 5.0, round(sugars / max(energy, 1), 4), 0.001)
        high_sugar_val = 1 if sugars > 22.5 else 0
        st.caption(f"Indicateur sucres élevés : `{'OUI ⚠️' if high_sugar_val else 'NON ✅'}`")

    with col2:
        st.markdown("**🧈 Graisses & Sel**")
        fat            = st.number_input("Graisses totales (g/100g)", 0.0, 100.0, 5.0, 0.1)
        sat_fat        = st.number_input("Graisses saturées (g/100g)",0.0, 100.0, min(2.0, fat), 0.1)
        salt           = st.number_input("Sel (g/100g)",              0.0, 100.0, 0.5, 0.01)
        sat_fat_r      = round(sat_fat / max(fat, 0.001), 4)
        salt_sugar_int = round(salt * sugars, 4)
        st.caption(f"Ratio graisses saturées : `{sat_fat_r}`")
        st.caption(f"Interaction sel×sucres : `{salt_sugar_int}`")

    with col3:
        st.markdown("**🥦 Fibres, Protéines & Additifs**")
        fiber      = st.number_input("Fibres (g/100g)",      0.0, 100.0, 2.0, 0.1)
        proteins   = st.number_input("Protéines (g/100g)",   0.0, 100.0, 5.0, 0.1)
        additives  = st.number_input("Nombre d'additifs",    0, 50, 1, 1)
        risk_score = st.number_input("Score de risque nutritionnel", -10.0, 20.0, 2.0, 0.1)

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**🏷️ Catégorie & Pays**")
        main_cat = st.selectbox("Catégorie principale", [
            "Snacks", "Beverages", "Dairy", "Meat", "Cereals", "Fruits",
            "Vegetables", "Condiments", "Frozen foods", "Bakery", "Other"
        ])
        country = st.selectbox("Pays", ["France", "Morocco", "Spain", "Germany", "USA", "Other"])

    with col5:
        st.markdown("**📋 Labels & Image**")
        has_labels  = st.selectbox("Présence de labels nutritionnels", ["no", "yes"])
        image_saine = st.selectbox("Image saine (packaging)", ["no", "yes"])

    st.markdown("---")

    if st.button("🔍 Analyser ce produit", key="predict_btn"):
        payload = {
            "sugars_100g":           sugars,
            "fat_100g":              fat,
            "saturated_fat_100g":    min(sat_fat, fat),
            "salt_100g":             salt,
            "fiber_100g":            fiber,
            "proteins_100g":         proteins,
            "energy_kcal_100g":      energy,
            "additives_count":       int(additives),
            "sugar_energy_ratio":    sugar_energy,
            "sat_fat_ratio":         sat_fat_r,
            "salt_sugar_interaction": salt_sugar_int,
            "nutrition_risk_score":  risk_score,
            "main_category":         main_cat,
            "country":               country,
            "has_labels":            has_labels,
            "image_saine":           image_saine,
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    res = response.json()

                    if res["prediction"] == "mauvaise_nutrition":
                        st.markdown(f'<div class="result-danger">🔴 RISQUE ÉLEVÉ — Mauvaise nutrition détectée</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-safe">🟢 RISQUE FAIBLE — Bonne nutrition</div>', unsafe_allow_html=True)

                    st.markdown("---")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Prédiction",   res["prediction"])
                    m2.metric("Probabilité",  f"{res['probability']*100:.1f}%")
                    m3.metric("Confiance",    res["confidence"].upper())
                    m4.metric("Seuil utilisé", res["threshold"])

                else:
                    st.error(f"Erreur API {response.status_code} : {response.json().get('detail','')}")
            except Exception as e:
                st.error(f"Impossible de joindre l'API : {e}")

# ══════════════════════════════════════════════════════
# TAB 2 — Prédiction par lot
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Uploadez un fichier CSV pour analyser plusieurs produits")

    st.markdown('<div class="info-box">📋 Le CSV doit contenir les 16 colonnes du modèle. Téléchargez le template ci-dessous.</div>', unsafe_allow_html=True)

    # Template CSV
    template_cols = [
        "sugars_100g", "fat_100g", "saturated_fat_100g", "salt_100g",
        "fiber_100g", "proteins_100g", "energy_kcal_100g", "additives_count",
        "sugar_energy_ratio", "sat_fat_ratio", "salt_sugar_interaction",
        "nutrition_risk_score", "main_category", "country", "has_labels", "image_saine"
    ]
    template_data = {
        "sugars_100g": [25.0, 3.0],
        "fat_100g": [12.0, 1.5],
        "saturated_fat_100g": [5.0, 0.5],
        "salt_100g": [0.8, 0.1],
        "fiber_100g": [1.5, 3.0],
        "proteins_100g": [4.0, 8.0],
        "energy_kcal_100g": [450.0, 180.0],
        "additives_count": [3, 0],
        "sugar_energy_ratio": [0.22, 0.07],
        "sat_fat_ratio": [0.42, 0.33],
        "salt_sugar_interaction": [20.0, 0.3],
        "nutrition_risk_score": [3.5, -1.0],
        "main_category": ["Snacks", "Vegetables"],
        "country": ["France", "Morocco"],
        "has_labels": ["no", "yes"],
        "image_saine": ["no", "yes"],
    }
    template_df = pd.DataFrame(template_data)
    csv_template = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Télécharger le template CSV",
        data=csv_template,
        file_name="template_nutrition.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("📂 Choisir un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.markdown(f"**Aperçu — {len(df_preview)} produits détectés :**")
        st.dataframe(df_preview.head(5), use_container_width=True)

        if st.button("🚀 Analyser le fichier", key="batch_btn"):
            uploaded_file.seek(0)
            with st.spinner(f"Analyse de {len(df_preview)} produits..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                        timeout=30
                    )
                    if response.status_code == 200:
                        result_df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))

                        n_danger = (result_df["prediction"] == "mauvaise_nutrition").sum()
                        n_safe   = (result_df["prediction"] == "bonne_nutrition").sum()

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total analysés", len(result_df))
                        c2.metric("🔴 Mauvaise nutrition", n_danger)
                        c3.metric("🟢 Bonne nutrition",   n_safe)

                        st.dataframe(result_df[["prediction", "probability", "risk_level", "confidence"] +
                                               [c for c in result_df.columns if c not in ["prediction","probability","risk_level","confidence"]]],
                                     use_container_width=True)

                        st.download_button(
                            "⬇️ Télécharger les résultats",
                            data=response.content,
                            file_name=f"predictions_{uploaded_file.name}",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Erreur API : {response.json().get('detail','')}")
                except Exception as e:
                    st.error(f"Impossible de joindre l'API : {e}")

# ══════════════════════════════════════════════════════
# TAB 3 — Informations
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown("#### À propos du modèle")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**🎯 Objectif**
Détecter automatiquement les produits alimentaires présentant un profil nutritionnel risqué pour le consommateur.

**🤖 Modèle**
- Algorithme : XGBoost (Gradient Boosting)
- Optimisation : RandomizedSearchCV (20 itérations)
- Validation croisée : 5-fold stratifiée

**⚖️ Gestion du déséquilibre**
- Ratio classes : ~83% bonne / 17% mauvaise nutrition
- Stratégie : `scale_pos_weight = 4.71`
        """)

    with col_b:
        st.markdown("""
**📊 Performances (test set)**
| Métrique | Seuil 0.5 | Seuil optimal |
|---|---|---|
| F1-score | - | **0.8516** |
| Recall | - | **0.9546** |
| ROC-AUC | - | **0.9887** |

**💰 Matrice de coût**
- Faux Négatif (danger non détecté) : **coût × 10**
- Faux Positif (produit sain bloqué) : **coût × 1**
- Seuil optimal : **0.1** (favorise la détection)

**⚠️ Limites**
- Modèle entraîné sur Open Food Facts
- Performances dépendent de la qualité des données saisies
        """)

    st.markdown("---")
    st.markdown("**🔗 Endpoints API disponibles**")
    endpoints = pd.DataFrame({
        "Endpoint": ["GET /", "GET /health", "GET /model/info", "POST /predict", "POST /predict/batch"],
        "Description": [
            "Page d'accueil",
            "Vérification santé de l'API",
            "Métadonnées du modèle",
            "Prédiction unitaire",
            "Prédiction par lot (CSV)"
        ]
    })
    st.dataframe(endpoints, use_container_width=True, hide_index=True)