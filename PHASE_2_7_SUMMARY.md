# Phase 2.7 — Synthèse des Stratégies de Gestion du Déséquilibre

## 📌 Vue d'ensemble rapide

### Contexte
- **Déséquilibre** : 82.5% (classe 0) vs 17.5% (classe 1)
- **Objectif** : Maximiser Recall sans sacrifier Precision
- **Coût asymétrique** : Faux négatif (produit mauvais non détecté) > Faux positif

---

## ✅ 3 Stratégies Implémentées (Minimum Requis)

### 1️⃣ **BASELINE** - Référence
```python
Pipeline([
    ('preprocessor', preprocessor),
    ('clf', Model(class_weight='balanced'))
])
```
| Aspect | Évaluation |
|--------|-----------|
| **Approche** | Pas de rééquilibrage, ajustement des poids |
| **Vitesse** | ⭐⭐⭐⭐⭐ Très rapide |
| **Perte de données** | ❌ Aucune |
| **Bruit ajouté** | ❌ Aucun |
| **Cas d'usage** | Référence, données très volumineuses |

---

### 2️⃣ **SMOTE** - Oversampling Intelligent
```python
ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(k_neighbors=5, random_state=42)),
    ('clf', Model())
])
```
| Aspect | Évaluation |
|--------|-----------|
| **Approche** | Synthèse de nouvelles instances minoritaires |
| **Vitesse** | ⭐⭐⭐ Modéré |
| **Perte de données** | ❌ Aucune |
| **Bruit ajouté** | ⚠️ Peut créer du bruit |
| **Cas d'usage** | **STANDARD** - Recommandé pour la plupart |
| **Avantage clé** | Augmente info sans dupliquer |

---

### 3️⃣ **RandomUnderSampler** - Undersampling Aléatoire
```python
ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampler', RandomUnderSampler(random_state=42)),
    ('clf', Model())
])
```
| Aspect | Évaluation |
|--------|-----------|
| **Approche** | Suppression aléatoire des instances majoritaires |
| **Vitesse** | ⭐⭐⭐⭐⭐ Très rapide |
| **Perte de données** | ❌❌ Importante |
| **Bruit ajouté** | ❌ Aucun |
| **Cas d'usage** | Big data, vitesse critique |
| **Avantage clé** | Très rapide pour données volumineuses |

---

## 🎯 3 Stratégies Bonus (À Implémenter)

### 4️⃣ **SMOTEENN** ⭐ Recommandé
```python
ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampler', SMOTEENN(random_state=42)),
    ('clf', Model())
])
```
| Aspect | Évaluation |
|--------|-----------|
| **Approche** | SMOTE + Edited Nearest Neighbours (nettoyage agressif) |
| **Vitesse** | ⭐⭐ Lent |
| **Amélioration vs SMOTE** | ✅ Nettoie bruit généré |
| **Cas d'usage** | **MEILLEUR BONUS** |

---

### 5️⃣ **SMOTETomek** ⭐ Alternative
```python
ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampler', SMOTETomek(random_state=42)),
    ('clf', Model())
])
```
| Aspect | Évaluation |
|--------|-----------|
| **Approche** | SMOTE + Tomek Links (nettoyage modéré) |
| **Vitesse** | ⭐⭐ Modéré-Lent |
| **Amélioration vs SMOTE** | ✅ Nettoie frontière |
| **Cas d'usage** | Alternative si SMOTEENN trop agressif |

---

### 6️⃣ **NearMiss (v1)** - Undersampling Intelligent
```python
ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampler', NearMiss(version=1, random_state=42)),
    ('clf', Model())
])
```
| Aspect | Évaluation |
|--------|-----------|
| **Approche** | Sélectionne majoritaires les plus informatiques |
| **Vitesse** | ⭐⭐⭐ Modéré |
| **Perte de données** | ⚠️ Moins grave que Random |
| **Cas d'usage** | Alternative légère undersampling |

---

## 🚀 Ordre de Test Recommandé

```
Phase 3 - Étape 1 : Test Rapide
├─ BASELINE (référence)
├─ SMOTE ★ (meilleur oversampling)
└─ RandomUnderSampler (validation rapide)

Phase 3 - Étape 2 : Refinement (Top 3 modèles)
├─ SMOTEENN ★★ (meilleur bonus oversampling)
├─ SMOTETomek (nettoyage modéré)
└─ NearMiss (undersampling intelligent)
```

---

## 📊 Performance Attendue

### Baselines (Historique du projet)
- **Baseline** : F1 ≈ 0.55-0.60
- **SMOTE** : F1 ≈ 0.65-0.70
- **Undersampler** : F1 ≈ 0.60-0.65

### Objectifs Phase 3
| Métrique | Cible |
|----------|-------|
| **Recall** | ≥ 0.75 (détecter mauvais produits) |
| **Precision** | ≥ 0.50 (fausses alertes acceptables) |
| **F1-score** | ≥ 0.60 |
| **PR-AUC** | À maximiser |

---

## 🔧 Configuration Pipeline Phase 3

```python
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

# ÉTAPE 1 : Définir stratégies
strategies = {
    # Base
    'baseline': None,
    'smote': SMOTE(random_state=42),
    'undersample': RandomUnderSampler(random_state=42),
    # Bonus
    'smoteenn': SMOTEENN(random_state=42),
    'smote_tomek': SMOTETomek(random_state=42),
    'near_miss': NearMiss(version=1, random_state=42),
}

# ÉTAPE 2 : Construire pipeline
def build_pipeline(preprocessor, model, strategy):
    if strategy is None:  # baseline
        return Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model)
        ])
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('sampler', strategy),
        ('clf', model)
    ])

# ÉTAPE 3 : Tester avec CV stratifiée
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for model_name, model in models.items():
    for strat_name, sampler in strategies.items():
        pipeline = build_pipeline(preprocessor, model, sampler)
        f1_scores = cross_val_score(pipeline, X_train, y_train, 
                                    cv=cv, scoring='f1')
        print(f"{model_name} + {strat_name}: {f1_scores.mean():.4f}")
```

---

## ⚠️ Points Critiques

### 1. **Prévention du Data Leakage**
```python
# ✓ BON : sampler APRÈS preprocessor
ImbPipeline([
    ('preprocessor', preprocessor),
    ('sampler', SMOTE()),
    ('clf', model)
])

# ✗ MAUVAIS : sampler AVANT preprocessor
ImbPipeline([
    ('sampler', SMOTE()),
    ('preprocessor', preprocessor),
    ('clf', model)
])
```

### 2. **Validation Croisée Stratifiée**
- Toujours utiliser `StratifiedKFold`
- Assure même distribution de classes dans chaque fold

### 3. **Random State Fixe**
- Tous les samplers avec `random_state=42`
- Assure reproductibilité

### 4. **Métriques Correctes**
```python
# ✓ Adaptées au déséquilibre
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)
roc_auc_score(y_true, y_pred)
average_precision_score(y_true, y_pred)  # PR-AUC

# ✗ À éviter
accuracy_score(y_true, y_pred)  # Trop optimiste sur données déséquilibrées
```

---

## 📚 Fichiers Clés du Projet

| Fichier | Rôle |
|---------|------|
| `IMBALANCE_STRATEGIES.md` | Documentation détaillée des 6 stratégies |
| `PHASE_2_7_IMBALANCE_GUIDE.md` | Guide complet avec exemples |
| `scripts/run_phase3_model_selection.py` | Script original (3 stratégies) |
| `scripts/run_phase3_model_selection_extended.py` | **Script amélioré (6 stratégies)** ⭐ |
| `notebooks/04_modeling.ipynb` | Notebook Phase 3 existant |

---

## ✨ Résumé Exécutif

### État Actuel ✅
- ✅ 3 stratégies implémentées (Baseline, SMOTE, RandomUnderSampler)
- ✅ Pipeline reproductible avec imblearn
- ✅ Validation croisée stratifiée
- ✅ Documentation complète

### À Améliorer 📌
- 📌 Ajouter 3 stratégies bonus (SMOTEENN, SMOTETomek, NearMiss)
- 📌 Exécuter tests Phase 3 avec 6 stratégies
- 📌 Comparer F1, Recall, Precision, PR-AUC
- 📌 Documenter résultats

### Fichier Recommandé pour Phase 3
👉 **Utiliser `run_phase3_model_selection_extended.py`** pour tester tous les modèles × 6 stratégies

---

**Dernière mise à jour** : 2026-05-30  
**Phase** : 2.7 — Préparation des stratégies de gestion du déséquilibre  
**Statut** : ✅ Documentation complète, implémentation prête
