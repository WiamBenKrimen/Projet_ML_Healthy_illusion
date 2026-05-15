# Fiche de cadrage — Projet Machine Learning

## Titre du projet

**Détection des produits alimentaires à illusion saine à partir d’Open Food Facts**

---

## 1. Domaine métier

Le projet appartient au domaine de la **santé alimentaire** et de la **protection du consommateur**.

Dans le marché alimentaire, certains produits peuvent être présentés comme sains grâce à leurs labels, leurs catégories ou leur positionnement marketing : bio, protéiné, sans sucre ajouté, muesli, yaourt, smoothie, barres énergétiques, etc.

Cependant, cette image positive ne garantit pas toujours une bonne qualité nutritionnelle réelle. Certains produits peuvent contenir beaucoup de sucres, de graisses, de sel ou d’additifs.

L’objectif général du projet est donc d’identifier les produits qui présentent une mauvaise qualité nutritionnelle, puis d’analyser parmi eux ceux qui possèdent une image saine.

---

## 2. Question métier

La question métier principale est :

> **Peut-on prédire automatiquement si un produit alimentaire possède une mauvaise qualité nutritionnelle à partir de ses caractéristiques nutritionnelles et catégorielles ?**

Dans un second temps, l’analyse métier complémentaire est :

> **Parmi les produits présentés comme sains, peut-on détecter ceux qui sont en réalité de mauvaise qualité nutritionnelle ?**

---

## 3. Exploration des APIs candidates

Avant de choisir la source finale des données, nous avons exploré plusieurs APIs publiques liées au domaine de l’alimentation et de la nutrition.

L’objectif était de trouver une API capable de fournir :

- un volume suffisant de données ;
- des variables exploitables pour le Machine Learning ;
- un mélange de variables numériques et catégorielles ;
- une variable cible naturelle ou une information permettant de construire une cible claire ;
- des conditions d’accès compatibles avec une collecte de plus de 10 000 lignes.

### 3.1 APIs étudiées

| API candidate | Domaine | Données disponibles | Questions métier possibles | Avantages | Limites | Décision |
|---|---|---|---|---|---|---|
| Open Food Facts | Santé alimentaire / produits alimentaires | Produits, marques, catégories, labels, pays, valeurs nutritionnelles, additifs, Nutri-Score | 1. Peut-on prédire si un produit possède une mauvaise qualité nutritionnelle ? <br> 2. Peut-on identifier les produits présentés comme sains mais ayant une mauvaise qualité nutritionnelle ? <br> 3. Peut-on analyser le lien entre labels marketing et qualité nutritionnelle réelle ? | API publique et gratuite, grand volume de données, présence du Nutri-Score, mélange de variables numériques et catégorielles | Certaines données peuvent être manquantes car la base est collaborative | Retenue |
| USDA FoodData Central | Nutrition alimentaire | Aliments, nutriments détaillés, catégories alimentaires | 1. Peut-on classifier les aliments selon leur profil nutritionnel ? <br> 2. Peut-on détecter les aliments riches en sucre ou en graisse ? <br> 3. Peut-on comparer la qualité nutritionnelle entre catégories d’aliments ? | Données nutritionnelles détaillées et fiables | Moins adaptée aux produits commerciaux, absence de labels marketing comme “bio”, “high-protein”, “no added sugar” | Non retenue |
| Edamam Food Database API | Alimentation / nutrition | Informations nutritionnelles sur des aliments et ingrédients | 1. Peut-on classifier les aliments selon leur apport énergétique ? <br> 2. Peut-on identifier des aliments à risque nutritionnel ? <br> 3. Peut-on prédire si un aliment est adapté à certains régimes ? | Données nutritionnelles structurées | Accès API avec quotas, moins adaptée à une collecte massive de plus de 10 000 lignes | Non retenue |

### 3.2 Tests exploratoires réalisés

Pour chaque API candidate, nous avons consulté la documentation officielle afin de vérifier :

- la disponibilité gratuite de l’API ;
- la possibilité de récupérer un volume important de données ;
- la présence de variables exploitables pour un modèle de classification ;
- l’existence d’une variable cible naturelle ou construite à partir d’une règle claire ;
- les limites d’utilisation, notamment les quotas et les conditions d’accès.

### 3.3 Choix final de l’API

Nous avons retenu l’API **Open Food Facts**, car elle correspond le mieux à notre problématique métier.

Cette API fournit à la fois :

- des caractéristiques nutritionnelles numériques : sucres, matières grasses, graisses saturées, sel, fibres, protéines, énergie ;
- des caractéristiques catégorielles : catégories, pays, marques ;
- des informations liées à l’image du produit : labels et catégories ;
- une information naturelle permettant de construire la variable cible : le Nutri-Score.

---

## 4. Objectifs métiers quantifiés

L’objectif métier principal est de détecter les produits alimentaires ayant une mauvaise qualité nutritionnelle à partir de leurs caractéristiques nutritionnelles et catégorielles.

Dans ce projet, un produit est considéré comme ayant une mauvaise qualité nutritionnelle si son Nutri-Score est **D** ou **E**.

Les objectifs quantifiés du projet sont les suivants :

| Objectif métier | Traduction mesurable |
|---|---|
| Détecter les produits de mauvaise qualité nutritionnelle | Identifier correctement les produits avec `bad_nutrition = 1` |
| Réduire le risque de laisser passer un produit réellement mauvais | Obtenir un recall élevé sur la classe `bad_nutrition = 1` |
| Limiter les fausses alertes | Garder une précision acceptable sur la classe `bad_nutrition = 1` |
| Travailler sur un dataset déséquilibré | Maintenir une classe minoritaire comprise entre 5 % et 25 % |
| Préparer un dataset exploitable pour la phase de modélisation | Obtenir au moins 10 000 lignes et au moins 8 features |

Dans notre dataset final, la classe minoritaire `bad_nutrition = 1` représente **17.51 %**, ce qui respecte la condition de déséquilibre demandée.

---

## 5. Tableau de traduction métier → Machine Learning

| Élément métier | Traduction ML | Type | Rôle dans le projet |
|---|---|---|---|
| Produit alimentaire | Ligne du dataset | Observation | Élément analysé par le modèle |
| Valeurs nutritionnelles | `sugars_100g`, `fat_100g`, `saturated_fat_100g`, `salt_100g`, `fiber_100g`, `proteins_100g`, `energy_kcal_100g` | Features numériques | Entrées du modèle |
| Nombre d’additifs | `additives_count` | Feature numérique | Entrée du modèle |
| Catégorie du produit | `main_category` | Feature catégorielle | Décrit le type de produit |
| Pays associé au produit | `country` | Feature catégorielle | Donnée descriptive |
| Présence de labels | `has_labels` | Feature binaire | Indique si le produit possède au moins un label |
| Image saine du produit | `image_saine` | Feature métier binaire | Indique si le produit est associé à une image nutritionnelle positive |
| Nutri-Score | `nutriscore_grade` | Source de la cible | Sert uniquement à construire la variable cible |
| Mauvaise qualité nutritionnelle | `bad_nutrition` | Target binaire | Variable à prédire |
| Produit à illusion saine | `healthy_illusion` | Variable métier d’analyse | Produit avec `image_saine = 1` et `bad_nutrition = 1` |

La variable cible principale du modèle est :

```text
bad_nutrition
```

Règle de construction :

```text
Nutri-Score A, B ou C → bad_nutrition = 0
Nutri-Score D ou E   → bad_nutrition = 1
```

La variable `healthy_illusion` n’est pas la cible principale du modèle. Elle sert à l’analyse métier finale pour identifier les produits qui ont une image saine mais une mauvaise qualité nutritionnelle.

---

## 6. Analyse du coût asymétrique

Dans ce projet, les erreurs du modèle n’ont pas toutes le même impact.

### 6.1 Faux négatif

Un faux négatif correspond à un produit réellement mauvais nutritionnellement, mais prédit comme acceptable par le modèle.

```text
Vraie classe : bad_nutrition = 1
Prédiction   : bad_nutrition = 0
```

Cette erreur est plus coûteuse dans notre contexte, car le système laisse passer un produit problématique sans le signaler.

### 6.2 Faux positif

Un faux positif correspond à un produit nutritionnellement acceptable, mais prédit comme mauvais par le modèle.

```text
Vraie classe : bad_nutrition = 0
Prédiction   : bad_nutrition = 1
```

Cette erreur est moins grave qu’un faux négatif, car elle provoque surtout une alerte inutile.

### 6.3 Conclusion

Dans notre projet, le coût d’un faux négatif est plus élevé que celui d’un faux positif.

Il est donc préférable de privilégier un modèle capable de bien détecter les produits `bad_nutrition = 1`, même si cela peut produire quelques fausses alertes.

---

## 7. Choix des métriques d’évaluation

Comme le dataset est déséquilibré, l’accuracy seule n’est pas suffisante pour évaluer le modèle.

Nous utiliserons les métriques suivantes :

| Métrique | Rôle | Justification |
|---|---|---|
| Recall sur `bad_nutrition = 1` | Mesurer la capacité du modèle à détecter les produits réellement mauvais | Métrique prioritaire, car les faux négatifs sont les erreurs les plus coûteuses |
| Precision sur `bad_nutrition = 1` | Mesurer la fiabilité des alertes du modèle | Permet de limiter les fausses alertes |
| F1-score sur `bad_nutrition = 1` | Équilibrer recall et precision | Utile pour avoir un compromis entre détection et précision |
| PR-AUC | Évaluer le modèle sur un dataset déséquilibré | Plus adaptée que l’accuracy lorsque la classe positive est minoritaire |

Objectifs indicatifs pour la phase de modélisation :

| Métrique | Objectif visé |
|---|---|
| Recall sur `bad_nutrition = 1` | ≥ 0.75 |
| Precision sur `bad_nutrition = 1` | ≥ 0.50 |
| F1-score sur `bad_nutrition = 1` | ≥ 0.60 |
| PR-AUC | À maximiser |

Le recall est prioritaire, car l’objectif principal est de ne pas manquer les produits ayant une mauvaise qualité nutritionnelle.

---

## 8. Variable cible et prévention du data leakage

La colonne `nutriscore_grade` provient directement de l’API Open Food Facts. Elle est utilisée uniquement pour construire la variable cible `bad_nutrition`.

Nous conservons `nutriscore_grade` dans le dataset afin de garder une trace claire de la construction de la cible et de faciliter la vérification du dataset.

Cependant, cette colonne ne sera pas utilisée comme variable d’entrée lors de l’entraînement des modèles. En effet, comme elle sert directement à construire la cible, l’utiliser comme feature donnerait indirectement la réponse au modèle et fausserait l’évaluation.

Ainsi, lors de la phase d’entraînement, les variables utilisées seront uniquement les caractéristiques descriptives du produit : valeurs nutritionnelles, nombre d’additifs, catégorie principale, pays, présence de labels et variable `image_saine`.

---

## 9. Synthèse des livrables liés au cadrage

Les livrables associés à cette phase sont :

| Fichier | Rôle |
|---|---|
| `cadrage.md` | Définition du sujet, question métier, objectifs ML, coût des erreurs et métriques |
| `src/data_collection.py` | Script de collecte et de préparation du dataset |
| `data/raw/*.json` | Réponses brutes de l’API |
| `data/dataset.csv` | Dataset final exploitable |
| `data/sample.csv` | Échantillon de 100 lignes |
| `DATASET.md` | Documentation détaillée du dataset |
| `notebooks/01_discovery.ipynb` | Notebook de vérification et découverte du dataset |

---

## 10. Conclusion

Ce projet répond à un problème de classification supervisée binaire.

La cible principale est `bad_nutrition`, construite à partir du Nutri-Score fourni par l’API Open Food Facts. L’analyse métier complémentaire permet ensuite d’identifier les produits à `healthy_illusion`, c’est-à-dire les produits qui possèdent une image saine mais une mauvaise qualité nutritionnelle.

Le dataset final est prêt pour la phase suivante : l’entraînement, l’évaluation et la comparaison de modèles de classification.
