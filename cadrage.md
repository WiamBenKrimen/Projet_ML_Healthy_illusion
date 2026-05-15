# Fiche de cadrage — Projet Machine Learning

## Titre du projet

**Détection des produits alimentaires à illusion saine à partir d’Open Food Facts**

---

## 1. Domaine métier

Le projet appartient au domaine de la **santé alimentaire** et de la **protection du consommateur**.

Dans le marché alimentaire, certains produits peuvent être présentés comme sains grâce à leurs labels, catégories ou positionnement marketing : bio, protéiné, sans sucre ajouté, muesli, yaourt, smoothie, etc. Cependant, ces produits peuvent parfois avoir une mauvaise qualité nutritionnelle réelle.

L’objectif général du projet est donc d’aider à identifier les produits qui présentent un risque nutritionnel, puis d’analyser parmi eux ceux qui possèdent une image saine.

---

## 2. Question métier

**Peut-on prédire automatiquement si un produit alimentaire possède une mauvaise qualité nutritionnelle à partir de ses caractéristiques nutritionnelles et catégorielles ?**

Puis, dans un second temps :

**Parmi les produits présentés comme sains, peut-on détecter ceux qui sont en réalité de mauvaise qualité nutritionnelle ?**

---
---

## 3. Exploration des APIs candidates

Avant de choisir la source finale des données, nous avons exploré plusieurs APIs publiques liées au domaine de l’alimentation et de la nutrition.  
L’objectif était de trouver une API capable de fournir un volume suffisant de données, des variables exploitables pour le Machine Learning, ainsi qu’une variable cible naturelle pour un problème de classification supervisée.

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
- l’existence d’une variable cible naturelle ;
- les limites d’utilisation, notamment les quotas et les conditions d’accès.

Des tests simples d’endpoints ont été réalisés afin de vérifier le format des réponses et la richesse des données retournées.

### 3.3 Choix final de l’API

Nous avons retenu l’API **Open Food Facts**, car elle correspond le mieux à notre problématique métier.

Cette API fournit à la fois :

- des caractéristiques nutritionnelles numériques : sucres, matières grasses, sel, fibres, protéines, énergie ;
- des caractéristiques catégorielles : catégories, pays, marques ;
- des informations liées à l’image du produit : labels, catégories marketing ;
- une information naturelle permettant de construire la variable cible : le Nutri-Score.

Le Nutri-Score permet de créer la cible `bad_nutrition` sans inventer artificiellement la classe à prédire :

```text
bad_nutrition = 1 si nutriscore_grade ∈ {d, e}
bad_nutrition = 0 si nutriscore_grade ∈ {a, b, c}

## 4. Source des données

### API utilisée

- **Nom de l’API :** Open Food Facts
- **URL :** https://world.openfoodfacts.org
- **Type :** API REST publique et gratuite
- **Authentification :** aucune clé API nécessaire
- **Données disponibles :** nom du produit, catégories, labels, Nutri-Score, valeurs nutritionnelles, additifs, pays, marques.

### Endpoint principal utilisé

```text
GET https://world.openfoodfacts.org/cgi/search.pl
```

Exemple de paramètres :

```text
action=process
json=1
page_size=200
page={num_page}
fields=code,product_name,categories_tags,labels_tags,countries_tags,nutriscore_grade,nutriments,additives_n
```

---

## 5. Nature du problème ML

Le projet est un problème de **classification supervisée binaire**.

La variable cible principale est :

```text
bad_nutrition
```

Elle indique si le produit possède une mauvaise qualité nutritionnelle.

---

## 6. Définition de la variable cible principale

La variable cible `bad_nutrition` est créée à partir du champ `nutriscore_grade` fourni par Open Food Facts.

Règle :

```text
bad_nutrition = 1 si nutriscore_grade ∈ {d, e}
bad_nutrition = 0 si nutriscore_grade ∈ {a, b, c}
```

Interprétation :

| Nutri-Score | bad_nutrition | Interprétation |
|---|---:|---|
| A | 0 | Qualité nutritionnelle acceptable |
| B | 0 | Qualité nutritionnelle acceptable |
| C | 0 | Qualité nutritionnelle acceptable |
| D | 1 | Mauvaise qualité nutritionnelle |
| E | 1 | Mauvaise qualité nutritionnelle |

Important : `nutriscore_grade` sert uniquement à créer la cible. Il ne sera pas utilisé comme variable d’entrée du modèle afin d’éviter le data leakage.

---

## 7. Variables explicatives prévues

Le modèle apprendra à prédire `bad_nutrition` à partir de caractéristiques du produit.

| Variable | Type | Description |
|---|---|---|
| `sugars_100g` | Numérique | Quantité de sucres pour 100g |
| `fat_100g` | Numérique | Quantité de matières grasses pour 100g |
| `saturated_fat_100g` | Numérique | Quantité de graisses saturées pour 100g |
| `salt_100g` | Numérique | Quantité de sel pour 100g |
| `fiber_100g` | Numérique | Quantité de fibres pour 100g |
| `proteins_100g` | Numérique | Quantité de protéines pour 100g |
| `energy_kcal_100g` | Numérique | Énergie en kcal pour 100g |
| `additives_count` | Numérique | Nombre d’additifs renseignés |
| `has_labels` | Catégorielle binaire | Indique si le produit possède au moins un label |
| `image_saine` | Binaire (dérivée) | Indicateur métier si le produit est présenté comme sain |
| `main_category` | Catégorielle | Catégorie principale du produit |
| `country` | Catégorielle | Pays principal du produit |

---

## 8. Variable métier : `image_saine`

La variable `image_saine` n’est pas la cible principale du modèle.

Elle représente l’image perçue du produit, c’est-à-dire si le produit est présenté comme sain à travers ses catégories ou ses labels.

Elle ne veut pas dire que le produit est réellement sain.

Règle proposée :

```text
image_saine = 1 si le produit contient au moins un indicateur d’image saine
image_saine = 0 sinon
```

### Catégories associées à une image saine

```text
muesli, granola, breakfast-cereals, cereals,
yogurts, fermented-milk-products,
smoothies, fruit-juices,
protein-bars, energy-bars, cereal-bars,
sports-nutrition, diet-products, light-products
```

### Labels associés à une image saine

```text
organic, bio, no-added-sugar, low-fat, reduced-fat,
high-protein, source-of-fibre, rich-in-fibre, natural
```

Remarque importante : tous les labels Open Food Facts ne sont pas considérés comme des labels santé. Par exemple, `halal`, `kosher`, `fair-trade`, `recyclable` ou `made-in-france` ne sont pas utilisés comme indicateurs d’image saine.

---

## 9. Conclusion métier : `healthy_illusion`

La variable `healthy_illusion` est une conclusion métier obtenue après la création de `bad_nutrition` et `image_saine`.

Règle :

```text
healthy_illusion = 1 si image_saine = 1 ET bad_nutrition = 1
healthy_illusion = 0 sinon
```

Interprétation : un produit est considéré comme une illusion saine s’il est présenté comme sain mais possède une mauvaise qualité nutritionnelle.

| Produit | image_saine | bad_nutrition | healthy_illusion |
|---|---:|---:|---:|
| Yaourt bio très sucré | 1 | 1 | 1 |
| Barre protéinée Nutri-Score E | 1 | 1 | 1 |
| Soda classique Nutri-Score E | 0 | 1 | 0 |
| Muesli Nutri-Score B | 1 | 0 | 0 |

---

## 10. Objectifs métiers quantifiés

| Objectif métier | Critère de succès |
|---|---|
| Détecter les produits à mauvaise qualité nutritionnelle | Identifier au moins 75 % des produits réellement mauvais nutritionnellement |
| Limiter les fausses alertes | Garder une précision minimale de 50 % |
| Identifier les illusions saines comme analyse finale | Repérer les produits avec `image_saine = 1` et `bad_nutrition = 1` |
| Obtenir un système exploitable | Temps de prédiction inférieur à 100 ms par produit |

---

## 11. Traduction en objectifs ML

| Objectif métier | Objectif ML | Métrique principale | Seuil cible |
|---|---|---|---|
| Détecter les produits mauvais nutritionnellement | Maximiser le recall sur la classe `bad_nutrition = 1` | Recall | ≥ 0.75 |
| Réduire les fausses alertes | Maintenir une précision acceptable | Precision | ≥ 0.50 |
| Obtenir un bon compromis global | Équilibrer precision et recall | F1-score | ≥ 0.60 |
| Évaluer sur données déséquilibrées | Utiliser une métrique adaptée au déséquilibre | PR-AUC | À maximiser |

---

## 12. Analyse du coût métier asymétrique

### Faux négatif

Un faux négatif correspond à un produit réellement mauvais nutritionnellement, mais prédit comme acceptable.

Conséquences : le consommateur n’est pas alerté, le produit peut être consommé régulièrement malgré sa mauvaise qualité nutritionnelle, et le risque est plus important si le produit possède une image saine.

Le coût métier est considéré comme élevé.

### Faux positif

Un faux positif correspond à un produit acceptable nutritionnellement, mais prédit comme mauvais.

Conséquences : le consommateur peut éviter un produit pourtant acceptable. L’impact est principalement une gêne ou une perte de confiance.

Le coût métier est considéré comme plus faible.

### Conclusion

Les faux négatifs sont plus coûteux que les faux positifs. Le projet privilégiera donc le **recall** sur la classe `bad_nutrition = 1`.

---

## 13. Métriques retenues

| Métrique | Statut | Justification |
|---|---|---|
| Recall classe 1 | Principale | Priorité : ne pas rater les mauvais produits |
| Precision classe 1 | Secondaire | Limiter les fausses alertes |
| F1-score classe 1 | Secondaire | Compromis entre recall et precision |
| PR-AUC | Validation | Adaptée aux données déséquilibrées |
| Accuracy | Non principale | Peut être trompeuse si les classes sont déséquilibrées |
| ROC-AUC seule | Non principale | Peut être optimiste sur données déséquilibrées |

---

## 14. Contraintes dataset à respecter

| Critère | Exigence |
|---|---|
| Type de tâche | Classification supervisée binaire |
| Taille totale | Actuellement 15 114 lignes (data/dataset.csv) — exigence minimale : 10 000 lignes |
| Nombre de features | Au moins 8 après feature engineering |
| Classe minoritaire | Entre 5 % et 25 % |
| Types de variables | Mélange de numériques et catégorielles |

---

## 15. Pipeline général du projet

```text
1. Collecter les produits depuis Open Food Facts.
2. Sauvegarder les données brutes en JSON.
3. Extraire les variables nutritionnelles et catégorielles.
4. Créer la cible bad_nutrition à partir du Nutri-Score.
5. Supprimer nutriscore_grade des features utilisées par le modèle.
6. Créer image_saine à partir des labels et catégories.
7. Créer healthy_illusion comme conclusion métier.
8. Vérifier la distribution des classes.
9. Exporter dataset.csv et sample.csv.
10. Utiliser le dataset en Phase 2 pour entraîner et évaluer les modèles.
```

---

## 16. Positionnement final du projet

Le modèle ML ne prédit pas directement la notion subjective d’illusion saine.

Le modèle prédit :

```text
bad_nutrition
```

Ensuite, la conclusion métier est faite avec :

```text
healthy_illusion = image_saine ET bad_nutrition
```

Ce choix permet de garder une cible ML plus naturelle, basée sur le Nutri-Score fourni par l’API, tout en conservant l’objectif métier initial de détection des produits à illusion saine.
