# Tableau de décision — Preprocessing

Ce fichier documente les décisions prises pendant la Phase 2.

| Variable                             | Action                              | Justification                                                                                                            |
|:-------------------------------------|:------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| code                                 | Exclusion                           | Identifiant unique du produit, non informatif pour la prédiction.                                                        |
| product_name                         | Exclusion                           | Texte libre non exploité dans cette version du projet.                                                                   |
| brands                               | Exclusion                           | Forte cardinalité et risque de bruit.                                                                                    |
| nutriscore_grade                     | Exclusion                           | Utilisé pour construire la cible bad_nutrition, donc risque de data leakage.                                             |
| healthy_illusion                     | Exclusion                           | Variable construite à partir de bad_nutrition, donc risque de data leakage.                                              |
| Variables numériques nutritionnelles | Imputation médiane + StandardScaler | Les valeurs manquantes sont traitées sans utiliser le jeu de test. Le scaling prépare les modèles sensibles à l'échelle. |
| Variables catégorielles              | Imputation mode + One-Hot Encoding  | Les variables sont nominales et ne possèdent pas d'ordre naturel.                                                        |
| Outliers                             | Conservation                        | Les valeurs extrêmes peuvent être des signaux utiles pour détecter les produits de mauvaise qualité nutritionnelle.      |
| sugar_energy_ratio                   | Création                            | Mesure la part des sucres par rapport à l'énergie du produit.                                                            |
| sat_fat_ratio                        | Création                            | Mesure la part des graisses saturées dans les graisses totales.                                                          |
| salt_sugar_interaction               | Création                            | Capture une interaction entre deux facteurs nutritionnels défavorables.                                                  |
| nutrition_risk_score                 | Création                            | Score métier simple combinant sucre, graisses saturées, sel, énergie et fibres.                                          |
| bad_nutrition                        | Cible                               | Variable cible à prédire, construite à partir du Nutri-Score.                                                            |


---

## Génération automatisée des tableaux manquants / outliers

Un script permet de calculer et d'ajouter automatiquement :

- le tableau des taux de valeurs manquantes par variable (avec décision proposée),
- le résumé des outliers numériques (méthode IQR) avec counts et pourcentages.

Exécuter le script depuis la racine du projet :

```bash
python scripts/generate_preprocessing_summary.py
```

Le script appendra les sections calculées à la fin de ce fichier.

(Le contenu ajouté sera horodaté.)

| bad_nutrition                        | Cible                               | Variable cible à prédire, construite à partir du Nutri-Score.                                                            |
---

**Section générée automatiquement le 2026-05-26T17:55:54**


## Tableau automatique des valeurs manquantes (généré)

| Variable | Valeurs manquantes | Pourcentage (%) | Décision proposée |
|:--|--:|--:|:--:|
| code | 0 | 0.0 | Aucune action |
| product_name | 274 | 1.81 | Supprimer ligne (cas isolé) ou imputer (médiane/mode) |
| brands | 404 | 2.67 | Supprimer ligne (cas isolé) ou imputer (médiane/mode) |
| main_category | 0 | 0.0 | Aucune action |
| country | 0 | 0.0 | Aucune action |
| sugars_100g | 0 | 0.0 | Aucune action |
| fat_100g | 0 | 0.0 | Aucune action |
| saturated_fat_100g | 0 | 0.0 | Aucune action |
| salt_100g | 0 | 0.0 | Aucune action |
| fiber_100g | 0 | 0.0 | Aucune action |
| proteins_100g | 0 | 0.0 | Aucune action |
| energy_kcal_100g | 0 | 0.0 | Aucune action |
| additives_count | 0 | 0.0 | Aucune action |
| has_labels | 0 | 0.0 | Aucune action |
| image_saine | 0 | 0.0 | Aucune action |
| nutriscore_grade | 0 | 0.0 | Aucune action |
| bad_nutrition | 0 | 0.0 | Aucune action |
| healthy_illusion | 0 | 0.0 | Aucune action |

## Résumé automatique des outliers (méthode IQR)

| Variable | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers count | Outliers % |
|:--|--:|--:|--:|--:|--:|--:|--:|
| sugars_100g | 0.599449 | 9.55627 | 8.95682 | -12.8358 | 22.9915 | 690 | 4.57 |
| fat_100g | 0.5 | 4.5 | 4 | -5.5 | 10.5 | 2020 | 13.37 |
| saturated_fat_100g | 0.1 | 1.7 | 1.6 | -2.3 | 4.1 | 1491 | 9.87 |
| salt_100g | 0.01 | 0.542375 | 0.532375 | -0.788563 | 1.34094 | 794 | 5.25 |
| fiber_100g | 0.8 | 3.8 | 3 | -3.7 | 8.3 | 1394 | 9.22 |
| proteins_100g | 1.6 | 9.4 | 7.8 | -10.1 | 21.1 | 928 | 6.14 |
| energy_kcal_100g | 57 | 350 | 293 | -382.5 | 789.5 | 50 | 0.33 |
| additives_count | 0 | 1 | 1 | -1.5 | 2.5 | 1152 | 7.62 |
| has_labels | 0 | 1 | 1 | -1.5 | 2.5 | 0 | 0.0 |
| image_saine | 1 | 1 | 0 | 1 | 1 | 2490 | 16.47 |
| bad_nutrition | 0 | 0 | 0 | 0 | 0 | 2647 | 17.51 |
| healthy_illusion | 0 | 0 | 0 | 0 | 0 | 2089 | 13.82 |


