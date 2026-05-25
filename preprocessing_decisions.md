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