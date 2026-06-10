# Tableau de decision - Preprocessing

Ce tableau documente les decisions appliquees par le pipeline final. Les
statistiques d'imputation sont apprises uniquement sur `train.csv`, puis
appliquees a validation, test et production.

| Variable | Action | Justification |
|---|---|---|
| `code` | Exclue du modele | Identifiant unique, non generalisable |
| `product_name` | Exclue du modele | Texte libre hors perimetre |
| `brands` | Exclue du modele | Tres forte cardinalite |
| `nutriscore_grade` | Exclue du modele | Source directe de la cible, fuite de donnees |
| `healthy_illusion` | Exclue du modele | Depend directement de la cible |
| Nutriments numeriques | Imputation mediane + `StandardScaler` | Imputation robuste, ajustee uniquement sur le train |
| `main_category`, `country` | Imputation mode + One-Hot Encoding | Variables nominales |
| `has_labels`, `image_saine` | Conservation comme variables binaires dans le pipeline | Variables 0/1 directement exploitables par le modele |
| Valeurs negatives | Suppression de la ligne | Incoherence metier |
| Nutriments superieurs a 100 g/100g | Suppression de la ligne | Incoherence metier |
| Energie superieure a 1000 kcal/100g | Suppression de la ligne | Incoherence metier |
| Graisses saturees superieures aux graisses totales | Suppression de la ligne | Incoherence logique |
| Autres outliers IQR | Conservation | Valeurs extremes valides et potentiellement predictives |
| `sugar_energy_ratio` | Creation | Rapport sucre / energie |
| `sat_fat_ratio` | Creation | Part de graisses saturees dans les graisses |
| `salt_sugar_interaction` | Creation | Interaction de deux facteurs defavorables |
| `nutrition_risk_score` | Creation | Score metier combinant nutriments favorables et defavorables |
| `bad_nutrition` | Cible | Nutri-Score D/E = 1, A/B/C = 0 |

## Verification finale

- Dataset: 15 079 lignes, sans doublon de code.
- Classe minoritaire: environ 17,45 %.
- Aucune valeur negative.
- Aucune valeur nutritionnelle impossible superieure a 100 g/100g.
- Aucune ligne avec `saturated_fat_100g > fat_100g`.
- Valeurs manquantes conservees avant le split et traitees dans le pipeline.
- Split stratifie reproductible: 60 % train, 20 % validation, 20 % test.
- Strategies comparees: class weights, SMOTE, undersampling et SMOTETomek.
