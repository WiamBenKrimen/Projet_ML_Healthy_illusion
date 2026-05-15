# DATASET.md - Documentation du dataset

## 1. Identification

| Element | Valeur |
|---|---|
| Nom du dataset | Healthy Illusion Food Dataset |
| Version | 1.1 |
| Auteurs | HAJJOU Hajar - ERRABOUN Nouha - BENKRIMEN Wiam  |
| Date de collecte | Mai 2026 |
| Source | Open Food Facts |
| Type de tache | Classification supervisee binaire |
| Variable cible principale | bad_nutrition |

---

## 2. Source des donnees

Les donnees sont collectees depuis l'API publique Open Food Facts.

- Site principal : https://world.openfoodfacts.org
- API : https://world.openfoodfacts.org/cgi/search.pl
- Authentification : aucune cle API necessaire
- Format de reponse : JSON
- Licence : Open Database License (ODbL)
- Date d'acces : Mai 2026

### Endpoint utilise

```
GET https://world.openfoodfacts.org/cgi/search.pl
```

### Parametres principaux

```
action=process
json=1
page_size=100
page={page}
fields=code,product_name,brands,categories_tags,labels_tags,
       countries_tags,nutriscore_grade,nutrition_grades,
       nutriments,additives_n,additives_tags
sort_by=unique_scans_n
tagtype_0=categories
tag_contains_0=contains
tag_0={categorie}
```

### Categories interrogees

La collecte a volontairement melange deux familles de categories pour capturer
le phenomene de "healthy illusion" :

Categories a forte image saine :

```
breakfast-cereals, muesli, granola, yogurts,
fermented-milk-products, fruit-juices, smoothies,
protein-bars, energy-bars, cereal-bars, sports-nutrition,
plant-based-foods, organic-foods, plant-based-beverages,
fruit-based-beverages
```

Categories de contraste et de diversification :

```
milk, cheeses, breads, biscuits, chocolates, sodas, beverages,
salty-snacks, ready-meals, frozen-foods, canned-foods, condiments,
breakfasts, desserts, snacks, groceries, vegetables, fruits
```

Ce choix permet de reunir des produits qui semblent sains et d'autres qui ne
portent pas cette image, afin de construire une cible binaire desequilibree.

---

## 3. Objectif du dataset

Ce dataset est construit pour entrainer un modele de classification supervisee binaire
capable de predire si un produit alimentaire possede une mauvaise qualite nutritionnelle,
a partir de ses caracteristiques nutritionnelles, categorielle et derivees.

### Question metier principale

Peut-on predire automatiquement si un produit alimentaire possede une mauvaise qualite
nutritionnelle a partir de ses caracteristiques nutritionnelles, categorielle et
de quelques variables derivees de l'emballage et du positionnement produit ?

### Variable cible

```
bad_nutrition = 1  si Nutri-Score appartient a {D, E}  -> mauvaise qualite nutritionnelle
bad_nutrition = 0  si Nutri-Score appartient a {A, B, C} -> qualite acceptable
```

### Analyse metier complementaire

```
healthy_illusion = 1  si image_saine = 1 ET bad_nutrition = 1
healthy_illusion = 0  sinon
```

---

## 4. Description generale du dataset

| Element | Valeur |
|---|---|
| Nombre de lignes | 15 114 |
| Nombre de colonnes | 18 |
| Nombre de features ML utilisables | 12 |
| Format final | CSV (UTF-8) |
| Chemin du dataset complet | data/dataset.csv |
| Chemin de l'echantillon | data/sample.csv (100 lignes) |

---

## 5. Schema detaille des variables

| Variable | Type | Role | Description | Valeurs / Unite |
|---|---|---|---|---|
| code | Categorielle | Identifiant | Code-barres EAN du produit | Texte |
| product_name | Categorielle | Information | Nom commercial du produit | Texte libre |
| brands | Categorielle | Information | Marque du produit | Texte libre |
| main_category | Categorielle | Feature | Categorie principale du produit | Texte normalise |
| country | Categorielle | Feature | Pays principal du produit | Texte normalise |
| sugars_100g | Numerique | Feature | Quantite de sucres pour 100g | g/100g |
| fat_100g | Numerique | Feature | Matieres grasses totales pour 100g | g/100g |
| saturated_fat_100g | Numerique | Feature | Graisses saturees pour 100g | g/100g |
| salt_100g | Numerique | Feature | Quantite de sel pour 100g | g/100g |
| fiber_100g | Numerique | Feature | Quantite de fibres pour 100g | g/100g |
| proteins_100g | Numerique | Feature | Quantite de proteines pour 100g | g/100g |
| energy_kcal_100g | Numerique | Feature | Energie du produit pour 100g | kcal/100g |
| additives_count | Numerique | Feature | Nombre d'additifs alimentaires | Entier >= 0 |
| has_labels | Binaire | Feature | Presence d'au moins un label | 0 ou 1 |
| image_saine | Binaire | Feature derivee | Produit percu comme sain | 0 ou 1 |
| nutriscore_grade | Categorielle | Source cible uniquement | Nutri-Score brut de l'API | a, b, c, d, e |
| bad_nutrition | Binaire | Cible ML | Mauvaise qualite nutritionnelle | 0 ou 1 |
| healthy_illusion | Binaire | Conclusion metier | Image saine + mauvaise nutrition | 0 ou 1 |

---

## 6. Variable cible principale : bad_nutrition

### Definition

| Nutri-Score | bad_nutrition | Interpretation |
|---|---:|---|
| A | 0 | Bonne qualite nutritionnelle |
| B | 0 | Qualite acceptable |
| C | 0 | Qualite moyenne mais acceptable |
| D | 1 | Mauvaise qualite nutritionnelle |
| E | 1 | Tres mauvaise qualite nutritionnelle |

### Justification

Le Nutri-Score est une information officielle calculee par Open Food Facts a partir
des valeurs nutritionnelles reelles. La variable bad_nutrition en est une derivee
binaire directe, ce qui garantit une cible fiable et naturellement presente dans
les donnees.

### ATTENTION - Data leakage

La colonne nutriscore_grade NE DOIT PAS etre utilisee comme feature d'entree du modele
car elle sert a construire la variable cible. L'inclure provoquerait une fuite de
donnees (data leakage) et invaliderait l'evaluation du modele.

---

## 7. Variable metier : image_saine

### Definition

image_saine indique si un produit est presente ou percu comme sain a travers ses
categories ou ses labels marketing. Elle ne represente PAS la qualite nutritionnelle reelle.

### Regle appliquee

```
image_saine = 1  si categories_tags OU labels_tags contient au moins
                    un indicateur d'image saine
image_saine = 0  sinon
```

### Indicateurs dans categories_tags

```
muesli, granola, breakfast-cereals, cereals,
yogurts, fermented-milk-products,
smoothies, fruit-juices,
protein-bars, energy-bars, cereal-bars,
sports-nutrition, diet-products, light-products,
plant-based-foods, organic-foods
```

### Indicateurs dans labels_tags

```
organic, bio, no-added-sugar, low-fat, reduced-fat,
high-protein, source-of-fibre, rich-in-fibre, natural
```

### Labels exclus

Les labels suivants ne sont pas considers comme indicateurs d'image saine :

```
halal, kosher, fair-trade, recyclable, made-in-france
```

---

## 8. Conclusion metier : healthy_illusion

### Definition

```
healthy_illusion = 1  si image_saine = 1 ET bad_nutrition = 1
healthy_illusion = 0  sinon
```

### Exemples

| Produit | image_saine | bad_nutrition | healthy_illusion |
|---|---:|---:|---:|
| Yaourt bio tres sucre (Nutri-Score D) | 1 | 1 | 1 |
| Barre proteines Nutri-Score E | 1 | 1 | 1 |
| Soda classique Nutri-Score E | 0 | 1 | 0 |
| Muesli Nutri-Score B | 1 | 0 | 0 |
| Chips Nutri-Score D sans label | 0 | 1 | 0 |

---

## 9. Features utilisees en modelisation

### Variables d'entree du modele (12 features)

Variables numeriques (7) :
```
sugars_100g, fat_100g, saturated_fat_100g, salt_100g,
fiber_100g, proteins_100g, energy_kcal_100g
```

Variable numerique derivee (1) :
```
additives_count
```

Variables categorielle (2) :
```
main_category, country
```

Variables binaires (2) :
```
has_labels, image_saine
```

### Variables a NE PAS utiliser comme features

| Variable | Raison d'exclusion |
|---|---|
| nutriscore_grade | Source directe de la cible -> data leakage |
| bad_nutrition | C'est la cible elle-meme |
| healthy_illusion | Conclusion metier construite apres prediction |
| code | Identifiant technique sans valeur predictive |
| product_name | Texte libre non structure |
| brands | Texte libre non structure |

---

## 10. Nettoyage des donnees applique

1. Suppression des produits sans nutriscore_grade valide (hors {a,b,c,d,e}).
2. Deduplication sur la base du code-barres (code).
3. Suppression des lignes avec plus de 2 valeurs nutritionnelles manquantes sur 8.
4. Imputation des valeurs manquantes restantes par la mediane de la colonne.
5. Extraction de main_category depuis categories_tags (premier tag normalise).
6. Extraction de country depuis countries_tags (premier tag normalise).
7. Creation des variables derivees : bad_nutrition, has_labels, image_saine, healthy_illusion.
8. Melange aleatoire des lignes avec random_state=42.

---

## 11. Distribution des classes

### Contrainte imposee par l'enonce

| Classe bad_nutrition | Exigence |
|---|---|
| Classe minoritaire (1) | Entre 5% et 25% du total |
| Classe majoritaire (0) | Entre 75% et 95% du total |

### Distribution mesurée

| Classe bad_nutrition | Interpretation | Nombre | Pourcentage |
|---:|---|---:|---:|
| 0 | Qualite nutritionnelle acceptable | 12 467 | 82.49% |
| 1 | Mauvaise qualite nutritionnelle | 2 647 | 17.51% |
| Total | | 15 114 | 100% |

Statistiques complémentaires (calculées à partir de `data/dataset.csv`):

- `image_saine = 1` : 12 624 produits
- `healthy_illusion = 1` : 2 089 produits

Ces valeurs proviennent du dernier run de `src/data_collection.py` (log du
15/05/2026). Recalculer la distribution si la collecte est relancée.

### Niveau de déséquilibre (rappel)

La contrainte métier est que la classe minoritaire (`bad_nutrition = 1`) se
situe idéalement entre 5 % et 25 % du total. Utiliser les métriques adaptées
(Recall, F1-score, PR-AUC) sur des jeux déséquilibrés.

---

## 12. Fichiers produits

| Fichier | Description |
|---|---|
| data/raw/*.json | Reponses brutes sauvegardees depuis l'API |
| data/dataset.csv | Dataset final nettoye |
| data/sample.csv | Echantillon de 100 lignes |
| src/data_collection.py | Script de collecte et preparation |

---

## 13. Limites connues du dataset

- Open Food Facts est une base collaborative : certaines informations peuvent etre
  incompletes ou incorrectes.
- Tous les produits ne possedent pas de Nutri-Score ; ceux sans Nutri-Score sont exclus.
- image_saine est une variable construite a partir d'une regle documentee,
  pas d'une annotation officielle.
- La couverture geographique est variable selon les pays.
- La distribution des classes depend des categories collectees.

---

## 14. Reproductibilite

Le script src/data_collection.py garantit la reproductibilite via :

- random_state=42 pour tous les echantillonnages aleatoires.
- Sauvegarde des donnees brutes JSON pour eviter de re-requeter l'API.
- Logging horodate de chaque appel API.
- Verification automatique de la conformite en fin d'execution.

Pour relancer la collecte :
```
python src/data_collection.py
```

---

## 15. Utilisation prevue en Phase 2

1. Entrainer plusieurs modeles de classification (Logistic Regression, Random Forest, XGBoost).
2. Comparer les modeles : Recall, Precision, F1-score, PR-AUC.
3. Appliquer des techniques de gestion du desequilibre si necessaire (SMOTE, class_weight).
4. Analyser les erreurs du modele (matrice de confusion).
5. Identifier les produits a illusion saine : image_saine=1 ET predicted_bad_nutrition=1.
6. Interpreter les features importantes (feature importance, SHAP values).
