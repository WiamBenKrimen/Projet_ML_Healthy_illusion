# Healthy Illusion

Healthy Illusion est un projet de Machine Learning qui detecte les produits
alimentaires presentant une mauvaise qualite nutritionnelle, definie par un
Nutri-Score D ou E. Le modele final XGBoost est expose par une API FastAPI et
utilisable par une interface Streamlit.

## Captures d'ecran

### Prediction unitaire

![Prediction unitaire](figures/Pr%C3%A9diction%20Unitaire.png)


### Prediction par lot ou informations du modele

![Prediction par lot](figures/Pr%C3%A9diction%20par%20Lot.png)



## Performances du modele

| Element | Valeur |
|---|---:|
| Dataset final | 15 079 produits |
| Classe minoritaire | 17,45 % |
| Meilleure combinaison | XGBoost + stratégie baseline avec `scale_pos_weight` |
| F1 moyen en validation croisee apres tuning | 0,8954 |
| Seuil metier optimal | 0,10 |
| Precision test au seuil optimal | 0,7835 |
| Recall test au seuil optimal | 0,9563 |
| F1 test au seuil optimal | 0,8613 |
| ROC-AUC test | 0,9880 |
| PR-AUC test | 0,9573 |

Le seuil de decision est optimise sur le jeu de validation. La matrice de cout
considere qu'un faux negatif coute dix fois plus qu'un faux positif. Sur le jeu
de test, le seuil optimal reduit le cout metier estime de `599` a `369`.

## Installation avec Docker

### Prerequis

- Docker Desktop ou Docker Engine
- Docker Compose

### Demarrer l'application complete

Depuis la racine du projet:

```bash
docker compose up
```

Une fois les services demarres:

- Interface Streamlit: <http://localhost:8501>
- Documentation Swagger: <http://localhost:8000/docs>
- Verification de l'API: <http://localhost:8000/health>
- Informations du modele: <http://localhost:8000/model/info>

### Arreter l'application

```bash
docker compose down
```

## Installation sans Docker

Python 3.11 est recommande.

```bash
python -m venv .venv
```

Activation sous Windows:

```bash
.venv\Scripts\activate
```

Installation des dependances:

```bash
pip install -r requirements.txt
```

Demarrage de l'API:

```bash
uvicorn main:app --reload
```

Dans un second terminal, demarrage de l'interface:

```bash
streamlit run app.py
```

## Exemple d'utilisation de l'API

Exemple de prediction unitaire avec `curl.exe` sous PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"sugars_100g\":25,\"fat_100g\":12,\"saturated_fat_100g\":5,\"salt_100g\":0.8,\"fiber_100g\":1.5,\"proteins_100g\":4,\"energy_kcal_100g\":450,\"additives_count\":3,\"main_category\":\"snacks\",\"country\":\"france\",\"has_labels\":0,\"image_saine\":1}"
```

Exemple de reponse:

```json
{
  "prediction": "mauvaise_nutrition",
  "probability": 0.78,
  "threshold": 0.10,
  "confidence": "high",
  "risk_level": "RISQUE ELEVE"
}
```

L'API calcule automatiquement les quatre variables derivees utilisees par le
modele. L'utilisateur saisit uniquement les informations brutes du produit.

## Parcours utilisateur

1. Ouvrir l'interface Streamlit sur <http://localhost:8501>.
2. Choisir l'onglet `Prediction unitaire`.
3. Saisir les valeurs nutritionnelles, la categorie, le pays et les labels.
4. Cliquer sur `Analyser ce produit`.
5. Consulter le niveau de risque, la probabilite et le seuil applique.
6. Pour plusieurs produits, utiliser l'onglet `Prediction par lot`, telecharger
   le modele CSV, remplir les lignes puis envoyer le fichier.

## Architecture du depot

```text
.
|-- app.py                         # Interface utilisateur Streamlit
|-- main.py                        # API REST FastAPI
|-- src/
|   |-- __init__.py
|   |-- data_collection.py         # Collecte et nettoyage Open Food Facts
|   `-- features.py                # Features partagees entre train et inference
|-- tests/
|   |-- conftest.py
|   |-- test_api.py                # Tests des endpoints FastAPI
|   `-- test_features.py           # Tests du feature engineering
|-- notebooks/
|   |-- 01_discovery.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_preprocessing.ipynb
|   |-- 04_modeling.ipynb
|   |-- 05_tuning.ipynb
|   `-- 06_evaluation.ipynb
|-- data/
|   |-- raw/                       # Reponses brutes Open Food Facts
|   |-- processed/                 # Splits train, validation et test
|   `-- dataset.csv                # Dataset nettoye avant preprocessing
|-- models/
|   |-- preprocessor.joblib
|   |-- tuned_model.joblib
|   `-- final_model.joblib         # Pipeline final, seuil et metadonnees
|-- figures/                       # Figures EDA et captures d'ecran de l'interface
|-- docker-compose.yml
|-- requirements.txt               # Dependances du projet
|-- DATASET.md
|-- preprocessing_decisions.md
```

## Reproduire l'analyse Machine Learning

Pour reproduire l'analyse, executer les notebooks dans leur ordre numerique,
de `01_discovery.ipynb` a `06_evaluation.ipynb`. Ils documentent la collecte,
l'analyse exploratoire, le preprocessing, la comparaison des modeles, le
tuning et l'evaluation finale.

## Tests

Le dossier `tests/` est conserve car la liste des livrables finaux demande un
depot contenant du code modulaire avec `src/`, `app/` et `tests/`.

Executer les tests:

```bash
pytest -q
```

Verifier la configuration Docker:

```bash
docker compose config
```

## Limites connues

- Open Food Facts est une base collaborative et peut contenir des erreurs.
- Le modele predit le Nutri-Score D/E et ne constitue pas un diagnostic medical.
- Les performances dependent de la qualite des valeurs nutritionnelles saisies.
- Les categories rares ou absentes des donnees d'entrainement sont moins fiables.
- Le modele doit etre reentraine lorsque la distribution des donnees evolue.
