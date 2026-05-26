from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib

root = Path(__file__).resolve().parents[1]
train = pd.read_csv(root / 'data' / 'processed' / 'train.csv')
X_train = train.drop(columns=['bad_nutrition'])
y_train = train['bad_nutrition']
preprocessor = joblib.load(root / 'models' / 'preprocessor.joblib')

models = {
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=5000, class_weight='balanced', solver='liblinear'),
    'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=200),
    'MLP': MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
}

strategies = {
    'baseline': None,
    'smote': SMOTE(random_state=42),
    'undersample': RandomUnderSampler(random_state=42)
}

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    for strat_name, sampler in strategies.items():
        pt = clone(preprocessor)
        if strat_name == 'baseline':
            pipeline = Pipeline([('preprocessor', pt), ('clf', clone(model))])
        else:
            pipeline = ImbPipeline([('preprocessor', pt), ('sampler', sampler), ('clf', clone(model))])
        scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1', n_jobs=1)
        results.append({
            'model': model_name,
            'strategy': strat_name,
            'mean_f1': float(scores.mean()),
            'std_f1': float(scores.std())
        })

res_df = pd.DataFrame(results).sort_values(['mean_f1', 'std_f1'], ascending=[False, True]).reset_index(drop=True)
res_df.to_csv(root / 'models' / 'model_selection_results.csv', index=False)
print(res_df.to_string(index=False))
print(f"Saved model selection results to {root / 'models' / 'model_selection_results.csv'}")
