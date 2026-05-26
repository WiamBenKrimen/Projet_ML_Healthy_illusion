from pathlib import Path
import argparse
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def build_pipeline(preprocessor, model, strategy):
    if strategy == 'baseline':
        return Pipeline([('preprocessor', clone(preprocessor)), ('clf', model)])
    sampler = SMOTE(random_state=42) if strategy == 'smote' else RandomUnderSampler(random_state=42)
    return ImbPipeline([('preprocessor', clone(preprocessor)), ('sampler', sampler), ('clf', model)])


def get_model_and_params(model_name):
    if model_name == 'LogisticRegression':
        model = LogisticRegression(random_state=42, solver='liblinear', max_iter=5000, class_weight='balanced')
        params = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1', 'l2']
        }
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        params = {
            'clf__max_depth': [3, 5, 7, 10],
            'clf__min_samples_leaf': [1, 5, 10, 20]
        }
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', n_estimators=100)
        params = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20],
            'clf__min_samples_leaf': [1, 5, 10],
            'clf__max_features': ['sqrt']
        }
    else:
        model = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
        params = {
            'clf__hidden_layer_sizes': [(128, 64), (128, 64, 32), (64, 64)],
            'clf__activation': ['relu', 'tanh'],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__learning_rate_init': [0.0001, 0.001, 0.01],
            'clf__batch_size': [32, 64, 128]
        }
    return model, params


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data' / 'processed'
    model_dir = root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / 'train.csv')
    X_train = train_df.drop(columns=['bad_nutrition'])
    y_train = train_df['bad_nutrition']

    model_selection = pd.read_csv(model_dir / 'model_selection_results.csv')
    best_config = model_selection.sort_values(by=['mean_f1', 'std_f1'], ascending=[False, True]).iloc[0]
    best_model_name = best_config['model']
    best_strategy = best_config['strategy']

    print('Best model from selection:', best_model_name)
    print('Best strategy from selection:', best_strategy)
    print('Mean F1 (CV):', best_config['mean_f1'])

    preprocessor = joblib.load(model_dir / 'preprocessor.joblib')
    base_model, param_distributions = get_model_and_params(best_model_name)
    pipeline = build_pipeline(preprocessor, base_model, best_strategy)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    print('Best CV score:', best_score)
    print('Best parameters:', best_params)

    joblib.dump({'pipeline': best_model, 'best_params': best_params, 'best_score': best_score, 'model': best_model_name, 'strategy': best_strategy}, model_dir / 'tuned_model.joblib')
    print('Saved tuned model to', model_dir / 'tuned_model.joblib')


if __name__ == '__main__':
    main()
