from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def compute_cost_thresholds(y_true, y_prob, fn_cost, fp_cost, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.01)

    records = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn_cost * fn + fp_cost * fp
        records.append({
            'threshold': float(threshold),
            'precision': float(precision_score(y_true, y_pred)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'cost': float(cost),
            'fp': int(fp),
            'fn': int(fn)
        })

    return pd.DataFrame(records)


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'data' / 'processed'
    model_dir = root / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description='Run Phase 3 evaluation and threshold optimization')
    parser.add_argument('--fn-cost', type=float, default=1000.0, help='Coût d une fausse négative')
    parser.add_argument('--fp-cost', type=float, default=50.0, help='Coût d une fausse positive')
    args = parser.parse_args()

    validation_df = pd.read_csv(data_dir / 'validation.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    X_val = validation_df.drop(columns=['bad_nutrition'])
    y_val = validation_df['bad_nutrition']
    X_test = test_df.drop(columns=['bad_nutrition'])
    y_test = test_df['bad_nutrition']

    trained = joblib.load(model_dir / 'tuned_model.joblib')
    model = trained['pipeline']

    print('Loaded tuned model:', trained.get('model'), 'strategy:', trained.get('strategy'))
    print('Validation set shape:', X_val.shape)
    print('Test set shape:', X_test.shape)

    y_prob_test = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob_test >= 0.5).astype(int)
    metrics_default = {
        'accuracy': float(accuracy_score(y_test, y_pred_default)),
        'precision': float(precision_score(y_test, y_pred_default)),
        'recall': float(recall_score(y_test, y_pred_default)),
        'f1': float(f1_score(y_test, y_pred_default)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_test))
    }

    print('Default threshold metrics (0.5):')
    print(metrics_default)
    print('Default confusion matrix:')
    print(confusion_matrix(y_test, y_pred_default))

    y_prob_val = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.01)
    records = []
    for threshold in thresholds:
        y_pred_val = (y_prob_val >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()
        cost = args.fn_cost * fn + args.fp_cost * fp
        records.append({
            'threshold': float(threshold),
            'precision': float(precision_score(y_val, y_pred_val)),
            'recall': float(recall_score(y_val, y_pred_val)),
            'f1': float(f1_score(y_val, y_pred_val)),
            'cost': float(cost),
            'fp': int(fp),
            'fn': int(fn)
        })

    threshold_df = pd.DataFrame(records)
    threshold_df.to_csv(model_dir / 'threshold_optimization.csv', index=False)

    best_row = threshold_df.loc[threshold_df['cost'].idxmin()]
    best_threshold = float(best_row['threshold'])

    print('Best threshold (validation):', best_threshold)
    print('Best validation cost:', best_row['cost'])
    print('Best threshold precision:', best_row['precision'])
    print('Best threshold recall:', best_row['recall'])
    print('Best threshold f1:', best_row['f1'])

    y_pred_test_opt = (y_prob_test >= best_threshold).astype(int)
    metrics_opt = {
        'accuracy': float(accuracy_score(y_test, y_pred_test_opt)),
        'precision': float(precision_score(y_test, y_pred_test_opt)),
        'recall': float(recall_score(y_test, y_pred_test_opt)),
        'f1': float(f1_score(y_test, y_pred_test_opt)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_test))
    }

    final_artifact = {
        'pipeline': model,
        'threshold': best_threshold,
        'cost_config': {'fn_cost': args.fn_cost, 'fp_cost': args.fp_cost},
        'metrics_default': metrics_default,
        'metrics_optimal': metrics_opt
    }
    joblib.dump(final_artifact, model_dir / 'final_model.joblib')
    print('Saved final model artifact to', model_dir / 'final_model.joblib')

    print('Optimal test metrics:')
    print(metrics_opt)


if __name__ == '__main__':
    main()
