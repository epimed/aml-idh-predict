"""
Logistic regression for prediction of IDH mutation status from transcriptomic data
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

random_state = 0
seed_everything(random_state)

# ============================================================
# Hyperparameter grid
# ============================================================
PARAM_GRID = {
     # Variance filtering threshold
    "var__threshold": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    "clf__C": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "clf__l1_ratio": [0, 1],
    "clf__max_iter": [500]
}

PARAMS = {
    "prob_threshold": 0.5,
    "search_best_threshold": False,
    }

N_OUTER_SPLITS = 5
N_INNER_SPLITS = 4

# ============================================================
# Nested cross-validation
# ============================================================
def nested_cv(X, y):
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=random_state
    )

    outer_results = []
    fold_cv_results = []

    best_thres_across_folds = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\n=== Outer Fold {fold} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ----------------------------------------------------
        # Pipeline (scaler fit ONLY on training fold)
        # ----------------------------------------------------
        
        clf = LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                random_state=random_state,
            )
        
        pipeline = Pipeline([
            ("var", VarianceThreshold()), 
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=random_state
        )

        scoring = {
            "roc_auc": "roc_auc",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1"
        }
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRID,
            scoring=scoring,
            refit="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            return_train_score=False
        )

        # ----------------------------------------------------
        # Hyperparameter tuning (training data only)
        # ----------------------------------------------------
        grid.fit(X_train, y_train)
        
        cv_results = pd.DataFrame(grid.cv_results_)
        cv_results["outer_fold"] = fold
        cv_results["scoring"] = "roc_auc"
        
        # print('cv_results')
        # print(cv_results)

        print("Best hyperparameters:")
        for k, v in grid.best_params_.items():
            print(f"  {k}: {v}")

        # ----------------------------------------------------
        # Evaluation on held-out test fold
        # ----------------------------------------------------
        best_model = grid.best_estimator_
        
        y_true = np.array(y_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        
        # -------------------------
        # Probability threshold optimization for IDH-MUT precision
        # -------------------------
        
        thresholds = [PARAMS["prob_threshold"]]
        
        if PARAMS["search_best_threshold"]:
            thresholds = np.linspace(0, 0.99, 100)
        
        best_threshold = PARAMS["prob_threshold"]
        best_precision = 0.0
        best_specificity = 0.0

        for t in thresholds:
            y_pred_t = (y_prob > t).astype(int)
            # precision = precision_score(y_true, y_pred_t, pos_label=1)
            specificity = recall_score(y_true, y_pred_t, pos_label=0)
            # if precision > best_precision:
                # best_precision = precision
                # best_threshold = t
            if specificity > best_specificity: 
                best_specificity = specificity
                best_threshold = t

        print(f"Optimal threshold for highest IDH-MUT precision: {best_threshold:.2f}")
        y_pred = (y_prob > best_threshold).astype(int)
        
        cv_results['best_threshold'] = best_threshold
        cv_results['best_precision'] = best_precision
        best_thres_across_folds.append(best_threshold)
    
        metrics = {
            "fold": fold,
            "optimal_prob_threshold": best_threshold,
            "roc_auc": roc_auc_score(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision_IDH_WT": precision_score(y_true, y_pred, pos_label=0),
            "recall_IDH_WT": recall_score(y_true, y_pred, pos_label=0),
            "f1_IDH_WT": f1_score(y_true, y_pred, pos_label=0),
            "precision_IDH_MUT": precision_score(y_true, y_pred, pos_label=1),
            "recall_IDH_MUT": recall_score(y_true, y_pred, pos_label=1),
            "f1_IDH_MUT": f1_score(y_true, y_pred, pos_label=1),
            "support_IDH_WT": int(np.sum(y_true == 0)),
            "support_IDH_MUT": int(np.sum(y_true == 1)),
            "support_total": int(len(y_true))
        }

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(
            classification_report(
                y_test,
                (best_model.predict_proba(X_test)[:, 1] > best_threshold).astype(int),
                target_names=["IDH-WT", "IDH-MUT"],
                digits=4
            )
        )

        fold_cv_results.append(cv_results)
        outer_results.append(metrics)

    return outer_results, fold_cv_results

# ============================================================
# Main
# X: gene expression matrix, shape (n_samples, n_genes)
# y: binary IDH labels, 0 = IDH-WT, 1 = IDH-MUT
# ============================================================


if __name__ == "__main__":
    
    annot_filename = '../data/idh_mutation_status.csv'
    print('Importing annotation file:', annot_filename)
    y = pd.read_csv(annot_filename, sep=';', index_col=0)
    print('Done.')
    y = y['idh_mutant'].dropna()
    print('y', y.shape)
    print(y.head())
    
    data_filename = '../data/expression_data_pooled_19_AML_datasets_pycombat_corrected.csv'
    print('\nImporting data file:', data_filename)
    print('Please wait...')
    X = pd.read_csv(data_filename, sep=';', index_col=0)
    X = X.loc[y.index]
    print('Done.')
    print('X', X.shape)
    print(X.head())
    
    print('\nNested cross-validations in progress')
    print('Please wait...')
    X = X.values.astype(np.float32)  # converts DataFrame to 2D NumPy array
    y = y.values.astype(int)         # ensure integer labels {'IDH-WT': 0, 'IDH-MUT': 1}
    outer_results, cv_results = nested_cv(X, y)
    print('Done.')
    
    # Uncomment the lines below to save results
    # grid_search_scores = pd.concat(cv_results, axis=0, ignore_index=True)
    # grid_search_scores.to_csv("grid_search_scores_LR.csv", index=True, sep=';')