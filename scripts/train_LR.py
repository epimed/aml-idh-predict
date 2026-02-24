"""
Logistic Regression prediction of IDH mutation status from transcriptomic data
"""

# ============================================================
# Imports
# ============================================================
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold
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
# Fixed hyperparameters (from grid search)
# ============================================================
PARAMS = {
    "solver": "liblinear",          # stable for high-dimensional data
    "l1_ratio": 0,                # l1_ratio=0 → pure L2
    "C": 8.0, # 3.0, 
    "class_weight": "balanced",
    "max_iter": 500,
    "n_splits": 5,
    "var_threshold": 0.4,
    "prob_threshold": 0.5,
}

# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)

random_state = 0
seed_everything(random_state)


# ============================================================
# Cross-validation evaluation
# ============================================================
def cross_validate_model(X, y):
    cv = StratifiedKFold(
        n_splits=PARAMS["n_splits"],
        shuffle=True,
        random_state=random_state
    )

    fold_metrics = []
    fold_predictions = []
    fold_coefs = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n=== Fold {fold} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        id_test = id_samples[test_idx]


        clf = LogisticRegression(
                solver=PARAMS["solver"],
                l1_ratio=PARAMS["l1_ratio"],
                C=PARAMS["C"],
                class_weight=PARAMS["class_weight"],
                max_iter=PARAMS["max_iter"],
                random_state=random_state
            )

        pipeline = Pipeline([
            ("var", VarianceThreshold(threshold=PARAMS["var_threshold"])),
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        pipeline.fit(X_train, y_train)
        
        
        log_reg = pipeline.named_steps["clf"]
        coef = log_reg.coef_.ravel()      # shape: (n_features,)
        intercept = log_reg.intercept_[0]
        
        var_selector = pipeline.named_steps["var"]
        support_mask = var_selector.get_support()

        coef = log_reg.coef_.ravel()

        fold_coefs.append({
            "fold": fold,
            "intercept": intercept,
            "coefs": coef,
            "feature_mask": support_mask,
            })
        
        y_true = np.array(y_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > PARAMS["prob_threshold"]).astype(int)
        
    
        metrics = {
            "fold": fold,
            "prob_threshold": PARAMS["prob_threshold"],
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

        fold_metrics.append(metrics)
        
        print(
            f"Features kept after variance filtering: "
            f"{pipeline.named_steps['var'].get_support().sum()} / {X.shape[1]}"
            )

        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=["IDH-WT", "IDH-MUT"],
                digits=4
            )
        )
        
        for i, _ in enumerate(y_true):
            fold_predictions.append({
                'fold': fold,
                'element': i,
                'id_sample': id_test[i],
                'y_true': y_true[i],
                'y_pred': y_pred[i],
                'y_prob': y_prob[i],
                })  

    return fold_metrics, fold_predictions, fold_coefs

# ============================================================
# Final training on full dataset
# ============================================================
def train_final_model(X, y):
    
    clf = LogisticRegression(
                solver=PARAMS["solver"],
                l1_ratio=PARAMS["l1_ratio"],
                C=PARAMS["C"],
                class_weight=PARAMS["class_weight"],
                max_iter=PARAMS["max_iter"],
                random_state=random_state
            )
    
    final_model = Pipeline([
        ("var", VarianceThreshold(threshold=PARAMS["var_threshold"])),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    final_model.fit(X, y)
    
    log_reg = final_model.named_steps["clf"]
    var_selector = final_model.named_steps["var"]
    support_mask = var_selector.get_support()
    intercept = log_reg.intercept_[0]
    coef = log_reg.coef_.ravel()

    coefs = {
         "intercept": intercept,
         "coefs": coef,
         "feature_mask": support_mask,
        }
    
    return final_model, coefs

# ============================================================
# Prediction on new samples
# ============================================================
def predict_new_samples(model, X_new):
    prob = model.predict_proba(X_new)[:, 1]
    pred = (prob > PARAMS["prob_threshold"]).astype(int)
    return prob, pred


# ============================================================
# Main
# ============================================================

import pandas as pd

if __name__ == "__main__":
    
    # 1. Import data
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
    X_orig = pd.read_csv(data_filename, sep=';', index_col=0)
    X = X_orig.loc[y.index]
    print('Done.')
    print('X', X.shape)
    print(X.head())
    
    id_samples = X.index.values
    
    X = X.values.astype(np.float32)  # converts DataFrame to 2D NumPy array
    y = y.values.astype(int)         # ensure integer labels {'IDH-WT': 0, 'IDH-MUT': 1}
    
    
    # 2. Cross-validation evaluation
    print('Cross-validation evaluation')
    fold_metrics, fold_predictions, fold_coefs = cross_validate_model(X, y)
    
    
    # 3. Train final model on all data
    print('Training the final model on all data')
    final_model, coefs = train_final_model(X, y)
    
    
    # 4. Predict on new samples
    print('Predictions on new samples')
    X_new = X_orig
    id_samples_new = X_new.index.values
    X_new = X_new.values.astype(np.float32) 
    y_new_prob, y_new_pred = predict_new_samples(final_model, X_new)
