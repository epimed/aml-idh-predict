"""
Neural network training and prediction of IDH mutation status from transcriptomic data
"""

# ============================================================
# Parameters (final model)
# ============================================================


PARAMS = {
    "hidden_layers": {'dims': (64, 16), 'dropouts': (0.3, 0.2)},
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 200,
    # fixed parameters
    "patience": 20,
    "early_stop_fraction": 0.1,
    "prob_threshold": 0.5,
    "outer_cv_n_splits": 5,
}

# ============================================================
# Imports
# ============================================================
# import os
# import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_state = 0
seed_everything(random_state)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset
# ============================================================
class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# Neural network model
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        
        hidden_dims = hidden_layers['dims']
        dropouts = hidden_layers['dropouts']
    
        if len(hidden_dims) != len(dropouts):
            print('hidden_dims', hidden_dims)
            print('dropouts', dropouts)
            raise ValueError("hidden_dims and dropouts must have same length")

        layers = []
        in_dim = input_dim

        for h, d in zip(hidden_dims, dropouts):
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(d)
            ])
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze()

# ============================================================
# Training with early stopping and logging
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    max_epochs,
    patience,
    fold_id
):
    history = []
    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0
    best_epoch = 1

    for epoch in range(1, max_epochs + 1):
        # ---------------- Training
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ---------------- Validation
        model.eval()
        val_losses = []
        y_true, y_prob = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits)

                val_losses.append(loss.item())
                y_true.extend(yb.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob > PARAMS['prob_threshold']).astype(int)
        
        values, counts = np.unique(y_true, return_counts=True)
        support = dict(zip(values, counts))
        support['total'] = len(y_true)

        epoch_metrics = {
        "fold": fold_id,
        "epoch": epoch,
        "best_model_epoch": best_epoch,
        "train_loss": float(np.mean(train_losses)),
        "val_loss": float(np.mean(val_losses)),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        # Per-class metrics
        "precision_IDH_WT": precision_score(y_true, y_pred, pos_label=0),
        "recall_IDH_WT": recall_score(y_true, y_pred, pos_label=0),
        "f1_IDH_WT": f1_score(y_true, y_pred, pos_label=0),
        "precision_IDH_MUT": precision_score(y_true, y_pred, pos_label=1),
        "recall_IDH_MUT": recall_score(y_true, y_pred, pos_label=1),
        "f1_IDH_MUT": f1_score(y_true, y_pred, pos_label=1),
        # Support
        'support': support['total'],
        'support_IDH_WT': support[0],
        'support_IDH_MUT': support[1],
        }

        history.append(epoch_metrics)

        # ---------------- Early stopping
        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            best_state = model.state_dict()
            epochs_no_improve = 0
            best_epoch = epoch
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)

    return model, history

# ============================================================
# Evaluation
# ============================================================
def evaluate(model, loader):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            y_prob.extend(probs)
            y_true.extend(yb.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > PARAMS['prob_threshold']).astype(int)
    
    report_str = classification_report(y_true, y_pred, target_names=['IDH-WT', 'IDH-MUT'])
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(report_str)
    
    support_IDH_WT = int(np.sum(y_true == 0))
    support_IDH_MUT = int(np.sum(y_true == 1))
    support_total = int(len(y_true))
    
    metrics = {
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
        # Support
        "support_IDH_WT": support_IDH_WT,
        "support_IDH_MUT": support_IDH_MUT,
        "support_total": support_total,
    }

    predictions = {
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
        }

    return metrics, predictions

# ============================================================
# Nested cross-validation (final model)
# ============================================================
def nested_cv_training(X, y):
    outer_cv = StratifiedKFold(n_splits=PARAMS['outer_cv_n_splits'], shuffle=True, random_state=random_state)
    outer_results = []
    all_fold_history = []
    fold_predictions = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"\n=== Outer Fold {fold} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        id_test = id_samples[test_idx]

        # Standardization (training fold only)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Early-stopping split
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=PARAMS["early_stop_fraction"],
            random_state=random_state
        )
        tr_idx, val_idx = next(sss.split(X_train, y_train))

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        train_ds = GeneExpressionDataset(X_tr, y_tr)
        val_ds = GeneExpressionDataset(X_val, y_val)
        test_ds = GeneExpressionDataset(X_test, y_test)

        train_loader = DataLoader(
            train_ds,
            batch_size=PARAMS["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=PARAMS["batch_size"],
            shuffle=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=PARAMS["batch_size"],
            shuffle=False
        )

        pos_weight = torch.tensor(
            [(len(y_tr) - y_tr.sum()) / y_tr.sum()],
            device=DEVICE
        )

        model = MLP(
            input_dim=X.shape[1],
            hidden_layers=PARAMS["hidden_layers"],
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=PARAMS["learning_rate"],
            weight_decay=PARAMS["weight_decay"]
        )

        # ---------------- Train ----------------
        model, fold_history = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            PARAMS["epochs"],
            PARAMS["patience"],
            fold
        )
        
        all_fold_history.extend(fold_history)  # append to master list

        # ---------------- Evaluate on test ----------------
        fold_metrics, predictions = evaluate(model, test_loader)
        outer_results.append(fold_metrics)

        for i, _ in enumerate(predictions['y_true']):
            fold_predictions.append({
                'fold': fold,
                'element': i,
                'id_sample': id_test[i],
                'y_true': predictions['y_true'][i],
                'y_pred': predictions['y_pred'][i],
                'y_prob': predictions['y_prob'][i],
                })

    return outer_results, all_fold_history, fold_predictions


# ============================================================
# Final training on full dataset and prediction on new samples
# ============================================================

def train_final_model(X, y):
    """
    Train final model on the entire dataset using fixed hyperparameters.
    Early stopping is still applied using a stratified 10% validation split.
    Returns trained model and fitted scaler.
    """
    # ---------------- Standardization (fit on all data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------- Early-stopping split
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=PARAMS["early_stop_fraction"],
        random_state=random_state
    )
    train_idx, val_idx = next(sss.split(X_scaled, y))

    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    train_ds = GeneExpressionDataset(X_tr, y_tr)
    val_ds = GeneExpressionDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=PARAMS["batch_size"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=PARAMS["batch_size"],
        shuffle=False
    )

    # ---------------- Class imbalance handling
    pos_weight = torch.tensor(
        [(len(y_tr) - y_tr.sum()) / y_tr.sum()],
        device=DEVICE
    )

    model = MLP(
        input_dim=X.shape[1],
        hidden_layers=PARAMS["hidden_layers"]
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=PARAMS["learning_rate"],
        weight_decay=PARAMS["weight_decay"]
    )

    # ---------------- Train
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=PARAMS["epochs"],
        patience=PARAMS["patience"],
        fold_id="final"
    )

    return model, scaler, history

def predict_new_samples(model, scaler, X_new):
    """
    Predict IDH status for new samples.
    Returns probabilities and binary predictions.
    """
    model.eval()

    # Standardize using training scaler
    X_new_scaled = scaler.transform(X_new)
    X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X_new_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs > PARAMS["prob_threshold"]).astype(int)

    return probs, preds


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
    results, history, fold_predictions = nested_cv_training(X, y)
    

    # 3. Train final model on all data
    print('Training the final model on all data')
    final_model, final_scaler, final_history = train_final_model(X, y)
    
    
    # 4. Predict on new samples
    print('Predictions on new samples')
    X_new = X_orig
    id_samples_new = X_new.index.values
    X_new = X_new.values.astype(np.float32) 
    y_new_prob, y_new_pred = predict_new_samples(final_model, final_scaler, X_new)
    