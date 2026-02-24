"""
Neural Network classifier for prediction of IDH mutation status
from transcriptomic data
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    ParameterGrid
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score
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
seed_everything(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Parameters
# ============================================================

grid_search_params = {
    "hidden_layers": [
        {'dims': (64, 16), 'dropouts': (0.3, 0.3)},
        {'dims': (64, 16), 'dropouts': (0.3, 0.2)},
        {'dims': (64, 16), 'dropouts': (0.3, 0.1)},
        {'dims': (64, 16), 'dropouts': (0.3, 0.0)},
        ],
    "batch_size": [32, 64],
    "learning_rate": [1e-4, 5e-5],
    "weight_decay": [1e-4, 1e-5],
    "epochs": [100, 200],
}

input_params = {
    "patience": 20,
    "early_stop_fraction": 0.1,
    "outer_cv_splits": 5,
    "inner_cv_splits": 4,
    "prob_threshold": 0.5,
    }


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
# Model
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
# Training with Early Stopping
# ============================================================
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    max_epochs,
    patience
):
    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(
                    criterion(model(xb), yb).item()
                )

        mean_val_loss = np.mean(val_losses)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return model

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
    y_pred = (y_prob > input_params['prob_threshold']).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }

# ============================================================
# Nested CV + Grid Search (Early-Stop Fraction Included)
# ============================================================
def nested_cv_training(X, y):

    outer_cv = StratifiedKFold(
        n_splits=input_params['outer_cv_splits'], shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=input_params['inner_cv_splits'], shuffle=True, random_state=random_state
    )

    outer_results = []
    all_inner_cv_results = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n=== Outer Fold {fold + 1} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        best_score = -np.inf
        best_params = None

        # ---------------- Inner CV (hyperparameter selection)
        for params in ParameterGrid(grid_search_params):
            inner_scores = []

            for inner_fold, (inner_train_idx, _) in enumerate(inner_cv.split(X_train, y_train)):
                X_inner = X_train[inner_train_idx]
                y_inner = y_train[inner_train_idx]

                sss = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=input_params["early_stop_fraction"],
                    random_state=random_state
                )
                tr_idx, val_idx = next(sss.split(X_inner, y_inner))

                X_tr, X_val = X_inner[tr_idx], X_inner[val_idx]
                y_tr, y_val = y_inner[tr_idx], y_inner[val_idx]

                train_ds = GeneExpressionDataset(X_tr, y_tr)
                val_ds = GeneExpressionDataset(X_val, y_val)

                train_loader = DataLoader(
                    train_ds,
                    batch_size=params["batch_size"],
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=params["batch_size"],
                    shuffle=False
                )

                pos_weight = torch.tensor(
                    [(len(y_tr) - y_tr.sum()) / y_tr.sum()],
                    device=DEVICE
                )

                model = MLP(
                    input_dim=X.shape[1],
                    hidden_layers=params["hidden_layers"]
                ).to(DEVICE)

                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=params["learning_rate"],
                    weight_decay=params["weight_decay"]
                )

                model = train_model(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    params["epochs"],
                    input_params["patience"]
                )

                metrics = evaluate(model, val_loader)
               
                inner_scores.append(metrics["roc_auc"])

                all_inner_cv_results.append({
                    "outer_fold": fold + 1,
                    "inner_fold": inner_fold + 1,
                    "hidden_layers": params["hidden_layers"],
                    "batch_size": params["batch_size"],
                    "learning_rate": params["learning_rate"],
                    "weight_decay": params["weight_decay"],
                    "epochs": params["epochs"],
                    "roc_auc": metrics["roc_auc"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                })

            mean_auc = np.mean(inner_scores)
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params

        print("Best hyperparameters:")
        for k, v in best_params.items():
            print(f"{k}: {v}")

        # ---------------- Final outer-fold training
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=input_params["early_stop_fraction"],
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
            batch_size=best_params["batch_size"],
            shuffle=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=best_params["batch_size"],
            shuffle=False
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=best_params["batch_size"],
            shuffle=False
        )

        pos_weight = torch.tensor(
            [(len(y_tr) - y_tr.sum()) / y_tr.sum()],
            device=DEVICE
        )

        final_model = MLP(
            input_dim=X.shape[1],
            hidden_layers=best_params["hidden_layers"]
        ).to(DEVICE)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            final_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"]
        )

        final_model = train_model(
            final_model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            best_params["epochs"],
            input_params["patience"]
        )

        fold_metrics = evaluate(final_model, test_loader)
        outer_results.append(fold_metrics)

        print("Outer fold results:", fold_metrics)

    return outer_results, all_inner_cv_results


# ============================================================
# Main
# X shape: (n_samples, n_genes)
# y shape: (n_samples,)
# ============================================================


import pandas as pd

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
    outer_results, inner_results = nested_cv_training(X, y)
    print('Done.')

    # Uncomment the lines below to save results
    # df_inner = pd.DataFrame(inner_results)
    # df_inner.to_csv("grid_search_scores_NN.csv", index=True, sep=';')