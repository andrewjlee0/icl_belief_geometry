"""Linear probes WITH BIAS for belief-state decoding.

The probe fits y = X @ W + b via OLS by augmenting X with a ones column.
EVERY probe in this codebase uses bias for consistency.
"""
import torch
import numpy as np

def _augment(X):
    """Append ones column for bias."""
    return torch.cat([X, torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)], dim=1)

def fit_and_evaluate_multi(X_train, X_test, targets, use_bias=True):
    """Fit probes for multiple targets sharing X. Returns {name: R²}.
    targets: {name: (Y_train, Y_test)}
    Pseudoinverse computed once and reused.
    """
    Xa_tr = _augment(X_train) if use_bias else X_train
    Xa_te = _augment(X_test) if use_bias else X_test
    P = torch.linalg.pinv(Xa_tr)
    results = {}
    for tname, (Y_tr, Y_te) in targets.items():
        W = P @ Y_tr
        pred = Xa_te @ W
        ss_res = ((Y_te - pred) ** 2).sum().item()
        ss_tot = ((Y_te - Y_te.mean(dim=0)) ** 2).sum().item()
        results[tname] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return results

def fit_probe(X_train, Y_train, use_bias=True):
    """Fit and return weight matrix W (last row = bias if use_bias)."""
    Xa = _augment(X_train) if use_bias else X_train
    return torch.linalg.pinv(Xa) @ Y_train

def predict_probe(X, W, use_bias=True):
    Xa = _augment(X) if use_bias else X
    return Xa @ W

def compute_r2(Y_true, Y_pred):
    ss_res = ((Y_true - Y_pred) ** 2).sum().item()
    ss_tot = ((Y_true - Y_true.mean(dim=0)) ** 2).sum().item()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def fit_and_evaluate(X_tr, Y_tr, X_te, Y_te, use_bias=True):
    W = fit_probe(X_tr, Y_tr, use_bias)
    return compute_r2(Y_te, predict_probe(X_te, W, use_bias))

def fit_and_evaluate_multi_stacked(X_train, X_test, Y_train_stacked, Y_test_stacked, n_states, K_values, use_bias=True):
    """For redr2: Y is horizontally stacked [beliefs_k1 | beliefs_k2 | ...].
    Fits ONE probe, slices R² per k. Returns {k: R²}.
    """
    Xa_tr = _augment(X_train) if use_bias else X_train
    Xa_te = _augment(X_test) if use_bias else X_test
    P = torch.linalg.pinv(Xa_tr)
    W = P @ Y_train_stacked
    pred = Xa_te @ W
    results = {}
    for ki, k in enumerate(K_values):
        sl = slice(ki * n_states, (ki + 1) * n_states)
        ss_res = ((Y_test_stacked[:, sl] - pred[:, sl]) ** 2).sum().item()
        ss_tot = ((Y_test_stacked[:, sl] - Y_test_stacked[:, sl].mean(dim=0)) ** 2).sum().item()
        results[k] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return results
