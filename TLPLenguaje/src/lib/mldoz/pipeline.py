"""
pipeline.py — Utilidades de pipeline de ML para TREZ
Según "Grokking Machine Learning" (Serrano, 2021) Cap 13.
Python puro, sin dependencias externas.

Incluye: K-Fold CV, Grid Search, normalización, estandarización, train/val/test split.
"""
import math, random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── Splits ─────────────────────────────────────────────────────────────────────
def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=None):
    """
    Divide (X, y) en train / val / test.
    Retorna {'train_X', 'train_y', 'val_X', 'val_y', 'test_X', 'test_y'}
    """
    if seed is not None:
        random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        'train_X': [X[i] for i in train_idx],
        'train_y': [y[i] for i in train_idx],
        'val_X': [X[i] for i in val_idx],
        'val_y': [y[i] for i in val_idx],
        'test_X': [X[i] for i in test_idx],
        'test_y': [y[i] for i in test_idx],
    }


# ── K-Fold Cross-Validation ────────────────────────────────────────────────────
def kfold_split(X, y, k=5, shuffle=True, seed=None):
    """
    Genera k folds de (train_X, train_y, val_X, val_y).
    Retorna lista de k dicts.
    """
    if k < 2:
        raise TrezRuntimeError("kfold_split: k debe ser >= 2.")
    n = len(X)
    indices = list(range(n))
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append({
            'train_X': [X[j] for j in train_idx],
            'train_y': [y[j] for j in train_idx],
            'val_X': [X[j] for j in val_idx],
            'val_y': [y[j] for j in val_idx],
        })
    return folds


def kfold_cv(fit_fn, predict_fn, score_fn, X, y, k=5, seed=None, **fit_kwargs):
    """
    Ejecuta k-fold cross-validation.
    fit_fn(X_train, y_train, **fit_kwargs) -> model
    predict_fn(model, X_val) -> y_pred
    score_fn(y_true, y_pred) -> float

    Retorna {'scores': [float,...], 'mean': float, 'std': float}
    """
    folds = kfold_split(X, y, k, seed=seed)
    scores = []
    for fold in folds:
        model = fit_fn(fold['train_X'], fold['train_y'], **fit_kwargs)
        preds = predict_fn(model, fold['val_X'])
        scores.append(score_fn(fold['val_y'], preds))

    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean)**2 for s in scores) / len(scores))
    return {'scores': scores, 'mean': mean, 'std': std}


# ── Grid Search ────────────────────────────────────────────────────────────────
def _dict_product(param_grid):
    """Genera producto cartesiano de param_grid dict."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    def _product(pools):
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        return result

    combos = _product(values)
    return [dict(zip(keys, combo)) for combo in combos]


def grid_search(fit_fn, predict_fn, score_fn, X, y, param_grid, k=5, seed=None):
    """
    Grid search con k-fold CV.
    param_grid: dict {param_name: [val1, val2, ...]}
    fit_fn(X, y, **params) -> model
    predict_fn(model, X_val) -> y_pred
    score_fn(y_true, y_pred) -> float (más alto = mejor)

    Retorna {'best_params': dict, 'best_score': float, 'results': [...]}
    """
    combos = _dict_product(param_grid)
    best_score = -float('inf')
    best_params = None
    results = []

    for params in combos:
        cv = kfold_cv(fit_fn, predict_fn, score_fn, X, y, k, seed, **params)
        results.append({'params': params, 'mean': cv['mean'], 'std': cv['std']})
        if cv['mean'] > best_score:
            best_score = cv['mean']
            best_params = params

    return {'best_params': best_params, 'best_score': best_score, 'results': results}


# ── Normalización y Estandarización nativas ─────────────────────────────────
def normalize_minmax(X):
    """
    Min-Max normalization por columna.
    Retorna (X_norm, {'min': [...], 'max': [...]})
    """
    if not X:
        return X, {}
    n_feats = len(X[0])
    mins = [min(row[f] for row in X) for f in range(n_feats)]
    maxs = [max(row[f] for row in X) for f in range(n_feats)]
    ranges = [maxs[f] - mins[f] if maxs[f] != mins[f] else 1.0 for f in range(n_feats)]
    X_norm = [[(row[f] - mins[f]) / ranges[f] for f in range(n_feats)] for row in X]
    return X_norm, {'min': mins, 'max': maxs}


def normalize_apply(X, params):
    """Aplica normalización min-max guardada (p.ej., del train set)."""
    n_feats = len(X[0])
    mins, maxs = params['min'], params['max']
    ranges = [maxs[f] - mins[f] if maxs[f] != mins[f] else 1.0 for f in range(n_feats)]
    return [[(row[f] - mins[f]) / ranges[f] for f in range(n_feats)] for row in X]


def standardize(X):
    """
    Z-score standardization por columna.
    Retorna (X_std, {'mean': [...], 'std': [...]})
    """
    if not X:
        return X, {}
    n_feats = len(X[0])
    means = [sum(row[f] for row in X) / len(X) for f in range(n_feats)]
    stds = [
        math.sqrt(sum((row[f] - means[f])**2 for row in X) / len(X)) or 1.0
        for f in range(n_feats)
    ]
    X_std = [[(row[f] - means[f]) / stds[f] for f in range(n_feats)] for row in X]
    return X_std, {'mean': means, 'std': stds}


def standardize_apply(X, params):
    """Aplica estandarización guardada."""
    n_feats = len(X[0])
    means, stds = params['mean'], params['std']
    return [[(row[f] - means[f]) / stds[f] for f in range(n_feats)] for row in X]
