"""
Árbol de decisión — clasificación (Gini) y regresión (MSE).

Árbol recursivo, Python puro, sin librerías externas.
Nodo hoja: {'leaf': True, 'value': v, 'modo': str}
Nodo interno: {'leaf': False, 'feature': j, 'threshold': t, 'left': ..., 'right': ..., 'modo': str}
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── Impurezas ─────────────────────────────────────────────────────────────────

def _gini(y):
    if not y:
        return 0.0
    n = len(y)
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _variance(y):
    if not y:
        return 0.0
    n = len(y)
    mean = sum(y) / n
    return sum((v - mean) ** 2 for v in y) / n


def _weighted_impurity(y_left, y_right, fn):
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    return (len(y_left) / n) * fn(y_left) + (len(y_right) / n) * fn(y_right)


# ── Construcción ──────────────────────────────────────────────────────────────

def _majority(y):
    counts = {}
    for v in y:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _build(X, y, depth, max_depth, min_samples, modo):
    n = len(y)
    impurity_fn = _gini if modo == 'clf' else _variance

    def make_leaf():
        val = _majority(y) if modo == 'clf' else sum(y) / n
        return {'leaf': True, 'value': val, 'modo': modo}

    # Condiciones de parada
    if depth >= max_depth or n <= min_samples:
        return make_leaf()
    if modo == 'clf' and len(set(y)) == 1:
        return make_leaf()
    if modo == 'reg' and _variance(y) == 0.0:
        return make_leaf()

    d = len(X[0])
    best_score = float('inf')
    best_feat = None
    best_thr = None
    best_left = None
    best_right = None

    for feat in range(d):
        vals = sorted(set(X[i][feat] for i in range(n)))
        for k in range(len(vals) - 1):
            thr = (vals[k] + vals[k + 1]) / 2.0
            left_idx  = [i for i in range(n) if X[i][feat] <= thr]
            right_idx = [i for i in range(n) if X[i][feat] >  thr]
            if not left_idx or not right_idx:
                continue
            y_l = [y[i] for i in left_idx]
            y_r = [y[i] for i in right_idx]
            score = _weighted_impurity(y_l, y_r, impurity_fn)
            if score < best_score:
                best_score = score
                best_feat = feat
                best_thr = thr
                best_left = (left_idx, y_l)
                best_right = (right_idx, y_r)

    if best_feat is None:
        return make_leaf()

    X_l = [X[i] for i in best_left[0]]
    X_r = [X[i] for i in best_right[0]]
    return {
        'leaf': False,
        'feature': best_feat,
        'threshold': best_thr,
        'left':  _build(X_l, best_left[1],  depth + 1, max_depth, min_samples, modo),
        'right': _build(X_r, best_right[1], depth + 1, max_depth, min_samples, modo),
        'modo': modo,
    }


def _predict_one(node, x):
    if node['leaf']:
        return node['value']
    if x[node['feature']] <= node['threshold']:
        return _predict_one(node['left'], x)
    return _predict_one(node['right'], x)


# ── API pública ───────────────────────────────────────────────────────────────

def tree_clf_fit(X, y, max_depth=5, min_samples=1):
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("tree_clf_fit: X debe ser lista no vacía.")
    if not isinstance(X[0], list):
        X = [[v] for v in X]
    return _build(X, y, 0, int(max_depth), int(min_samples), 'clf')


def tree_reg_fit(X, y, max_depth=5, min_samples=1):
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("tree_reg_fit: X debe ser lista no vacía.")
    if not isinstance(X[0], list):
        X = [[v] for v in X]
    return _build(X, y, 0, int(max_depth), int(min_samples), 'reg')


def tree_predict(modelo, X):
    """Despacha por modelo['modo']. Acepta matriz o vector."""
    if not isinstance(modelo, dict) or 'modo' not in modelo:
        raise TrezRuntimeError("tree_predict: modelo inválido.")
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("tree_predict: X debe ser lista no vacía.")
    if not isinstance(X[0], list):
        return _predict_one(modelo, X)
    return [_predict_one(modelo, row) for row in X]
