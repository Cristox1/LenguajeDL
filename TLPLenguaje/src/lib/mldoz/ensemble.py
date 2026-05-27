"""
ensemble.py — Bagging, Random Forest y XGBoost-style para TREZ
Según "Grokking Machine Learning" (Serrano, 2021) Cap 12.
Python puro, sin dependencias externas.
"""
import math, random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── Árbol de decisión base (reutilizable) ─────────────────────────────────────
def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _gini(groups, classes):
    total = sum(len(g) for g in groups)
    score = 0.0
    for g in groups:
        size = len(g)
        if size == 0:
            continue
        p_sum = 0.0
        for c in classes:
            p = sum(1 for row in g if row[-1] == c) / size
            p_sum += p * p
        score += (1.0 - p_sum) * (size / total)
    return score


def _split(index, value, dataset):
    left = [row for row in dataset if row[index] < value]
    right = [row for row in dataset if row[index] >= value]
    return left, right


def _best_split(dataset, n_features=None, feature_indices=None):
    classes = list(set(row[-1] for row in dataset))
    best_idx, best_val, best_score, best_groups = None, None, float('inf'), None
    n_cols = len(dataset[0]) - 1

    if feature_indices is None:
        feature_indices = range(n_cols)

    for idx in feature_indices:
        for row in dataset:
            groups = _split(idx, row[idx], dataset)
            gini = _gini(groups, classes)
            if gini < best_score:
                best_idx, best_val, best_score, best_groups = idx, row[idx], gini, groups
    return {'index': best_idx, 'value': best_val, 'groups': best_groups}


def _to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def _split_node(node, max_depth, min_size, depth, n_features):
    left, right = node['groups']
    del node['groups']
    if not left or not right:
        node['left'] = node['right'] = _to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = _to_terminal(left), _to_terminal(right)
        return
    # Left
    if len(left) <= min_size:
        node['left'] = _to_terminal(left)
    else:
        feat_idx = random.sample(range(len(left[0]) - 1), min(n_features, len(left[0]) - 1))
        node['left'] = _best_split(left, feature_indices=feat_idx)
        _split_node(node['left'], max_depth, min_size, depth + 1, n_features)
    # Right
    if len(right) <= min_size:
        node['right'] = _to_terminal(right)
    else:
        feat_idx = random.sample(range(len(right[0]) - 1), min(n_features, len(right[0]) - 1))
        node['right'] = _best_split(right, feature_indices=feat_idx)
        _split_node(node['right'], max_depth, min_size, depth + 1, n_features)


def _build_tree(train, max_depth, min_size, n_features):
    n_cols = len(train[0]) - 1
    feat_idx = random.sample(range(n_cols), min(n_features, n_cols))
    root = _best_split(train, feature_indices=feat_idx)
    _split_node(root, max_depth, min_size, 1, n_features)
    return root


def _predict_tree(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return _predict_tree(node['left'], row)
        return node['left']
    else:
        if isinstance(node['right'], dict):
            return _predict_tree(node['right'], row)
        return node['right']


# ── Bootstrap sampling ─────────────────────────────────────────────────────────
def _bootstrap_sample(dataset, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(dataset)
    return [random.choice(dataset) for _ in range(n)]


# ── Bagging ───────────────────────────────────────────────────────────────────
def bagging_fit(X, y, n_estimators=10, max_depth=5, min_size=1, seed=None):
    """
    Bagging Classifier.
    Entrena n_estimators árboles sobre muestras bootstrap.
    """
    if seed is not None:
        random.seed(seed)
    dataset = [list(X[i]) + [y[i]] for i in range(len(X))]
    n_features = len(X[0])

    trees = []
    for _ in range(n_estimators):
        sample = _bootstrap_sample(dataset)
        tree = _build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    return {'trees': trees, 'type': 'bagging'}


def bagging_predict(model, x):
    votes = [_predict_tree(tree, x) for tree in model['trees']]
    return max(set(votes), key=votes.count)


def bagging_predict_batch(model, X):
    return [bagging_predict(model, x) for x in X]


# ── Random Forest ──────────────────────────────────────────────────────────────
def rf_fit(X, y, n_estimators=10, max_depth=5, min_size=1,
           max_features=None, seed=None):
    """
    Random Forest: Bagging + selección aleatoria de features por split.
    max_features: número de features aleatorias por split (default: sqrt(n_feats))
    """
    if seed is not None:
        random.seed(seed)
    dataset = [list(X[i]) + [y[i]] for i in range(len(X))]
    n_cols = len(X[0])
    n_feats = max_features or max(1, int(math.sqrt(n_cols)))

    trees = []
    for _ in range(n_estimators):
        sample = _bootstrap_sample(dataset)
        tree = _build_tree(sample, max_depth, min_size, n_feats)
        trees.append(tree)
    return {'trees': trees, 'type': 'random_forest'}


def rf_predict(model, x):
    votes = [_predict_tree(tree, x) for tree in model['trees']]
    return max(set(votes), key=votes.count)


def rf_predict_batch(model, X):
    return [rf_predict(model, x) for x in X]


# ── XGBoost-style: Gradient Boosting con regularización ──────────────────────
def _mse(ys):
    if not ys:
        return 0.0
    m = _mean(ys)
    return sum((v - m)**2 for v in ys) / len(ys)


def _xgb_fit_tree(X, residuals, max_depth=3, reg_lambda=1.0):
    """Árbol de regresión con regularización L2 en las hojas (estilo XGBoost)."""
    def _leaf_value(indices):
        g = sum(residuals[i] for i in indices)
        h = len(indices)  # segunda derivada = 1 para MSE
        return g / (h + reg_lambda)

    def build(indices, depth):
        if depth == 0 or len(indices) < 2:
            return {"leaf": True, "value": _leaf_value(indices)}
        best_gain, best_split = -INF, None
        n_feats = len(X[0])
        G_total = sum(residuals[i] for i in indices)
        H_total = float(len(indices))

        for f in range(n_feats):
            vals = sorted(set(X[i][f] for i in indices))
            for j in range(len(vals) - 1):
                t = (vals[j] + vals[j+1]) / 2
                left = [i for i in indices if X[i][f] < t]
                right = [i for i in indices if X[i][f] >= t]
                if not left or not right:
                    continue
                GL = sum(residuals[i] for i in left)
                HL = float(len(left))
                GR = G_total - GL
                HR = H_total - HL
                gain = (GL**2 / (HL + reg_lambda) + GR**2 / (HR + reg_lambda)
                        - G_total**2 / (H_total + reg_lambda)) / 2
                if gain > best_gain:
                    best_gain = gain
                    best_split = (f, t, left, right)

        if best_split is None:
            return {"leaf": True, "value": _leaf_value(indices)}
        f, t, left_i, right_i = best_split
        return {
            "leaf": False, "feature": f, "threshold": t,
            "left": build(left_i, depth - 1),
            "right": build(right_i, depth - 1),
        }
    return build(list(range(len(X))), max_depth)


INF = float('inf')


def _xgb_tree_predict(tree, x):
    if tree["leaf"]:
        return tree["value"]
    return (_xgb_tree_predict(tree["left"], x) if x[tree["feature"]] < tree["threshold"]
            else _xgb_tree_predict(tree["right"], x))


def xgb_fit(X, y, n_estimators=100, learning_rate=0.1, max_depth=3, reg_lambda=1.0):
    """
    XGBoost-style Gradient Boosting con regularización L2.
    Retorna modelo con árboles y predicciones base.
    """
    base = _mean(y)
    preds = [base] * len(y)
    trees = []

    for _ in range(n_estimators):
        residuals = [y[i] - preds[i] for i in range(len(y))]
        tree = _xgb_fit_tree(X, residuals, max_depth, reg_lambda)
        trees.append(tree)
        for i in range(len(y)):
            preds[i] += learning_rate * _xgb_tree_predict(tree, X[i])

    return {'base': base, 'trees': trees, 'lr': learning_rate}


def xgb_predict(model, x):
    pred = model['base']
    for tree in model['trees']:
        pred += model['lr'] * _xgb_tree_predict(tree, x)
    return pred


def xgb_predict_batch(model, X):
    return [xgb_predict(model, x) for x in X]
