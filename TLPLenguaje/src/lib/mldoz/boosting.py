"""
boosting.py — AdaBoost y Gradient Boosting para TREZ
Según "Grokking Machine Learning" (Serrano, 2021) Cap 12.
Python puro, sin dependencias externas.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── Stump: árbol de decisión de profundidad 1 ─────────────────────────────────
def _best_stump(X, y, weights):
    """
    Encuentra el mejor stump (feature, threshold, polarity) minimizando error ponderado.
    y: etiquetas {-1, 1}
    """
    n, n_feats = len(X), len(X[0])
    best_err = float('inf')
    best = None

    for f in range(n_feats):
        vals = sorted(set(row[f] for row in X))
        thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals) - 1)]
        if not thresholds:
            thresholds = [vals[0]]

        for thresh in thresholds:
            for polarity in [1, -1]:
                preds = [polarity if X[i][f] < thresh else -polarity for i in range(n)]
                err = sum(weights[i] for i in range(n) if preds[i] != y[i])
                if err < best_err:
                    best_err = err
                    best = {"feature": f, "threshold": thresh,
                            "polarity": polarity, "error": err}
    return best


def _stump_predict(stump, x):
    f, t, p = stump["feature"], stump["threshold"], stump["polarity"]
    return p if x[f] < t else -p


# ── AdaBoost ──────────────────────────────────────────────────────────────────
def adaboost_fit(X, y, n_estimators=50):
    """
    Entrena AdaBoost binario.
    y: lista de etiquetas convertibles a {-1, 1}
    Retorna modelo con lista de (stump, alpha).
    """
    n = len(X)
    # Convertir etiquetas a {-1, 1}
    classes = sorted(set(y))
    if len(classes) != 2:
        raise TrezRuntimeError("AdaBoost: solo clasificación binaria.")
    label_map = {classes[0]: -1, classes[1]: 1}
    inv_map = {-1: classes[0], 1: classes[1]}
    y_bin = [label_map[yi] for yi in y]

    weights = [1.0 / n] * n
    stumps = []

    for _ in range(n_estimators):
        stump = _best_stump(X, y_bin, weights)
        err = max(stump["error"], 1e-10)
        alpha = 0.5 * math.log((1 - err) / err)
        stump["alpha"] = alpha

        # Actualizar pesos
        new_w = []
        for i in range(n):
            pred = _stump_predict(stump, X[i])
            new_w.append(weights[i] * math.exp(-alpha * y_bin[i] * pred))
        total = sum(new_w)
        weights = [w / total for w in new_w]

        stumps.append(stump)

    return {"stumps": stumps, "inv_map": inv_map}


def adaboost_predict(model, x):
    """Predice la clase para un ejemplo x."""
    score = sum(s["alpha"] * _stump_predict(s, x) for s in model["stumps"])
    label = 1 if score >= 0 else -1
    return model["inv_map"][label]


def adaboost_predict_batch(model, X):
    return [adaboost_predict(model, x) for x in X]


# ── Gradient Boosting (regresión) ─────────────────────────────────────────────
def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _mse_split(left_y, right_y):
    def _mse(ys):
        if not ys:
            return 0.0
        m = _mean(ys)
        return sum((v - m)**2 for v in ys) / len(ys)
    n = len(left_y) + len(right_y)
    return (len(left_y) * _mse(left_y) + len(right_y) * _mse(right_y)) / n if n else 0


def _fit_tree(X, residuals, max_depth=3):
    """Árbol de regresión simple para residuos."""
    def build(indices, depth):
        ys = [residuals[i] for i in indices]
        if depth == 0 or len(indices) < 2:
            return {"leaf": True, "value": _mean(ys)}
        best_err, best_split = float('inf'), None
        n_feats = len(X[0])
        for f in range(n_feats):
            vals = sorted(set(X[i][f] for i in indices))
            thresholds = [(vals[j] + vals[j+1]) / 2 for j in range(len(vals) - 1)]
            for t in thresholds:
                left = [i for i in indices if X[i][f] < t]
                right = [i for i in indices if X[i][f] >= t]
                if not left or not right:
                    continue
                err = _mse_split([residuals[i] for i in left],
                                  [residuals[i] for i in right])
                if err < best_err:
                    best_err = err
                    best_split = (f, t, left, right)
        if best_split is None:
            return {"leaf": True, "value": _mean(ys)}
        f, t, left, right = best_split
        return {"leaf": False, "feature": f, "threshold": t,
                "left": build(left, depth - 1),
                "right": build(right, depth - 1)}
    return build(list(range(len(X))), max_depth)


def _tree_predict(tree, x):
    if tree["leaf"]:
        return tree["value"]
    if x[tree["feature"]] < tree["threshold"]:
        return _tree_predict(tree["left"], x)
    return _tree_predict(tree["right"], x)


def gbm_fit(X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Gradient Boosting para regresión.
    Retorna modelo con base_prediction y lista de árboles.
    """
    base = _mean(y)
    preds = [base] * len(y)
    trees = []

    for _ in range(n_estimators):
        residuals = [y[i] - preds[i] for i in range(len(y))]
        tree = _fit_tree(X, residuals, max_depth)
        trees.append(tree)
        for i in range(len(y)):
            preds[i] += learning_rate * _tree_predict(tree, X[i])

    return {"base": base, "trees": trees, "lr": learning_rate}


def gbm_predict(model, x):
    """Predice valor para un ejemplo x."""
    pred = model["base"]
    for tree in model["trees"]:
        pred += model["lr"] * _tree_predict(tree, x)
    return pred


def gbm_predict_batch(model, X):
    return [gbm_predict(model, x) for x in X]
