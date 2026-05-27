"""
K-Vecinos más cercanos — lazy learner, sin librerías externas.

El modelo es solo una copia de (X, y, k). La predicción busca los k vecinos
más cercanos en distancia euclidiana y devuelve moda (clasif) o media (regr).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz
from errors import TrezRuntimeError


def _dist(a, b):
    return sqrt_doz(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _moda(vals):
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])


def knn_fit(X, y, k=3):
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("knn_fit: X debe ser lista no vacía.")
    if not isinstance(y, list) or len(y) != len(X):
        raise TrezRuntimeError("knn_fit: y debe tener el mismo largo que X.")
    k = int(k)
    if k < 1 or k > len(X):
        raise TrezRuntimeError(f"knn_fit: k={k} fuera de rango [1, {len(X)}].")
    if not isinstance(X[0], list):
        X = [[v] for v in X]
    return {'tipo': 'knn', 'X': X, 'y': y, 'k': k}


def _k_nearest(modelo, x):
    X_tr, y_tr, k = modelo['X'], modelo['y'], modelo['k']
    if not isinstance(x, list):
        x = [x]
    dists = [(i, _dist(x, X_tr[i])) for i in range(len(X_tr))]
    dists.sort(key=lambda t: t[1])
    return [y_tr[i] for i, _ in dists[:k]]


def knn_predict_clf(modelo, X):
    """Clasificación: mayoría de votos."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'knn':
        raise TrezRuntimeError("knn_predict_clf: modelo inválido.")
    if isinstance(X, list) and X and isinstance(X[0], list):
        return [_moda(_k_nearest(modelo, row)) for row in X]
    return _moda(_k_nearest(modelo, X))


def knn_predict_reg(modelo, X):
    """Regresión: media de los k vecinos."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'knn':
        raise TrezRuntimeError("knn_predict_reg: modelo inválido.")
    def _pred(x):
        vecinos = _k_nearest(modelo, x)
        return sum(vecinos) / len(vecinos)
    if isinstance(X, list) and X and isinstance(X[0], list):
        return [_pred(row) for row in X]
    return _pred(X)


def knn_predict(modelo, X):
    """Despacha clasif o regr según el tipo de etiquetas en y."""
    y = modelo.get('y', [])
    if y and isinstance(y[0], (int, float)):
        return knn_predict_reg(modelo, X)
    return knn_predict_clf(modelo, X)
