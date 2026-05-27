"""
K-Means — agrupamiento sin librerías externas.

Inicialización: k puntos distintos al azar con Randomdoz.sample.
Iteración:      asignar cada punto al centroide más cercano → recalcular centroides.
Convergencia:   cuando las asignaciones no cambian o se alcanza max_iter.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz
from lib.randomdoz.randomdoz import sample as _rand_sample
from errors import TrezRuntimeError


def _dist(a, b):
    return sqrt_doz(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _centroid(points):
    if not points:
        return None
    d = len(points[0])
    return [sum(p[j] for p in points) / len(points) for j in range(d)]


def kmeans_fit(X, k, max_iter=100):
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("kmeans_fit: X debe ser lista no vacía.")
    k = int(k)
    if k < 1 or k > len(X):
        raise TrezRuntimeError(f"kmeans_fit: k={k} fuera de rango [1, {len(X)}].")
    if not isinstance(X[0], list):
        X = [[v] for v in X]

    n = len(X)
    # Inicializar centroides con k puntos distintos
    idx_init = _rand_sample(list(range(n)), k)
    centroides = [list(X[i]) for i in idx_init]

    asignaciones = [-1] * n

    for _ in range(int(max_iter)):
        nuevas = [
            min(range(k), key=lambda ci: _dist(X[i], centroides[ci]))
            for i in range(n)
        ]
        if nuevas == asignaciones:
            break
        asignaciones = nuevas

        for ci in range(k):
            puntos = [X[i] for i in range(n) if asignaciones[i] == ci]
            if puntos:
                centroides[ci] = _centroid(puntos)

    return {'tipo': 'kmeans', 'centroides': centroides, 'asignaciones': asignaciones, 'k': k}


def kmeans_predict(modelo, X):
    """Índice del centroide más cercano para cada punto."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'kmeans':
        raise TrezRuntimeError("kmeans_predict: modelo inválido.")
    centroides = modelo['centroides']
    k = len(centroides)

    def _pred(x):
        return min(range(k), key=lambda ci: _dist(x, centroides[ci]))

    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("kmeans_predict: X debe ser lista no vacía.")
    if not isinstance(X[0], list):
        X = [[v] for v in X]
    return [_pred(x) for x in X]
