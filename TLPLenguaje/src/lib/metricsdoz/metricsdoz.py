"""
Metricsdoz — métricas de clasificación y regresión, Python puro.

Sin librerías externas. Usa sqrt_doz y abs_doz de core_mathdoz para los
cálculos de regresión (RMSE, MAE).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz
from errors import TrezRuntimeError


# ── Helper común ─────────────────────────────────────────────────────────────

def _check_pair(y_true, y_pred, name):
    if not isinstance(y_true, list) or not isinstance(y_pred, list):
        raise TrezRuntimeError(f"{name}: y_true e y_pred deben ser listas.")
    if len(y_true) != len(y_pred):
        raise TrezRuntimeError(
            f"{name}: largos distintos ({len(y_true)} vs {len(y_pred)})."
        )
    if len(y_true) == 0:
        raise TrezRuntimeError(f"{name}: listas vacías.")


# ── Clasificación ────────────────────────────────────────────────────────────

def accuracy(y_true, y_pred):
    """Fracción de predicciones correctas. Siempre en [0, 1]."""
    _check_pair(y_true, y_pred, "accuracy")
    aciertos = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return aciertos / len(y_true)


def precision(y_true, y_pred, clase=1):
    """TP / (TP + FP) para la clase indicada. 0.0 si no hubo predicciones positivas."""
    _check_pair(y_true, y_pred, "precision")
    tp = sum(1 for t, p in zip(y_true, y_pred) if p == clase and t == clase)
    fp = sum(1 for t, p in zip(y_true, y_pred) if p == clase and t != clase)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true, y_pred, clase=1):
    """TP / (TP + FN) para la clase indicada. 0.0 si no había positivos reales."""
    _check_pair(y_true, y_pred, "recall")
    tp = sum(1 for t, p in zip(y_true, y_pred) if p == clase and t == clase)
    fn = sum(1 for t, p in zip(y_true, y_pred) if p != clase and t == clase)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true, y_pred, clase=1):
    """Media armónica de precision y recall."""
    p = precision(y_true, y_pred, clase)
    r = recall(y_true, y_pred, clase)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def confusion_matrix(y_true, y_pred, clases=None):
    """
    Matriz de confusión K×K. Filas = clase real, columnas = clase predicha.
    Retorna {'matriz': [[...]], 'clases': [c0, c1, ...]} para que el caller
    sepa el orden de los ejes.
    """
    _check_pair(y_true, y_pred, "confusion_matrix")
    if clases is None:
        univ = set(y_true) | set(y_pred)
        try:
            clases = sorted(univ)
        except TypeError:
            clases = sorted(univ, key=str)
    else:
        clases = list(clases)
    idx = {c: i for i, c in enumerate(clases)}
    K = len(clases)
    M = [[0] * K for _ in range(K)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t]][idx[p]] += 1
    return {'matriz': M, 'clases': clases}


# ── Regresión ────────────────────────────────────────────────────────────────

def mse(y_true, y_pred):
    """Error Cuadrático Medio."""
    _check_pair(y_true, y_pred, "mse")
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    """Raíz del MSE — mismas unidades que y."""
    return sqrt_doz(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """Error Absoluto Medio. Robusto a outliers comparado con MSE."""
    _check_pair(y_true, y_pred, "mae")
    return sum(abs_doz(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true, y_pred):
    """
    Coeficiente de determinación: 1 − SS_res / SS_tot.
    Vale 1.0 si predicción perfecta; 0.0 si predice la media; negativo si peor que la media.
    """
    _check_pair(y_true, y_pred, "r2_score")
    n = len(y_true)
    media = sum(y_true) / n
    ss_tot = sum((t - media) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
