"""
metrics_ext.py — Métricas extendidas para TREZ (ROC, AUC, PR curve, F1, etc.)
Según "Grokking Machine Learning" (Serrano, 2021) Cap 7.
Python puro, sin dependencias externas.
"""
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def _check_binary(y_true):
    classes = set(y_true)
    if not classes.issubset({0, 1}):
        raise TrezRuntimeError("ROC/AUC: etiquetas deben ser 0/1.")


def roc_curve(y_true, y_scores):
    """
    Calcula la curva ROC.
    y_true:   lista de etiquetas reales (0/1)
    y_scores: lista de probabilidades de clase positiva (float 0..1)

    Retorna {'fpr': [...], 'tpr': [...], 'thresholds': [...]}
    """
    _check_binary(y_true)
    n = len(y_true)
    if n == 0:
        raise TrezRuntimeError("roc_curve: listas vacías.")

    thresholds = sorted(set(y_scores), reverse=True)
    thresholds = [t + 1e-10 for t in thresholds[:1]] + thresholds + [thresholds[-1] - 1e-10]
    thresholds = sorted(set(thresholds), reverse=True)

    pos = sum(y_true)
    neg = n - pos
    if pos == 0 or neg == 0:
        raise TrezRuntimeError("roc_curve: necesita ejemplos de ambas clases.")

    fprs, tprs = [0.0], [0.0]
    for thresh in thresholds:
        tp = sum(1 for i in range(n) if y_scores[i] >= thresh and y_true[i] == 1)
        fp = sum(1 for i in range(n) if y_scores[i] >= thresh and y_true[i] == 0)
        fprs.append(fp / neg)
        tprs.append(tp / pos)

    fprs.append(1.0)
    tprs.append(1.0)
    return {'fpr': fprs, 'tpr': tprs, 'thresholds': thresholds}


def auc(fpr, tpr):
    """
    Calcula el área bajo la curva ROC usando la regla del trapecio.
    fpr, tpr: listas paralelas de float.
    """
    if len(fpr) != len(tpr):
        raise TrezRuntimeError("auc: fpr y tpr deben tener el mismo largo.")
    area = 0.0
    for i in range(1, len(fpr)):
        dx = fpr[i] - fpr[i-1]
        area += dx * (tpr[i] + tpr[i-1]) / 2.0
    return abs(area)


def auc_roc(y_true, y_scores):
    """Atajo: calcula ROC y devuelve solo el AUC."""
    curve = roc_curve(y_true, y_scores)
    return auc(curve['fpr'], curve['tpr'])


def precision_recall_curve(y_true, y_scores):
    """
    Curva Precision-Recall.
    Retorna {'precision': [...], 'recall': [...], 'thresholds': [...]}
    """
    _check_binary(y_true)
    n = len(y_true)
    thresholds = sorted(set(y_scores), reverse=True)

    precisions, recalls = [], []
    for thresh in thresholds:
        tp = sum(1 for i in range(n) if y_scores[i] >= thresh and y_true[i] == 1)
        fp = sum(1 for i in range(n) if y_scores[i] >= thresh and y_true[i] == 0)
        fn = sum(1 for i in range(n) if y_scores[i] < thresh and y_true[i] == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    return {'precision': precisions, 'recall': recalls, 'thresholds': thresholds}


def f1_score(y_true, y_pred):
    """F1 = 2 * precision * recall / (precision + recall)."""
    _check_binary(y_true)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def confusion_matrix_vals(y_true, y_pred):
    """Retorna {'tp': int, 'tn': int, 'fp': int, 'fn': int}."""
    _check_binary(y_true)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
