"""
naive_bayes.py — Naive Bayes Gaussiano para TREZ
Según "Grokking Machine Learning" (Serrano, 2021) Cap 8.
P(y|X) ∝ P(y) * ∏ P(xi|y)  — asume independencia condicional
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _variance(vals):
    if len(vals) < 2:
        return 1e-9
    m = _mean(vals)
    return sum((v - m)**2 for v in vals) / len(vals)


def _gaussian_pdf(x, mean, var):
    if var <= 0:
        var = 1e-9
    coeff = 1.0 / math.sqrt(2 * math.pi * var)
    exponent = math.exp(-((x - mean)**2) / (2 * var))
    return coeff * exponent


def nb_fit(X, y):
    """
    Entrena Naive Bayes Gaussiano.
    X: [[x0,x1,...], ...], y: [label, ...]
    Retorna modelo dict con estadísticas por clase y prior.
    """
    classes = list(set(y))
    n = len(y)
    n_features = len(X[0]) if X else 0

    class_stats = {}
    for c in classes:
        indices = [i for i in range(n) if y[i] == c]
        prior = len(indices) / n
        feature_stats = []
        for f in range(n_features):
            vals = [X[i][f] for i in indices]
            feature_stats.append({"mean": _mean(vals), "var": _variance(vals)})
        class_stats[c] = {"prior": prior, "features": feature_stats}

    return {"classes": classes, "stats": class_stats, "n_features": n_features}


def nb_predict_proba(model, x):
    """
    Calcula log-probabilidad para cada clase dado x.
    Retorna dict {clase: log_prob}.
    """
    log_probs = {}
    for c in model["classes"]:
        stats = model["stats"][c]
        log_p = math.log(stats["prior"] + 1e-300)
        for f, fstats in enumerate(stats["features"]):
            pdf = _gaussian_pdf(x[f], fstats["mean"], fstats["var"])
            log_p += math.log(pdf + 1e-300)
        log_probs[c] = log_p
    return log_probs


def nb_predict(model, x):
    """Predice la clase más probable para un solo ejemplo x."""
    log_probs = nb_predict_proba(model, x)
    return max(log_probs, key=lambda c: log_probs[c])


def nb_predict_batch(model, X):
    """Predice clases para lista de ejemplos."""
    return [nb_predict(model, x) for x in X]
