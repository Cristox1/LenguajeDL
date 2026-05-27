"""
svm_kernel.py — SVM con kernel (RBF, polinomial, lineal) para TREZ
Según "Grokking Machine Learning" (Serrano, 2021) Cap 11.
Implementación dual con SMO simplificado. Python puro, sin dependencias.

Optimización: SMO simplificado (Platt, 1998) con heurísticas básicas.
"""
import math, random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── Kernels ───────────────────────────────────────────────────────────────────
def kernel_linear(x1, x2):
    return sum(a * b for a, b in zip(x1, x2))


def kernel_rbf(x1, x2, gamma=0.5):
    dist_sq = sum((a - b)**2 for a, b in zip(x1, x2))
    return math.exp(-gamma * dist_sq)


def kernel_poly(x1, x2, degree=3, coef=1.0):
    return (sum(a * b for a, b in zip(x1, x2)) + coef) ** degree


def kernel_sigmoid(x1, x2, alpha=0.01, coef=0.0):
    return math.tanh(alpha * sum(a * b for a, b in zip(x1, x2)) + coef)


# ── SMO simplificado ──────────────────────────────────────────────────────────
def _smo_fit(X, y, kernel_fn, C=1.0, tol=1e-3, max_passes=50):
    n = len(X)
    alphas = [0.0] * n
    b = 0.0

    def decision(i):
        return sum(alphas[j] * y[j] * kernel_fn(X[j], X[i]) for j in range(n)) + b

    passes = 0
    while passes < max_passes:
        changed = 0
        for i in range(n):
            Ei = decision(i) - y[i]
            if (y[i] * Ei < -tol and alphas[i] < C) or (y[i] * Ei > tol and alphas[i] > 0):
                # Elegir j ≠ i aleatorio
                j = random.randint(0, n - 2)
                if j >= i:
                    j += 1
                Ej = decision(j) - y[j]

                alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                if y[i] != y[j]:
                    L = max(0.0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0.0, alphas[i] + alphas[j] - C)
                    H = min(C, alphas[i] + alphas[j])

                if abs(H - L) < 1e-12:
                    continue

                eta = 2 * kernel_fn(X[i], X[j]) - kernel_fn(X[i], X[i]) - kernel_fn(X[j], X[j])
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = max(L, min(H, alphas[j]))

                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = (b - Ei
                      - y[i] * (alphas[i] - alpha_i_old) * kernel_fn(X[i], X[i])
                      - y[j] * (alphas[j] - alpha_j_old) * kernel_fn(X[i], X[j]))
                b2 = (b - Ej
                      - y[i] * (alphas[i] - alpha_i_old) * kernel_fn(X[i], X[j])
                      - y[j] * (alphas[j] - alpha_j_old) * kernel_fn(X[j], X[j]))

                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                changed += 1
        if changed == 0:
            passes += 1
        else:
            passes = 0

    # Vectores de soporte
    sv_idx = [i for i in range(n) if alphas[i] > 1e-5]
    return {
        'alphas': alphas,
        'b': b,
        'sv_X': [X[i] for i in sv_idx],
        'sv_y': [y[i] for i in sv_idx],
        'sv_alpha': [alphas[i] for i in sv_idx],
        'kernel_fn': kernel_fn,
    }


# ── API pública ───────────────────────────────────────────────────────────────
def ksvm_fit(X, y, kernel='rbf', C=1.0, gamma=0.5, degree=3, coef=1.0,
             tol=1e-3, max_passes=50, seed=None):
    """
    Entrena SVM con kernel.
    kernel: 'rbf' | 'linear' | 'poly' | 'sigmoid'
    y: etiquetas en {-1,+1} o {0,1} — se convierten a {-1,+1}
    """
    if seed is not None:
        random.seed(seed)
    if not X:
        raise TrezRuntimeError("ksvm_fit: X vacío.")
    if not isinstance(X[0], list):
        X = [[v] for v in X]

    classes = sorted(set(y))
    if len(classes) != 2:
        raise TrezRuntimeError("ksvm_fit: solo clasificación binaria.")
    label_map = {classes[0]: -1, classes[1]: 1}
    inv_map = {-1: classes[0], 1: classes[1]}
    y_bin = [label_map[yi] for yi in y]

    if kernel == 'rbf':
        kfn = lambda a, b: kernel_rbf(a, b, gamma)
    elif kernel == 'poly':
        kfn = lambda a, b: kernel_poly(a, b, degree, coef)
    elif kernel == 'sigmoid':
        kfn = lambda a, b: kernel_sigmoid(a, b, gamma, coef)
    else:
        kfn = kernel_linear

    model = _smo_fit(X, y_bin, kfn, C, tol, max_passes)
    model['inv_map'] = inv_map
    return model


def ksvm_predict(model, x):
    """Predice la clase para un ejemplo x."""
    if not isinstance(x, list):
        x = [x]
    score = (sum(model['sv_alpha'][i] * model['sv_y'][i] * model['kernel_fn'](model['sv_X'][i], x)
                 for i in range(len(model['sv_alpha'])))
             + model['b'])
    return model['inv_map'][1 if score >= 0 else -1]


def ksvm_predict_batch(model, X):
    return [ksvm_predict(model, x) for x in X]


def ksvm_decision(model, x):
    """Valor de decisión sin thresholding (útil para ROC)."""
    if not isinstance(x, list):
        x = [x]
    return (sum(model['sv_alpha'][i] * model['sv_y'][i] * model['kernel_fn'](model['sv_X'][i], x)
                for i in range(len(model['sv_alpha'])))
            + model['b'])
