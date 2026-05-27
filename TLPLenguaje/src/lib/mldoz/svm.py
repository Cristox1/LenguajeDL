"""
SVM lineal por descenso de subgradiente del hinge loss.

L(w,b) = (λ/2)||w||² + (1/n) Σ max(0, 1 − y_i·(w·x_i + b))   y ∈ {-1, +1}

Subgradiente:
  si margen ≥ 1:  grad_w = λ·w,          grad_b = 0
  si margen < 1:  grad_w = λ·w − y_i·x_i, grad_b = −y_i
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def svm_fit(X, y, lr=0.01, epochs=1000, lam=0.01):
    """
    X: lista de listas (n × d) o lista 1D (n,).
    y: lista de etiquetas en {-1, +1} o {0, 1} — se convierte a {-1,+1}.
    """
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("svm_fit: X debe ser lista no vacía.")
    if not isinstance(y, list) or len(y) != len(X):
        raise TrezRuntimeError("svm_fit: y debe tener el mismo largo que X.")

    if not isinstance(X[0], list):
        X = [[v] for v in X]

    # Convertir etiquetas a {-1, +1}
    clases = list(set(y))
    if len(clases) != 2:
        raise TrezRuntimeError(f"svm_fit: SVM lineal requiere 2 clases, encontradas {len(clases)}.")
    clases_sorted = sorted(clases)
    label_map = {clases_sorted[0]: -1, clases_sorted[1]: 1}
    y_pm = [label_map[v] for v in y]
    inv_map = {-1: clases_sorted[0], 1: clases_sorted[1]}

    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    historia = []

    for _ in range(int(epochs)):
        grad_w = [lam * wi for wi in w]
        grad_b = 0.0
        loss_sum = 0.0

        for i in range(n):
            margin = y_pm[i] * (sum(w[j] * X[i][j] for j in range(d)) + b)
            hinge = max(0.0, 1.0 - margin)
            loss_sum += hinge
            if margin < 1.0:
                for j in range(d):
                    grad_w[j] -= y_pm[i] * X[i][j] / n
                grad_b -= y_pm[i] / n

        w = [w[j] - lr * grad_w[j] for j in range(d)]
        b = b - lr * grad_b
        historia.append(lam / 2.0 * sum(wi * wi for wi in w) + loss_sum / n)

    return {'tipo': 'svm', 'w': w, 'b': b, 'historia': historia,
            'label_map': label_map, 'inv_map': inv_map}


def svm_predict(modelo, X):
    """Retorna etiqueta original (la misma escala que el y de entrenamiento)."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'svm':
        raise TrezRuntimeError("svm_predict: modelo inválido.")
    w, b = modelo['w'], modelo['b']
    inv_map = modelo['inv_map']
    d = len(w)

    def _predict_one(x):
        score = sum(w[j] * x[j] for j in range(d)) + b
        return inv_map[1 if score >= 0 else -1]

    if isinstance(X, list) and X and isinstance(X[0], list):
        return [_predict_one(row) for row in X]
    if isinstance(X, list):
        if len(X) != d:
            raise TrezRuntimeError(f"svm_predict: muestra con {len(X)} features, modelo espera {d}.")
        return _predict_one(X)
    raise TrezRuntimeError("svm_predict: X debe ser lista o matriz.")
