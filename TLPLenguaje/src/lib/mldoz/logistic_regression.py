"""
Regresión logística binaria por gradiente descendente — sin librerías externas.

Modelo:    p = σ(X · w + b)            con σ(z) = 1/(1+e^(-z))
Loss:      log-loss = -(1/n) Σ [y·log(p) + (1-y)·log(1-p)]
Gradiente: ∂L/∂w[j] = (1/n) Σ (p_i − y_i) · X[i][j]
           ∂L/∂b    = (1/n) Σ (p_i − y_i)

Para multiclase, entrenar K modelos one-vs-rest desde el .trez
(el lenguaje funcional permite hacerlo sin bucles imperativos extra).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import exp_doz, log_doz
from errors import TrezRuntimeError


def _sigmoid(z):
    """Sigmoide numéricamente estable: evita overflow para z muy negativo."""
    if z >= 0:
        return 1.0 / (1.0 + exp_doz(-z))
    ez = exp_doz(z)
    return ez / (1.0 + ez)


def logreg_fit(X, y, lr=0.1, epochs=500):
    """
    X: matriz (n_muestras × n_features) o vector 1D.
    y: lista de 0/1.
    Retorna {tipo, w, b, historia}.
    """
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("logreg_fit: X debe ser lista no vacía.")
    if not isinstance(y, list) or len(y) != len(X):
        raise TrezRuntimeError("logreg_fit: y debe ser lista del mismo largo que X.")

    if not isinstance(X[0], list):
        X = [[v] for v in X]

    # Validar y ∈ {0, 1}
    for v in y:
        if v != 0 and v != 1:
            raise TrezRuntimeError(f"logreg_fit: y debe contener solo 0 o 1 (encontrado {v}).")

    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    historia = []
    eps = 1e-12

    for _ in range(int(epochs)):
        # Forward
        z = [sum(X[i][j] * w[j] for j in range(d)) + b for i in range(n)]
        yp = [_sigmoid(zi) for zi in z]
        err = [yp[i] - y[i] for i in range(n)]

        # Gradientes
        grad_w = [(1.0 / n) * sum(err[i] * X[i][j] for i in range(n)) for j in range(d)]
        grad_b = (1.0 / n) * sum(err)

        # Actualización
        w = [w[j] - lr * grad_w[j] for j in range(d)]
        b = b - lr * grad_b

        # Log-loss con clipping para evitar log(0)
        loss = -sum(
            y[i] * log_doz(yp[i] if yp[i] > eps else eps)
            + (1 - y[i]) * log_doz((1 - yp[i]) if (1 - yp[i]) > eps else eps)
            for i in range(n)
        ) / n
        historia.append(loss)

    return {'tipo': 'logreg', 'w': w, 'b': b, 'historia': historia}


def logreg_predict_proba(modelo, X):
    """Probabilidad P(y=1|x). Retorna lista (batch) o número (una muestra)."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'logreg':
        raise TrezRuntimeError("logreg_predict_proba: modelo inválido.")
    w, b = modelo['w'], modelo['b']
    d = len(w)
    if isinstance(X, list) and X and isinstance(X[0], list):
        return [_sigmoid(sum(row[j] * w[j] for j in range(d)) + b) for row in X]
    if isinstance(X, list):
        if len(X) != d:
            raise TrezRuntimeError(f"logreg_predict_proba: muestra con {len(X)} features, modelo espera {d}.")
        return _sigmoid(sum(X[j] * w[j] for j in range(d)) + b)
    raise TrezRuntimeError("logreg_predict_proba: X debe ser lista o matriz.")


def logreg_predict(modelo, X, umbral=0.5):
    """Etiqueta 0/1 según umbral sobre la probabilidad."""
    proba = logreg_predict_proba(modelo, X)
    if isinstance(proba, list):
        return [1 if p >= umbral else 0 for p in proba]
    return 1 if proba >= umbral else 0
