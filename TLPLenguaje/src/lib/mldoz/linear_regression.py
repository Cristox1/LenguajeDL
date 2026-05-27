"""
Regresión lineal por gradiente descendente — sin librerías externas.

Modelo:  y_pred = X · w + b
Loss:    MSE = (1/n) Σ (y_pred − y)²
Gradiente:
    grad_w[j] = (2/n) Σ (y_pred_i − y_i) · X[i][j]
    grad_b    = (2/n) Σ (y_pred_i − y_i)
Actualización:
    w ← w − lr · grad_w     b ← b − lr · grad_b
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def linreg_fit(X, y, lr=0.01, epochs=500):
    """
    X: lista de listas (n_muestras × n_features). Aceptamos también vectores 1D
       (n_muestras,) tratándolos como una sola feature.
    y: lista de números (n_muestras,).
    Retorna {tipo, w, b, historia} donde historia es la pérdida MSE por época.
    """
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("linreg_fit: X debe ser lista no vacía.")
    if not isinstance(y, list) or len(y) != len(X):
        raise TrezRuntimeError("linreg_fit: y debe ser lista del mismo largo que X.")

    # Normalizar X a 2D para uniformidad
    if not isinstance(X[0], list):
        X = [[v] for v in X]

    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    historia = []

    for _ in range(int(epochs)):
        # Forward: y_pred = X · w + b
        yp = [sum(X[i][j] * w[j] for j in range(d)) + b for i in range(n)]
        residuo = [yp[i] - y[i] for i in range(n)]

        # Gradientes
        grad_w = [(2.0 / n) * sum(residuo[i] * X[i][j] for i in range(n)) for j in range(d)]
        grad_b = (2.0 / n) * sum(residuo)

        # Actualización
        w = [w[j] - lr * grad_w[j] for j in range(d)]
        b = b - lr * grad_b

        # Loss MSE
        loss = sum(r * r for r in residuo) / n
        historia.append(loss)

    return {'tipo': 'linreg', 'w': w, 'b': b, 'historia': historia}


def linreg_predict(modelo, X):
    """Aplica y = X · w + b. Acepta matriz (batch) o vector (una muestra)."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'linreg':
        raise TrezRuntimeError("linreg_predict: modelo inválido (esperado tipo='linreg').")
    w = modelo['w']
    b = modelo['b']
    d = len(w)

    if isinstance(X, list) and X and isinstance(X[0], list):
        return [sum(row[j] * w[j] for j in range(d)) + b for row in X]
    if isinstance(X, list):
        # Vector 1D: una sola muestra con d features
        if len(X) != d:
            raise TrezRuntimeError(f"linreg_predict: muestra con {len(X)} features, modelo espera {d}.")
        return sum(X[j] * w[j] for j in range(d)) + b
    raise TrezRuntimeError("linreg_predict: X debe ser lista o matriz.")
