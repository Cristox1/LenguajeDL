import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz
from errors import TrezRuntimeError


# ── helpers de álgebra interna ────────────────────────────────────────────────

def _mean_col(X, j):
    return sum(row[j] for row in X) / len(X)

def _std_col(X, j, mean):
    v = sum((row[j] - mean) ** 2 for row in X) / len(X)
    return sqrt_doz(v) if v > 0 else 1.0

def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def _mat_col(X, j):
    return [row[j] for row in X]

def _outer(u, v):
    return [[ui * vj for vj in v] for ui in u]

def _mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def _scale_vec(v, s):
    return [vi * s for vi in v]

def _norm(v):
    return sqrt_doz(sum(vi ** 2 for vi in v))

def _project(v, u):
    """Proyección de v sobre u (u unitario)."""
    return _scale_vec(u, _dot(v, u))

def _gram_schmidt(vecs):
    """Ortogonaliza una lista de vectores con Gram-Schmidt."""
    basis = []
    for v in vecs:
        w = list(v)
        for b in basis:
            proj = _dot(w, b)
            w = [w[i] - proj * b[i] for i in range(len(w))]
        n = _norm(w)
        if n > 1e-10:
            basis.append([wi / n for wi in w])
    return basis

def _covariance_matrix(X_centered, n, d):
    """Cov = (1/n) X^T X"""
    cov = [[0.0] * d for _ in range(d)]
    for row in X_centered:
        for i in range(d):
            for j in range(d):
                cov[i][j] += row[i] * row[j]
    return [[cov[i][j] / n for j in range(d)] for i in range(d)]

def _power_iteration(cov, d, n_iter=200):
    """Estima el eigenvector dominante de cov por iteración de potencias."""
    from lib.randomdoz.randomdoz import uniform as _rand_u
    v = [_rand_u(0.5, 1.5) for _ in range(d)]
    n = _norm(v)
    v = [vi / n for vi in v]
    for _ in range(n_iter):
        w = [sum(cov[i][j] * v[j] for j in range(d)) for i in range(d)]
        n = _norm(w)
        if n < 1e-12:
            break
        v = [wi / n for wi in w]
    eigenval = _dot([sum(cov[i][j] * v[j] for j in range(d)) for i in range(d)], v)
    return eigenval, v

def _deflate(cov, eigenval, eigenvec, d):
    """Deflación: cov ← cov − λ v vT"""
    outer = _outer(eigenvec, eigenvec)
    return [[cov[i][j] - eigenval * outer[i][j] for j in range(d)] for i in range(d)]


# ── PCA ───────────────────────────────────────────────────────────────────────

def pca_fit(X, n_components=2):
    """
    Ajusta PCA sobre X (lista de listas n×d). Sin NumPy — power iteration.
    Retorna modelo: {mean, components, explained_variance, n_components}
    """
    if not X or not isinstance(X[0], list):
        raise TrezRuntimeError("Preprocessdoz.pca_fit(): X debe ser matriz (lista de listas).")
    n = len(X)
    d = len(X[0])
    k = int(n_components)
    if k < 1 or k > d:
        raise TrezRuntimeError(f"Preprocessdoz.pca_fit(): n_components={k} fuera de rango [1,{d}].")

    mean = [_mean_col(X, j) for j in range(d)]
    X_c = [[row[j] - mean[j] for j in range(d)] for row in X]
    cov = _covariance_matrix(X_c, n, d)

    components = []
    variances = []
    cov_rem = [list(row) for row in cov]

    for _ in range(k):
        lam, vec = _power_iteration(cov_rem, d)
        components.append(vec)
        variances.append(abs(lam))
        cov_rem = _deflate(cov_rem, lam, vec, d)

    total_var = sum(variances) + 1e-12
    explained = [v / total_var for v in variances]

    return {
        'tipo': 'pca',
        'mean': mean,
        'components': components,
        'explained_variance': variances,
        'explained_ratio': explained,
        'n_components': k,
    }


def pca_transform(modelo, X):
    """
    Proyecta X (n×d) al espacio PCA (n×k).
    Retorna lista de listas n×k.
    """
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'pca':
        raise TrezRuntimeError("Preprocessdoz.pca_transform(): modelo inválido.")
    if not X or not isinstance(X[0], list):
        raise TrezRuntimeError("Preprocessdoz.pca_transform(): X debe ser matriz.")
    mean = modelo['mean']
    comps = modelo['components']
    d = len(mean)
    X_c = [[row[j] - mean[j] for j in range(d)] for row in X]
    return [[_dot(row, comp) for comp in comps] for row in X_c]


def pca_fit_transform(X, n_components=2):
    """Ajusta y transforma en un solo paso."""
    modelo = pca_fit(X, n_components)
    return pca_transform(modelo, X), modelo


def pca_explained_ratio(modelo):
    """Retorna lista de ratios de varianza explicada por componente."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'pca':
        raise TrezRuntimeError("Preprocessdoz.pca_explained_ratio(): modelo inválido.")
    return modelo['explained_ratio']


# ── Imputación de valores faltantes ──────────────────────────────────────────

def impute_mean(data):
    """
    Rellena None/NaN en cada columna numérica con la media de esa columna.
    data: lista de dicts (salida de read_csv/read_xlsx).
    Retorna nueva lista de dicts con valores imputados.
    """
    if not data:
        return data
    cols = list(data[0].keys())
    col_means = {}
    for col in cols:
        vals = [row[col] for row in data if isinstance(row[col], (int, float))]
        col_means[col] = sum(vals) / len(vals) if vals else 0.0

    result = []
    for row in data:
        new_row = {}
        for col in cols:
            v = row[col]
            if v is None or (isinstance(v, float) and v != v):
                new_row[col] = col_means[col]
            else:
                new_row[col] = v
        result.append(new_row)
    return result


def impute_median(data):
    """Rellena None/NaN con la mediana de cada columna numérica."""
    if not data:
        return data
    cols = list(data[0].keys())
    col_medians = {}
    for col in cols:
        vals = sorted(row[col] for row in data if isinstance(row[col], (int, float)))
        if not vals:
            col_medians[col] = 0.0
        else:
            mid = len(vals) // 2
            col_medians[col] = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

    result = []
    for row in data:
        new_row = {}
        for col in cols:
            v = row[col]
            if v is None or (isinstance(v, float) and v != v):
                new_row[col] = col_medians[col]
            else:
                new_row[col] = v
        result.append(new_row)
    return result


def impute_constant(data, value=0):
    """Rellena None/NaN con un valor constante."""
    if not data:
        return data
    cols = list(data[0].keys())
    result = []
    for row in data:
        new_row = {}
        for col in cols:
            v = row[col]
            if v is None or (isinstance(v, float) and v != v):
                new_row[col] = value
            else:
                new_row[col] = v
        result.append(new_row)
    return result


def drop_nulls(data):
    """Elimina filas que tengan al menos un None/NaN."""
    def _has_null(row):
        for v in row.values():
            if v is None or (isinstance(v, float) and v != v):
                return True
        return False
    return [row for row in data if not _has_null(row)]


def null_count(data):
    """Retorna dict {columna: cantidad_de_nulos} para diagnóstico."""
    if not data:
        return {}
    cols = list(data[0].keys())
    counts = {}
    for col in cols:
        counts[col] = sum(
            1 for row in data
            if row[col] is None or (isinstance(row[col], float) and row[col] != row[col])
        )
    return counts


# ── write_xlsx ────────────────────────────────────────────────────────────────

def write_xlsx(data, filepath):
    """Escribe lista de dicts como .xlsx. Requiere openpyxl."""
    if not data:
        raise TrezRuntimeError("Preprocessdoz.write_xlsx(): data vacía.")
    try:
        import openpyxl
    except ImportError:
        raise TrezRuntimeError("Preprocessdoz.write_xlsx(): openpyxl no instalado. pip install openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    headers = list(data[0].keys())
    ws.append(headers)
    for row in data:
        ws.append([row.get(h, '') for h in headers])
    try:
        wb.save(filepath)
        return True
    except Exception as e:
        raise TrezRuntimeError(f"Preprocessdoz.write_xlsx(): error guardando '{filepath}': {e}")


# ── Regresión logística multiclase (softmax one-vs-rest) ─────────────────────

def logreg_multi_fit(X, y, lr=0.1, epochs=500):
    """
    Regresión logística multiclase via softmax + cross-entropy. Sin librerías.
    X: (n, d). y: lista de enteros 0..K-1.
    Retorna modelo con W (K×d) y b (K,).
    """
    if not X or not isinstance(X[0], list):
        raise TrezRuntimeError("logreg_multi_fit: X debe ser matriz.")
    n = len(X)
    d = len(X[0])
    classes = sorted(set(int(v) for v in y))
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}
    y_int = [cls_idx[int(v)] for v in y]

    from lib.mathdoz.core_mathdoz import exp_doz, log_doz
    eps = 1e-12

    W = [[0.0] * d for _ in range(K)]
    b = [0.0] * K
    historia = []

    for _ in range(int(epochs)):
        # Forward: logits → softmax
        logits = [[sum(W[k][j] * X[i][j] for j in range(d)) + b[k] for k in range(K)] for i in range(n)]
        probs = []
        for row in logits:
            mx = max(row)
            exps = [exp_doz(v - mx) for v in row]
            s = sum(exps) + eps
            probs.append([e / s for e in exps])

        # Loss: cross-entropy
        loss = -sum(log_doz(probs[i][y_int[i]] + eps) for i in range(n)) / n
        historia.append(loss)

        # Grad: dL/dlogits = (probs - one_hot) / n
        grad = [[probs[i][k] - (1.0 if y_int[i] == k else 0.0) for k in range(K)] for i in range(n)]

        # Actualizar W y b
        for k in range(K):
            for j in range(d):
                W[k][j] -= lr * sum(grad[i][k] * X[i][j] for i in range(n)) / n
            b[k] -= lr * sum(grad[i][k] for i in range(n)) / n

    return {'tipo': 'logreg_multi', 'W': W, 'b': b, 'classes': classes, 'historia': historia}


def logreg_multi_predict(modelo, X):
    """Predice clase para cada fila de X."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'logreg_multi':
        raise TrezRuntimeError("logreg_multi_predict: modelo inválido.")
    from lib.mathdoz.core_mathdoz import exp_doz
    W, b, classes = modelo['W'], modelo['b'], modelo['classes']
    K = len(classes)
    d = len(W[0])
    eps = 1e-12
    if not isinstance(X[0], list):
        X = [X]
    results = []
    for row in X:
        logits = [sum(W[k][j] * row[j] for j in range(d)) + b[k] for k in range(K)]
        results.append(classes[logits.index(max(logits))])
    return results


def logreg_multi_proba(modelo, X):
    """Retorna probabilidades por clase para cada fila de X."""
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'logreg_multi':
        raise TrezRuntimeError("logreg_multi_proba: modelo inválido.")
    from lib.mathdoz.core_mathdoz import exp_doz
    W, b = modelo['W'], modelo['b']
    K = len(W)
    d = len(W[0])
    eps = 1e-12
    if not isinstance(X[0], list):
        X = [X]
    results = []
    for row in X:
        logits = [sum(W[k][j] * row[j] for j in range(d)) + b[k] for k in range(K)]
        mx = max(logits)
        exps = [exp_doz(v - mx) for v in logits]
        s = sum(exps) + eps
        results.append([e / s for e in exps])
    return results


# ── Feature importance (árboles) ─────────────────────────────────────────────

def feature_importance(tree_model, n_features):
    """
    Extrae importancia de features de un árbol (tree_clf_fit / tree_reg_fit).
    Retorna lista de n_features scores normalizados a suma 1.
    """
    if not isinstance(tree_model, dict):
        raise TrezRuntimeError("feature_importance: se esperaba modelo de árbol.")
    scores = [0.0] * int(n_features)

    def _traverse(node):
        if node is None or not isinstance(node, dict):
            return
        feat = node.get('feature')
        gain = node.get('gain', 0.0)
        if feat is not None and 0 <= int(feat) < len(scores):
            scores[int(feat)] += abs(float(gain))
        _traverse(node.get('left'))
        _traverse(node.get('right'))

    root = tree_model.get('tree') or tree_model
    _traverse(root)
    total = sum(scores) + 1e-12
    return [s / total for s in scores]


# ── Curva de validación ───────────────────────────────────────────────────────

def validation_curve(fit_fn, predict_fn, X_train, y_train, X_val, y_val,
                     param_values, metric_fn=None):
    """
    Entrena con distintos valores de un hiperparámetro y mide métrica en val.
    fit_fn(param) → modelo
    predict_fn(modelo, X) → predicciones
    metric_fn(y_true, y_pred) → escalar (default: accuracy)
    Retorna [train_scores, val_scores].
    """
    if metric_fn is None:
        def metric_fn(yt, yp):
            return sum(a == b for a, b in zip(yt, yp)) / len(yt)

    train_scores, val_scores = [], []
    for pv in param_values:
        modelo = fit_fn(pv)
        yp_train = predict_fn(modelo, X_train)
        yp_val   = predict_fn(modelo, X_val)
        train_scores.append(metric_fn(y_train, yp_train))
        val_scores.append(metric_fn(y_val, yp_val))
    return [train_scores, val_scores]
