import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.randomdoz.randomdoz import uniform as _rand_uniform, random as _rand
from lib.mathdoz.core_mathdoz import exp_doz, sqrt_doz, log_doz
from errors import TrezRuntimeError


# ------------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------------

def _zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]


def _matmul(a, b):
    """a: (m,n)  b: (n,p) → (m,p)."""
    m, n = len(a), len(a[0])
    if isinstance(b[0], list):
        p = len(b[0])
        return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
    else:
        return [sum(a[i][k] * b[k] for k in range(n)) for i in range(m)]


def _transpose(m):
    if not isinstance(m[0], list):
        return [[v] for v in m]
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def _add_broadcast(mat, bias_row):
    """Suma bias_row (1 x cols) a cada fila de mat (batch x cols)."""
    return [[mat[i][j] + bias_row[0][j] for j in range(len(mat[i]))] for i in range(len(mat))]


def _scale(data, scalar):
    if not isinstance(data, list):
        return data * scalar
    return [_scale(x, scalar) for x in data]


def _add_elem(a, b):
    if not isinstance(a, list):
        return a + b
    return [_add_elem(ai, bi) for ai, bi in zip(a, b)]


def _sum_cols(mat):
    """Suma a lo largo de las filas → vector de longitud cols (para grad del bias)."""
    cols = len(mat[0])
    return [[sum(mat[i][j] for i in range(len(mat))) for j in range(cols)]]


# ------------------------------------------------------------------
# Capas
# ------------------------------------------------------------------

def linear_init(in_features, out_features):
    """
    Crea e inicializa una capa Linear con Xavier/Glorot.
    Retorna un dict: {type: 'linear', W: [[...]], b: [[...]], in: int, out: int}
    """
    limit = sqrt_doz(6.0 / (in_features + out_features))
    W = [[_rand_uniform(-limit, limit) for _ in range(out_features)]
         for _ in range(in_features)]
    b = [[_rand_uniform(-limit, limit) for _ in range(out_features)]]
    return {'type': 'linear', 'W': W, 'b': b, 'in': in_features, 'out': out_features}


def linear_forward(layer, x):
    """
    Forward de una capa linear.
    x:    (batch, in_features)
    layer: dict de linear_init
    Retorna (batch, out_features).
    """
    if not isinstance(x, list) or not isinstance(x[0], list):
        raise TrezRuntimeError("NNdoz.linear_forward: x debe ser lista de listas (batch x features).")
    out = _matmul(x, layer['W'])
    return _add_broadcast(out, layer['b'])


def linear_backward(layer, x, grad_out):
    """
    Backward de linear. Retorna dict con grad_W, grad_b, grad_x.
    grad_out: (batch, out_features)
    """
    grad_W = _matmul(_transpose(x), grad_out)           # (in, out)
    grad_b = _sum_cols(grad_out)                          # (1, out)
    grad_x = _matmul(grad_out, _transpose(layer['W']))   # (batch, in)
    return {'grad_W': grad_W, 'grad_b': grad_b, 'grad_x': grad_x}


def relu_forward(x):
    """ReLU elemento a elemento. Retorna (out, mask) para el backward."""
    def _relu(d):
        if not isinstance(d, list):
            return max(0.0, d)
        return [_relu(v) for v in d]

    def _mask(d):
        if not isinstance(d, list):
            return 1.0 if d > 0 else 0.0
        return [_mask(v) for v in d]

    out = _relu(x)
    mask = _mask(x)
    return out, mask


def relu_backward(mask, grad_out):
    """Backward de ReLU usando la máscara guardada en forward."""
    def _apply(m, g):
        if not isinstance(m, list):
            return m * g
        return [_apply(mi, gi) for mi, gi in zip(m, g)]
    return _apply(mask, grad_out)


def softmax_forward(x):
    """
    Softmax numericamente estable por fila.
    x: (batch, n_clases)
    Retorna (out, out) — guarda la salida para backward.
    Nota: el backward de softmax standalone es el jacobiano completo;
    en la práctica se combina con cross_entropy y se usa cross_entropy_grad.
    """
    if not isinstance(x, list) or not isinstance(x[0], list):
        raise TrezRuntimeError("NNdoz.softmax_forward: x debe ser (batch x clases).")
    out = []
    for row in x:
        max_val = max(row)
        exp_vals = [exp_doz(v - max_val) for v in row]
        s = sum(exp_vals)
        out.append([e / s for e in exp_vals])
    return out


def sequential_forward(layers, x):
    """
    Pasa x por una lista de capas en orden.
    Cada capa es un dict con 'type' y los parámetros correspondientes.
    Retorna (output, cache) donde cache es la lista de activaciones intermedias.
    """
    cache = [x]
    current = x
    for layer in layers:
        t = layer.get('type')
        if t == 'linear':
            current = linear_forward(layer, current)
        elif t == 'relu':
            current, layer['_mask'] = relu_forward(current)
        elif t == 'softmax':
            current = softmax_forward(current)
        else:
            raise TrezRuntimeError(f"NNdoz.sequential_forward: tipo de capa desconocido '{t}'.")
        cache.append(current)
    return current, cache


def get_params(layers):
    """Extrae lista plana de [W, b, W, b, ...] de todas las capas linear."""
    params = []
    for layer in layers:
        if layer.get('type') == 'linear':
            params.append(layer['W'])
            params.append(layer['b'])
    return params


def get_param_count(layers):
    """Número total de parámetros entrenables."""
    total = 0
    for layer in layers:
        if layer.get('type') == 'linear':
            W = layer['W']
            total += len(W) * len(W[0])
            total += len(layer['b'][0])
    return total


# ------------------------------------------------------------------
# Perceptrón de Rosenblatt
# ------------------------------------------------------------------

def perceptron_fit(X, y, lr=0.1, epochs=100):
    """
    Clasificador lineal binario. y debe contener exactamente 2 clases distintas.
    Las etiquetas se convierten internamente a {-1, +1}.
    """
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("perceptron_fit: X debe ser lista no vacía.")
    if not isinstance(y, list) or len(y) != len(X):
        raise TrezRuntimeError("perceptron_fit: y debe tener el mismo largo que X.")
    if not isinstance(X[0], list):
        X = [[v] for v in X]

    clases = sorted(set(y))
    if len(clases) != 2:
        raise TrezRuntimeError(f"perceptron_fit: requiere 2 clases, encontradas {len(clases)}.")
    label_map = {clases[0]: -1, clases[1]: 1}
    inv_map   = {-1: clases[0], 1: clases[1]}
    y_pm = [label_map[v] for v in y]

    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    for _ in range(int(epochs)):
        for xi, yi in zip(X, y_pm):
            if yi * (sum(w[j] * xi[j] for j in range(d)) + b) <= 0:
                for j in range(d):
                    w[j] += lr * yi * xi[j]
                b += lr * yi

    return {'tipo': 'perceptron', 'w': w, 'b': b, 'inv_map': inv_map}


def perceptron_predict(modelo, X):
    if not isinstance(modelo, dict) or modelo.get('tipo') != 'perceptron':
        raise TrezRuntimeError("perceptron_predict: modelo inválido.")
    w, b = modelo['w'], modelo['b']
    inv = modelo['inv_map']
    d = len(w)

    def _pred(x):
        score = sum(w[j] * x[j] for j in range(d)) + b
        return inv[1 if score >= 0 else -1]

    if isinstance(X, list) and X and isinstance(X[0], list):
        return [_pred(row) for row in X]
    if isinstance(X, list):
        return _pred(X)
    raise TrezRuntimeError("perceptron_predict: X debe ser lista.")


# ------------------------------------------------------------------
# MLP entrenable (forward + backward + SGD)
# ------------------------------------------------------------------

def mlp_init(arch):
    """
    arch: [n_entrada, h1, h2, ..., n_salida]
    Construye lista de capas: Linear → ReLU → ... → Linear (sin activación final).
    La activación de la última capa (relu/softmax/linear) se decide en mlp_train.
    """
    if len(arch) < 2:
        raise TrezRuntimeError("mlp_init: arch necesita al menos [in, out].")
    layers = []
    for i in range(len(arch) - 1):
        layers.append(linear_init(int(arch[i]), int(arch[i + 1])))
        if i < len(arch) - 2:
            layers.append({'type': 'relu'})
    return layers


def _mse_loss(yp, y):
    n = len(yp)
    if isinstance(yp[0], list) and isinstance(y[0], list):
        k = len(yp[0])
        total = sum((yp[i][j] - y[i][j]) ** 2 for i in range(n) for j in range(k))
        return total / (n * k)
    if isinstance(yp[0], list):
        total = sum((yp[i][0] - y[i]) ** 2 for i in range(n))
    else:
        total = sum((yp[i] - y[i]) ** 2 for i in range(n))
    return total / n


def _mse_grad(yp, y):
    n = len(yp)
    if isinstance(yp[0], list) and isinstance(y[0], list):
        k = len(yp[0])
        scale = 2.0 / (n * k)
        return [[(yp[i][j] - y[i][j]) * scale for j in range(k)] for i in range(n)]
    if isinstance(yp[0], list):
        return [[(yp[i][0] - y[i]) * 2.0 / n] for i in range(n)]
    return [[(yp[i] - y[i]) * 2.0 / n] for i in range(n)]


def _ce_loss(yp, y):
    """Cross-entropy. yp: (n, k) softmax output, y: (n,) int labels."""
    eps = 1e-12
    n = len(yp)
    total = 0.0
    for i in range(n):
        p = yp[i][int(y[i])]
        total += -log_doz(p if p > eps else eps)
    return total / n


def _ce_grad(yp, y):
    """Grad de CE+softmax combinado: yp − one_hot(y)."""
    n = len(yp)
    k = len(yp[0])
    g = []
    for i in range(n):
        row = [yp[i][j] for j in range(k)]
        row[int(y[i])] -= 1.0
        g.append([v / n for v in row])
    return g


def _sgd_update_layers(layers, grads_map, lr):
    """Aplica SGD a las capas linear usando el dict grads_map[layer_idx]."""
    for idx, layer in enumerate(layers):
        if layer.get('type') != 'linear':
            continue
        gW = grads_map[idx]['grad_W']
        gb = grads_map[idx]['grad_b']
        in_f = layer['in']
        out_f = layer['out']
        layer['W'] = [[layer['W'][r][c] - lr * gW[r][c] for c in range(out_f)] for r in range(in_f)]
        layer['b'] = [[layer['b'][0][c] - lr * gb[0][c] for c in range(out_f)]]


def mlp_train(layers, X, y, lr=0.01, epochs=1000, loss='mse'):
    """
    Entrena el MLP in-place. Retorna {'layers': layers, 'historia': [loss_por_epoch]}.
    loss: 'mse' | 'cross_entropy'
    X: lista de listas (n, d). y: lista de números.
    """
    if not isinstance(X, list) or not X or not isinstance(X[0], list):
        raise TrezRuntimeError("mlp_train: X debe ser matriz (lista de listas).")

    historia = []
    use_softmax = (loss == 'cross_entropy')

    for ep in range(int(epochs)):
        # Forward
        out, cache = sequential_forward(layers, X)

        if use_softmax:
            out = softmax_forward(out)

        # Loss + grad inicial
        if loss == 'mse':
            ep_loss = _mse_loss(out, y)
            grad = _mse_grad(out, y)
        elif loss == 'cross_entropy':
            ep_loss = _ce_loss(out, y)
            grad = _ce_grad(out, y)
        else:
            raise TrezRuntimeError(f"mlp_train: loss '{loss}' no soportada (usa 'mse' o 'cross_entropy').")

        historia.append(ep_loss)

        # Backward: recorrer capas en orden inverso
        grads_map = {}
        current_grad = grad

        for idx in range(len(layers) - 1, -1, -1):
            layer = layers[idx]
            t = layer.get('type')
            x_in = cache[idx]

            if t == 'linear':
                bwd = linear_backward(layer, x_in, current_grad)
                grads_map[idx] = bwd
                current_grad = bwd['grad_x']
            elif t == 'relu':
                current_grad = relu_backward(layer['_mask'], current_grad)
            # softmax ya fusionado en grad CE

        _sgd_update_layers(layers, grads_map, lr)

    return {'layers': layers, 'historia': historia}


def mlp_predict(layers, X):
    """Forward sin gradientes. Retorna salida de la última capa."""
    if not isinstance(X, list) or not X:
        raise TrezRuntimeError("mlp_predict: X debe ser lista no vacía.")
    if not isinstance(X[0], list):
        X = [X]
    out, _ = sequential_forward(layers, X)
    return out


# ------------------------------------------------------------------
# Activaciones adicionales
# ------------------------------------------------------------------

def sigmoid_forward(x):
    """Sigmoid elemento a elemento. Retorna (out, out) — out se usa en backward."""
    def _sig(v):
        if not isinstance(v, list):
            return 1.0/(1.0+exp_doz(-v)) if v >= 0 else (e := exp_doz(v), e/(1.0+e))[1]
        return [_sig(vi) for vi in v]
    out = _sig(x)
    return out, out  # cache = salida (suficiente para grad)

def sigmoid_backward(out_cache, grad_out):
    """Backward de sigmoid: grad * s*(1-s)."""
    def _grad(s, g):
        if not isinstance(s, list):
            return g * s * (1.0 - s)
        return [_grad(si, gi) for si, gi in zip(s, g)]
    return _grad(out_cache, grad_out)

def tanh_forward(x):
    """Tanh elemento a elemento."""
    def _tanh(v):
        if not isinstance(v, list):
            if v > 20: return 1.0
            if v < -20: return -1.0
            e = exp_doz(2*v); return (e-1)/(e+1)
        return [_tanh(vi) for vi in v]
    out = _tanh(x)
    return out, out

def tanh_backward(out_cache, grad_out):
    def _grad(t, g):
        if not isinstance(t, list):
            return g * (1.0 - t*t)
        return [_grad(ti, gi) for ti, gi in zip(t, g)]
    return _grad(out_cache, grad_out)


# ------------------------------------------------------------------
# BatchNorm (entrenamiento + inferencia)
# ------------------------------------------------------------------

def batchnorm_init(num_features, eps=1e-5, momentum=0.1):
    """
    Crea capa BatchNorm 1D.
    gamma=1, beta=0 iniciales. running_mean y running_var para inferencia.
    """
    return {
        'type': 'batchnorm',
        'gamma': [1.0]*num_features,
        'beta':  [0.0]*num_features,
        'running_mean': [0.0]*num_features,
        'running_var':  [1.0]*num_features,
        'eps': eps, 'momentum': momentum,
        'num_features': num_features,
        '_cache': None,
    }

def batchnorm_forward(layer, x, training=True):
    """
    x: (batch, num_features).
    Retorna x normalizado escalado por gamma+beta.
    """
    n = len(x); d = layer['num_features']
    eps = layer['eps']; mom = layer['momentum']
    gamma = layer['gamma']; beta = layer['beta']
    if training:
        mean = [sum(x[i][j] for i in range(n))/n for j in range(d)]
        var  = [sum((x[i][j]-mean[j])**2 for i in range(n))/n for j in range(d)]
        x_hat = [[(x[i][j]-mean[j])/sqrt_doz(var[j]+eps) for j in range(d)] for i in range(n)]
        out   = [[gamma[j]*x_hat[i][j]+beta[j] for j in range(d)] for i in range(n)]
        # actualizar running stats
        layer['running_mean'] = [(1-mom)*layer['running_mean'][j]+mom*mean[j] for j in range(d)]
        layer['running_var']  = [(1-mom)*layer['running_var'][j] +mom*var[j]  for j in range(d)]
        layer['_cache'] = (x, x_hat, mean, var)
    else:
        rm = layer['running_mean']; rv = layer['running_var']
        x_hat = [[(x[i][j]-rm[j])/sqrt_doz(rv[j]+eps) for j in range(d)] for i in range(n)]
        out   = [[gamma[j]*x_hat[i][j]+beta[j] for j in range(d)] for i in range(n)]
    return out

def batchnorm_backward(layer, grad_out):
    """Backward de BatchNorm. Retorna grad_x, grad_gamma, grad_beta."""
    x, x_hat, mean, var = layer['_cache']
    n = len(x); d = layer['num_features']; eps = layer['eps']
    gamma = layer['gamma']
    grad_gamma = [sum(grad_out[i][j]*x_hat[i][j] for i in range(n)) for j in range(d)]
    grad_beta  = [sum(grad_out[i][j] for i in range(n)) for j in range(d)]
    # dx_hat
    dx_hat = [[grad_out[i][j]*gamma[j] for j in range(d)] for i in range(n)]
    std_inv = [1.0/sqrt_doz(var[j]+eps) for j in range(d)]
    grad_x = [[0.0]*d for _ in range(n)]
    for j in range(d):
        dvar = sum(dx_hat[i][j]*(x[i][j]-mean[j])*(-0.5)*(var[j]+eps)**(-1.5) for i in range(n))
        dmean = sum(-dx_hat[i][j]*std_inv[j] for i in range(n)) + dvar*sum(-2*(x[i][j]-mean[j]) for i in range(n))/n
        for i in range(n):
            grad_x[i][j] = dx_hat[i][j]*std_inv[j] + dvar*2*(x[i][j]-mean[j])/n + dmean/n
    return grad_x, grad_gamma, grad_beta


# ------------------------------------------------------------------
# Dropout
# ------------------------------------------------------------------

def dropout_forward(x, p=0.5, training=True):
    """
    Dropout inverted. p = probabilidad de apagar.
    Retorna (out, mask). En inferencia mask=None y out=x.
    """
    if not training or p == 0.0:
        return x, None
    scale = 1.0/(1.0-p)
    def _drop(v):
        if not isinstance(v, list):
            return (v*scale if _rand() > p else 0.0)
        return [_drop(vi) for vi in v]
    def _mask(v):
        if not isinstance(v, list):
            return (scale if _rand() > p else 0.0)
        return [_mask(vi) for vi in v]
    # Generar máscara primero, aplicar después
    from lib.randomdoz.randomdoz import random as rng
    def _gen_mask(v):
        if not isinstance(v, list):
            return scale if rng() > p else 0.0
        return [_gen_mask(vi) for vi in v]
    def _apply_mask(v, m):
        if not isinstance(v, list):
            return v * m
        return [_apply_mask(vi, mi) for vi, mi in zip(v, m)]
    mask = _gen_mask(x)
    out  = _apply_mask(x, mask)
    return out, mask

def dropout_backward(mask, grad_out):
    """Backward de dropout: grad * mask."""
    if mask is None:
        return grad_out
    def _apply(m, g):
        if not isinstance(m, list):
            return m * g
        return [_apply(mi, gi) for mi, gi in zip(m, g)]
    return _apply(mask, grad_out)


# ------------------------------------------------------------------
# Pesos compartidos (shared weights) para redes siamesas
# ------------------------------------------------------------------

def shared_forward(layer, x_a, x_b):
    """
    Pasa dos entradas por la MISMA capa (mismos pesos).
    Retorna (out_a, out_b).
    """
    return linear_forward(layer, x_a), linear_forward(layer, x_b)

def shared_backward(layer, x_a, x_b, grad_a, grad_b):
    """
    Backward de rama compartida: acumula gradientes de ambas ramas.
    Retorna dict con grad_W, grad_b, grad_xa, grad_xb.
    """
    bwd_a = linear_backward(layer, x_a, grad_a)
    bwd_b = linear_backward(layer, x_b, grad_b)
    d = layer['in']; out_f = layer['out']
    # Acumular grad_W y grad_b
    grad_W = [[bwd_a['grad_W'][r][c]+bwd_b['grad_W'][r][c] for c in range(out_f)] for r in range(d)]
    grad_b_acc = [[bwd_a['grad_b'][0][c]+bwd_b['grad_b'][0][c] for c in range(out_f)]]
    return {
        'grad_W': grad_W, 'grad_b': grad_b_acc,
        'grad_xa': bwd_a['grad_x'], 'grad_xb': bwd_b['grad_x'],
    }


# ------------------------------------------------------------------
# Embedding layer (para representaciones discretas)
# ------------------------------------------------------------------

def embedding_init(vocab_size, embed_dim):
    """Tabla de embeddings aleatorios: vocab_size × embed_dim."""
    limit = sqrt_doz(1.0/embed_dim)
    return {
        'type': 'embedding',
        'W': [[_rand_uniform(-limit, limit) for _ in range(embed_dim)] for _ in range(vocab_size)],
        'vocab_size': vocab_size, 'embed_dim': embed_dim,
    }

def embedding_forward(layer, indices):
    """indices: lista de enteros. Retorna lista de vectores embed_dim."""
    W = layer['W']
    return [list(W[int(i)]) for i in indices]

def embedding_backward(layer, indices, grad_out, lr):
    """Actualiza embeddings por SGD directamente (sparse update)."""
    for k, idx in enumerate(indices):
        idx = int(idx)
        for j in range(layer['embed_dim']):
            layer['W'][idx][j] -= lr * grad_out[k][j]


# ------------------------------------------------------------------
# L2 distance (útil para siamesas)
# ------------------------------------------------------------------

def l2_distance(emb_a, emb_b):
    """
    Distancia euclidiana entre dos batches de embeddings.
    emb_a, emb_b: (batch, dim) o vectores 1D.
    Retorna lista de distancias escalares.
    """
    if isinstance(emb_a[0], (int, float)):
        emb_a = [emb_a]; emb_b = [emb_b]
    return [sqrt_doz(sum((emb_a[i][j]-emb_b[i][j])**2 for j in range(len(emb_a[i]))))
            for i in range(len(emb_a))]

def cosine_similarity(emb_a, emb_b):
    """Similitud coseno entre dos batches de embeddings. Retorna lista en [-1, 1]."""
    if isinstance(emb_a[0], (int, float)):
        emb_a = [emb_a]; emb_b = [emb_b]
    result = []
    for a, b in zip(emb_a, emb_b):
        dot = sum(ai*bi for ai,bi in zip(a,b))
        na  = sqrt_doz(sum(ai*ai for ai in a)) + 1e-12
        nb  = sqrt_doz(sum(bi*bi for bi in b)) + 1e-12
        result.append(dot/(na*nb))
    return result
