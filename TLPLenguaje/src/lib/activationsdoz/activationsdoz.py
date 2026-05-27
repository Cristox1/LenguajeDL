def exp_doz(x, terms=50):
    if x < 0:
        return 1.0 / exp_doz(-x, terms)
    res = 1.0
    term = 1.0
    for i in range(1, terms):
        term *= (x / i)
        res += term
        if term < 1e-10:
            break
    return res

def relu(x):
    if isinstance(x, (int, float)):
        return max(0, x)
    if isinstance(x, list):
        return [relu(i) for i in x]


def relu_deriv(x):
    if isinstance(x, (int, float)):
        return 1.0 if x > 0 else 0.0
    if isinstance(x, list):
        return [relu_deriv(i) for i in x]


def sigmoid(x):
    if isinstance(x, (int, float)):
        return 1 / (1 + exp_doz(-x))
    if isinstance(x, list):
        return [sigmoid(i) for i in x]


def sigmoid_deriv(x):
    if isinstance(x, (int, float)):
        s = sigmoid(x)
        return s * (1 - s)
    if isinstance(x, list):
        return [sigmoid_deriv(i) for i in x]


def tanh(x):
    if isinstance(x, (int, float)):
        ep = exp_doz(x)
        en = exp_doz(-x)
        return (ep - en) / (ep + en)
    if isinstance(x, list):
        return [tanh(i) for i in x]


def tanh_deriv(x):
    if isinstance(x, (int, float)):
        t = tanh(x)
        return 1 - t * t
    if isinstance(x, list):
        return [tanh_deriv(i) for i in x]


def linear(x):
    """Identidad — útil como activación en regresión."""
    return x


def softmax(v):
    """Softmax numéricamente estable. Si v es matriz, opera por fila."""
    if not isinstance(v, list) or not v:
        return v
    if isinstance(v[0], list):
        return [softmax(row) for row in v]
    m = max(v)
    exps = [exp_doz(x - m) for x in v]
    s = sum(exps)
    return [e / s for e in exps]
