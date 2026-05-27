"""
sklearndoz.neural_network — MLPClassifier/Regressor, Python puro.

Envuelve la lógica de nndoz con interfaz sklearn-compatible.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz, log_doz
from lib.randomdoz.randomdoz import gauss as _gauss, random as _random
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0
def _dot(a, b): return sum(ai*bi for ai,bi in zip(a,b))

def _relu(x): return max(0.0, x)
def _relu_d(x): return 1.0 if x > 0 else 0.0
def _tanh(x):
    if x > 20: return 1.0
    if x < -20: return -1.0
    ep = exp_doz(2*x); return (ep-1)/(ep+1)
def _tanh_d(x): return 1 - _tanh(x)**2
def _logistic(x):
    if x >= 0: return 1.0/(1.0+exp_doz(-x))
    ex = exp_doz(x); return ex/(1.0+ex)
def _logistic_d(x): s = _logistic(x); return s*(1-s)
def _identity(x): return x
def _identity_d(x): return 1.0

_ACTS = {
    'relu': (_relu, _relu_d),
    'tanh': (_tanh, _tanh_d),
    'logistic': (_logistic, _logistic_d),
    'identity': (_identity, _identity_d),
}

def _softmax(v):
    mx = max(v)
    exps = [exp_doz(min(vi-mx, 700)) for vi in v]
    total = sum(exps) or 1e-10
    return [e/total for e in exps]


def _init_weights(n_in, n_out):
    scale = sqrt_doz(2.0/n_in)
    W = [[_gauss(0, scale) for _ in range(n_out)] for _ in range(n_in)]
    b = [0.0]*n_out
    return W, b

def _forward(X, layers, act_fn):
    activations = [X]
    pre_acts = []
    for (W, b) in layers:
        prev = activations[-1]
        n_out = len(b)
        z = [_dot([prev[j] for j in range(len(prev))], [W[j][k] for j in range(len(prev))])+b[k] for k in range(n_out)]
        pre_acts.append(z)
        a = [act_fn(zi) for zi in z]
        activations.append(a)
    return activations, pre_acts

def _forward_last_linear(X, layers, act_fn):
    """Forward pass, última capa sin activación."""
    activations = [X]
    pre_acts = []
    for i, (W, b) in enumerate(layers):
        prev = activations[-1]
        n_out = len(b)
        z = [_dot([prev[j] for j in range(len(prev))], [W[j][k] for j in range(len(prev))])+b[k] for k in range(n_out)]
        pre_acts.append(z)
        if i < len(layers)-1:
            a = [act_fn(zi) for zi in z]
        else:
            a = list(z)
        activations.append(a)
    return activations, pre_acts


def _backward(activations, pre_acts, layers, y_onehot, act_d_fn, lr, alpha):
    n_layers = len(layers)
    # Output delta
    out = activations[-1]
    delta = [out[k]-y_onehot[k] for k in range(len(out))]
    grads = []
    for i in range(n_layers-1, -1, -1):
        W, b = layers[i]
        prev = activations[i]
        n_in = len(prev); n_out = len(b)
        dW = [[delta[k]*prev[j] for k in range(n_out)] for j in range(n_in)]
        db = list(delta)
        grads.insert(0, (dW, db))
        if i > 0:
            d_fn = act_d_fn
            delta = [sum(W[j][k]*delta[k] for k in range(n_out))*d_fn(pre_acts[i-1][j])
                     for j in range(n_in)]
    new_layers = []
    for i, (W, b) in enumerate(layers):
        dW, db = grads[i]
        n_in, n_out = len(W), len(b)
        W2 = [[W[j][k]-lr*(dW[j][k]+alpha*W[j][k]) for k in range(n_out)] for j in range(n_in)]
        b2 = [b[k]-lr*db[k] for k in range(n_out)]
        new_layers.append((W2, b2))
    return new_layers


# ── MLPClassifier ─────────────────────────────────────────────────────────────

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                 alpha=1e-4, learning_rate_init=0.001, max_iter=200, tol=1e-4,
                 random_state=None, batch_size=32, momentum=0.0):
        self.hidden_layer_sizes=hidden_layer_sizes if isinstance(hidden_layer_sizes, tuple) else (hidden_layer_sizes,)
        self.activation=activation; self.solver=solver; self.alpha=alpha
        self.learning_rate_init=learning_rate_init; self.max_iter=int(max_iter)
        self.tol=tol; self.batch_size=int(batch_size); self.momentum=momentum
        self.coefs_=None; self.intercepts_=None
        self.classes_=None; self.n_iter_=0; self.loss_=None

    def _build_layers(self, n_in, n_out):
        sizes = [n_in] + list(self.hidden_layer_sizes) + [n_out]
        layers = []
        for i in range(len(sizes)-1):
            W, b = _init_weights(sizes[i], sizes[i+1])
            layers.append((W, b))
        return layers

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); K = len(self.classes_)
        d = len(X[0]); n = len(X)
        act_fn, act_d = _ACTS.get(self.activation, _ACTS['relu'])
        layers = self._build_layers(d, K)
        cls_idx = {c:i for i,c in enumerate(self.classes_)}
        lr = self.learning_rate_init
        prev_loss = float('inf')
        for it in range(self.max_iter):
            # Shuffle
            idx = list(range(n))
            for i in range(n-1, 0, -1):
                j = int(_random()*(i+1)); idx[i], idx[j] = idx[j], idx[i]
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start:start+self.batch_size]
                for bi in batch:
                    x = X[bi]
                    y_oh = [0.0]*K; y_oh[cls_idx[y[bi]]] = 1.0
                    activations, pre_acts = _forward(x, layers, act_fn)
                    out = _softmax(activations[-1])
                    loss = -sum(y_oh[k]*log_doz(out[k]+1e-10) for k in range(K))
                    total_loss += loss
                    delta = [out[k]-y_oh[k] for k in range(K)]
                    # Backward manual
                    for i in range(len(layers)-1, -1, -1):
                        W, b = layers[i]
                        prev = activations[i]; n_in = len(prev); n_out = len(b)
                        dW = [[delta[k]*prev[j]/len(batch) for k in range(n_out)] for j in range(n_in)]
                        db = [delta[k]/len(batch) for k in range(n_out)]
                        W2 = [[W[j][k]-lr*(dW[j][k]+self.alpha*W[j][k]) for k in range(n_out)] for j in range(n_in)]
                        b2 = [b[k]-lr*db[k] for k in range(n_out)]
                        if i > 0:
                            delta = [sum(W[j][k]*delta[k] for k in range(n_out))*act_d(pre_acts[i-1][j])
                                     for j in range(n_in)]
                        layers[i] = (W2, b2)
            self.loss_ = total_loss/n
            self.n_iter_ = it+1
            if abs_doz(prev_loss-self.loss_) < self.tol: break
            prev_loss = self.loss_
        self.coefs_ = [W for W,b in layers]
        self.intercepts_ = [b for W,b in layers]
        self._layers = layers
        self._act_fn = act_fn
        return self

    def predict_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        act_fn = self._act_fn
        result = []
        for x in X:
            activations, _ = _forward(x, self._layers, act_fn)
            result.append(_softmax(activations[-1]))
        return result

    def predict(self, X):
        probs = self.predict_proba(X)
        return [self.classes_[p.index(max(p))] for p in probs]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── MLPRegressor ──────────────────────────────────────────────────────────────

class MLPRegressor:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='sgd',
                 alpha=1e-4, learning_rate_init=0.001, max_iter=200, tol=1e-4,
                 batch_size=32):
        self.hidden_layer_sizes=hidden_layer_sizes if isinstance(hidden_layer_sizes, tuple) else (hidden_layer_sizes,)
        self.activation=activation; self.solver=solver; self.alpha=alpha
        self.learning_rate_init=learning_rate_init; self.max_iter=int(max_iter)
        self.tol=tol; self.batch_size=int(batch_size)
        self.coefs_=None; self.intercepts_=None
        self.n_iter_=0; self.loss_=None

    def _build_layers(self, n_in, n_out):
        sizes = [n_in] + list(self.hidden_layer_sizes) + [n_out]
        layers = []
        for i in range(len(sizes)-1):
            W, b = _init_weights(sizes[i], sizes[i+1])
            layers.append((W, b))
        return layers

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        y_list = list(y)
        multi = isinstance(y_list[0], list)
        n_out = len(y_list[0]) if multi else 1
        d = len(X[0]); n = len(X)
        act_fn, act_d = _ACTS.get(self.activation, _ACTS['relu'])
        layers = self._build_layers(d, n_out)
        lr = self.learning_rate_init; prev_loss = float('inf')
        for it in range(self.max_iter):
            idx = list(range(n))
            for i in range(n-1, 0, -1):
                j = int(_random()*(i+1)); idx[i], idx[j] = idx[j], idx[i]
            total_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start:start+self.batch_size]
                for bi in batch:
                    x = X[bi]
                    yt = y_list[bi] if multi else [y_list[bi]]
                    activations, pre_acts = _forward_last_linear(x, layers, act_fn)
                    out = activations[-1]
                    loss = sum((out[k]-yt[k])**2 for k in range(n_out))/n_out
                    total_loss += loss
                    delta = [(out[k]-yt[k])*2/n_out for k in range(n_out)]
                    for i in range(len(layers)-1, -1, -1):
                        W, b = layers[i]
                        prev = activations[i]; n_in = len(prev); n_out_l = len(b)
                        dW = [[delta[k]*prev[j]/len(batch) for k in range(n_out_l)] for j in range(n_in)]
                        db = [delta[k]/len(batch) for k in range(n_out_l)]
                        W2 = [[W[j][k]-lr*(dW[j][k]+self.alpha*W[j][k]) for k in range(n_out_l)] for j in range(n_in)]
                        b2 = [b[k]-lr*db[k] for k in range(n_out_l)]
                        if i > 0:
                            delta = [sum(W[j][k]*delta[k] for k in range(n_out_l))*act_d(pre_acts[i-1][j])
                                     for j in range(n_in)]
                        layers[i] = (W2, b2)
            self.loss_ = total_loss/n; self.n_iter_ = it+1
            if abs_doz(prev_loss-self.loss_) < self.tol: break
            prev_loss = self.loss_
        self.coefs_ = [W for W,b in layers]
        self.intercepts_ = [b for W,b in layers]
        self._layers = layers; self._act_fn = act_fn; self._multi = multi
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        act_fn = self._act_fn
        result = []
        for x in X:
            activations, _ = _forward_last_linear(x, self._layers, act_fn)
            out = activations[-1]
            result.append(out if self._multi else out[0])
        return result

    def score(self, X, y):
        preds = self.predict(X)
        y_list = list(y)
        if self._multi:
            ss_res = sum(sum((yi[k]-pi[k])**2 for k in range(len(yi))) for yi,pi in zip(y_list,preds))
            flat_y = [v for row in y_list for v in row]
            m = _mean(flat_y)
            ss_tot = sum((v-m)**2 for v in flat_y) or 1e-10
        else:
            ss_res = sum((yi-pi)**2 for yi,pi in zip(y_list,preds))
            m = _mean(y_list)
            ss_tot = sum((yi-m)**2 for yi in y_list) or 1e-10
        return 1-ss_res/ss_tot
