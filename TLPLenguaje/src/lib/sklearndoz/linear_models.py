"""
sklearndoz.linear_models — Modelos lineales, Python puro.

Cubre: LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
       RidgeClassifier, SGDClassifier, SGDRegressor, Perceptron,
       PassiveAggressiveClassifier, PassiveAggressiveRegressor,
       HuberRegressor, QuantileRegressor, BayesianRidge.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz, log_doz
from lib.randomdoz.randomdoz import seed as _seed, shuffle as _shuffle
from errors import TrezRuntimeError


# ── helpers internos ──────────────────────────────────────────────────────────

def _dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def _mat_dot(X, w):
    return [_dot(row, w) for row in X]

def _sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _clip(x, lo, hi):
    return max(lo, min(hi, x))

def _norm2(w):
    return sqrt_doz(sum(wi * wi for wi in w))

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + exp_doz(-z))
    ez = exp_doz(z)
    return ez / (1.0 + ez)

def _to2d(X):
    if X and not isinstance(X[0], list):
        return [[v] for v in X]
    return X

def _soft_threshold(x, lam):
    if x > lam:   return x - lam
    if x < -lam:  return x + lam
    return 0.0


# ── LinearRegression (OLS via gradient descent) ───────────────────────────────

class LinearRegression:
    """Regresión lineal ordinaria por gradiente descendente."""
    def __init__(self, lr=0.01, epochs=1000, fit_intercept=True):
        self.lr = lr; self.epochs = int(epochs); self.fit_intercept = fit_intercept
        self.coef_ = None; self.intercept_ = 0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = 0.0
        for _ in range(self.epochs):
            yp = [_dot(X[i], w) + (b if self.fit_intercept else 0) for i in range(n)]
            res = [yp[i]-y[i] for i in range(n)]
            gw = [(2/n)*sum(res[i]*X[i][j] for i in range(n)) for j in range(d)]
            gb = (2/n)*sum(res) if self.fit_intercept else 0
            w = [w[j]-self.lr*gw[j] for j in range(d)]
            b -= self.lr*gb
        self.coef_ = w; self.intercept_ = b; return self

    def predict(self, X):
        X = _to2d(X)
        return [_dot(row, self.coef_) + self.intercept_ for row in X]

    def score(self, X, y):
        yp = self.predict(X); n = len(y); m = _mean(y)
        ss_res = sum((y[i]-yp[i])**2 for i in range(n))
        ss_tot = sum((y[i]-m)**2 for i in range(n))
        return 1 - ss_res/ss_tot if ss_tot else 0.0


# ── Ridge (L2) ────────────────────────────────────────────────────────────────

class Ridge:
    def __init__(self, alpha=1.0, lr=0.01, epochs=1000, fit_intercept=True):
        self.alpha=alpha; self.lr=lr; self.epochs=int(epochs)
        self.fit_intercept=fit_intercept; self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = 0.0
        for _ in range(self.epochs):
            yp = [_dot(X[i],w)+b for i in range(n)]
            res = [yp[i]-y[i] for i in range(n)]
            gw = [(2/n)*sum(res[i]*X[i][j] for i in range(n))+2*self.alpha*w[j] for j in range(d)]
            gb = (2/n)*sum(res) if self.fit_intercept else 0
            w = [w[j]-self.lr*gw[j] for j in range(d)]; b -= self.lr*gb
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row, self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── Lasso (L1, coordinate descent) ───────────────────────────────────────────

class Lasso:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha=alpha; self.max_iter=int(max_iter); self.tol=tol
        self.fit_intercept=fit_intercept; self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = _mean(y) if self.fit_intercept else 0.0
        for _ in range(self.max_iter):
            w_old = list(w)
            res = [y[i]-(_dot(X[i],w)+b) for i in range(n)]
            if self.fit_intercept:
                b = b + _mean(res)
                res = [y[i]-(_dot(X[i],w)+b) for i in range(n)]
            for j in range(d):
                rj = [res[i]+X[i][j]*w[j] for i in range(n)]
                rho = sum(X[i][j]*rj[i] for i in range(n))/n
                xjsq = sum(X[i][j]**2 for i in range(n))/n
                w[j] = _soft_threshold(rho, self.alpha) / xjsq if xjsq else 0.0
                res = [rj[i]-X[i][j]*w[j] for i in range(n)]
            if max(abs_doz(w[j]-w_old[j]) for j in range(d)) < self.tol:
                break
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── ElasticNet (L1 + L2) ──────────────────────────────────────────────────────

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha=alpha; self.l1_ratio=l1_ratio; self.max_iter=int(max_iter)
        self.tol=tol; self.fit_intercept=fit_intercept
        self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        lam1 = self.alpha*self.l1_ratio; lam2 = self.alpha*(1-self.l1_ratio)
        w = [0.0]*d; b = _mean(y) if self.fit_intercept else 0.0
        for _ in range(self.max_iter):
            w_old = list(w)
            res = [y[i]-(_dot(X[i],w)+b) for i in range(n)]
            if self.fit_intercept: b += _mean(res); res=[y[i]-(_dot(X[i],w)+b) for i in range(n)]
            for j in range(d):
                rj = [res[i]+X[i][j]*w[j] for i in range(n)]
                rho = sum(X[i][j]*rj[i] for i in range(n))/n
                xjsq = sum(X[i][j]**2 for i in range(n))/n + lam2
                w[j] = _soft_threshold(rho, lam1) / xjsq if xjsq else 0.0
                res = [rj[i]-X[i][j]*w[j] for i in range(n)]
            if max(abs_doz(w[j]-w_old[j]) for j in range(d)) < self.tol: break
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── LogisticRegression ────────────────────────────────────────────────────────

class LogisticRegression:
    def __init__(self, C=1.0, lr=0.1, max_iter=1000, fit_intercept=True, tol=1e-4):
        self.C=C; self.lr=lr; self.max_iter=int(max_iter)
        self.fit_intercept=fit_intercept; self.tol=tol
        self.coef_=None; self.intercept_=0.0; self.classes_=None

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        self.classes_ = sorted(set(y))
        lam = 1.0/(self.C*n)
        if len(self.classes_) == 2:
            cm = {self.classes_[0]:0, self.classes_[1]:1}
            yb = [cm[v] for v in y]
            w = [0.0]*d; b = 0.0
            for _ in range(self.max_iter):
                yp = [_sigmoid(_dot(X[i],w)+b) for i in range(n)]
                err = [yp[i]-yb[i] for i in range(n)]
                gw = [(1/n)*sum(err[i]*X[i][j] for i in range(n))+lam*w[j] for j in range(d)]
                gb = (1/n)*sum(err) if self.fit_intercept else 0
                w_new = [w[j]-self.lr*gw[j] for j in range(d)]; b -= self.lr*gb
                if max(abs_doz(w_new[j]-w[j]) for j in range(d)) < self.tol:
                    w = w_new; break
                w = w_new
            self.coef_ = [w]; self.intercept_ = b
        else:
            # one-vs-rest
            K = len(self.classes_)
            all_w = []; all_b = []
            for k, cls in enumerate(self.classes_):
                yb = [1 if v==cls else 0 for v in y]
                w = [0.0]*d; b = 0.0
                for _ in range(self.max_iter):
                    yp = [_sigmoid(_dot(X[i],w)+b) for i in range(n)]
                    err = [yp[i]-yb[i] for i in range(n)]
                    gw = [(1/n)*sum(err[i]*X[i][j] for i in range(n))+lam*w[j] for j in range(d)]
                    gb = (1/n)*sum(err)
                    w = [w[j]-self.lr*gw[j] for j in range(d)]; b -= self.lr*gb
                all_w.append(w); all_b.append(b)
            self.coef_ = all_w; self.intercept_ = all_b
        return self

    def predict_proba(self, X):
        X = _to2d(X)
        if len(self.classes_) == 2:
            p1 = [_sigmoid(_dot(row,self.coef_[0])+self.intercept_) for row in X]
            return [[1-p, p] for p in p1]
        scores = [[_sigmoid(_dot(row,self.coef_[k])+self.intercept_[k]) for k in range(len(self.classes_))] for row in X]
        return [[s/sum(row) for s in row] for row in scores]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        yp = self.predict(X)
        return sum(1 for a,b in zip(y,yp) if a==b)/len(y)


# ── RidgeClassifier ───────────────────────────────────────────────────────────

class RidgeClassifier:
    """Ridge regression aplicada a etiquetas ±1 para clasificación."""
    def __init__(self, alpha=1.0, lr=0.01, epochs=1000):
        self.alpha=alpha; self.lr=lr; self.epochs=int(epochs)
        self.coef_=None; self.intercept_=0.0; self.classes_=None

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        if len(self.classes_) != 2:
            raise TrezRuntimeError("RidgeClassifier: solo binario.")
        cm = {self.classes_[0]:-1.0, self.classes_[1]:1.0}
        yenc = [cm[v] for v in y]
        r = Ridge(self.alpha, self.lr, self.epochs)
        r.fit(X, yenc)
        self.coef_ = r.coef_; self.intercept_ = r.intercept_; return self

    def predict(self, X):
        scores = [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]
        return [self.classes_[1] if s>=0 else self.classes_[0] for s in scores]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


# ── SGDClassifier ─────────────────────────────────────────────────────────────

class SGDClassifier:
    """SGD con hinge loss (SVM) o log loss (logistic)."""
    def __init__(self, loss='hinge', alpha=1e-4, lr=0.01, max_iter=1000, shuffle=True):
        self.loss=loss; self.alpha=alpha; self.lr=lr
        self.max_iter=int(max_iter); self.shuffle=shuffle
        self.coef_=None; self.intercept_=0.0; self.classes_=None

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        self.classes_ = sorted(set(y))
        cm = {self.classes_[0]:-1.0, self.classes_[1]:1.0}
        yenc = [cm[v] for v in y]
        w = [0.0]*d; b = 0.0; idx = list(range(n))
        for _ in range(self.max_iter):
            if self.shuffle: _shuffle(idx)
            for i in idx:
                score = _dot(X[i],w)+b
                if self.loss=='hinge':
                    if yenc[i]*score < 1:
                        for j in range(d): w[j] -= self.lr*(self.alpha*w[j]-yenc[i]*X[i][j])
                        b += self.lr*yenc[i]
                    else:
                        for j in range(d): w[j] -= self.lr*self.alpha*w[j]
                else:  # log
                    p = _sigmoid(yenc[i]*score)
                    err = yenc[i]*(p-1)
                    for j in range(d): w[j] -= self.lr*(err*X[i][j]+self.alpha*w[j])
                    b -= self.lr*err
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        scores = [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]
        return [self.classes_[1] if s>=0 else self.classes_[0] for s in scores]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


# ── SGDRegressor ──────────────────────────────────────────────────────────────

class SGDRegressor:
    def __init__(self, loss='squared_error', alpha=1e-4, lr=0.01, max_iter=1000, shuffle=True):
        self.loss=loss; self.alpha=alpha; self.lr=lr
        self.max_iter=int(max_iter); self.shuffle=shuffle
        self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = 0.0; idx = list(range(n))
        for _ in range(self.max_iter):
            if self.shuffle: _shuffle(idx)
            for i in idx:
                pred = _dot(X[i],w)+b; err = pred-y[i]
                for j in range(d): w[j] -= self.lr*(err*X[i][j]+self.alpha*w[j])
                b -= self.lr*err
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── Perceptron ────────────────────────────────────────────────────────────────

class Perceptron:
    def __init__(self, lr=1.0, max_iter=1000, shuffle=True):
        self.lr=lr; self.max_iter=int(max_iter); self.shuffle=shuffle
        self.coef_=None; self.intercept_=0.0; self.classes_=None

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        self.classes_ = sorted(set(y))
        if len(self.classes_) != 2: raise TrezRuntimeError("Perceptron: solo binario.")
        cm = {self.classes_[0]:-1.0, self.classes_[1]:1.0}
        yenc = [cm[v] for v in y]; w = [0.0]*d; b = 0.0; idx = list(range(n))
        for _ in range(self.max_iter):
            if self.shuffle: _shuffle(idx)
            for i in idx:
                if yenc[i]*(_dot(X[i],w)+b) <= 0:
                    for j in range(d): w[j] += self.lr*yenc[i]*X[i][j]
                    b += self.lr*yenc[i]
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        scores = [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]
        return [self.classes_[1] if s>=0 else self.classes_[0] for s in scores]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


# ── PassiveAggressiveClassifier ───────────────────────────────────────────────

class PassiveAggressiveClassifier:
    def __init__(self, C=1.0, max_iter=1000, shuffle=True):
        self.C=C; self.max_iter=int(max_iter); self.shuffle=shuffle
        self.coef_=None; self.intercept_=0.0; self.classes_=None

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        self.classes_ = sorted(set(y))
        cm = {self.classes_[0]:-1.0, self.classes_[1]:1.0}
        yenc = [cm[v] for v in y]; w = [0.0]*d; b = 0.0; idx = list(range(n))
        for _ in range(self.max_iter):
            if self.shuffle: _shuffle(idx)
            for i in idx:
                score = yenc[i]*(_dot(X[i],w)+b)
                loss = max(0.0, 1-score)
                if loss > 0:
                    norm_sq = _dot(X[i],X[i])+1
                    tau = _clip(loss/norm_sq, 0, self.C)
                    for j in range(d): w[j] += tau*yenc[i]*X[i][j]
                    b += tau*yenc[i]
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        scores = [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]
        return [self.classes_[1] if s>=0 else self.classes_[0] for s in scores]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


# ── PassiveAggressiveRegressor ────────────────────────────────────────────────

class PassiveAggressiveRegressor:
    def __init__(self, C=1.0, epsilon=0.1, max_iter=1000, shuffle=True):
        self.C=C; self.epsilon=epsilon; self.max_iter=int(max_iter); self.shuffle=shuffle
        self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = 0.0; idx = list(range(n))
        for _ in range(self.max_iter):
            if self.shuffle: _shuffle(idx)
            for i in idx:
                pred = _dot(X[i],w)+b; err = y[i]-pred
                loss = max(0.0, abs_doz(err)-self.epsilon)
                norm_sq = _dot(X[i],X[i])+1
                tau = _clip(loss/norm_sq, 0, self.C) * _sign(err)
                for j in range(d): w[j] += tau*X[i][j]
                b += tau
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── HuberRegressor (robust regression) ───────────────────────────────────────

class HuberRegressor:
    """Regresión robusta con pérdida Huber (menos sensible a outliers que MSE)."""
    def __init__(self, epsilon=1.35, alpha=1e-4, max_iter=1000, lr=0.01):
        self.epsilon=epsilon; self.alpha=alpha
        self.max_iter=int(max_iter); self.lr=lr
        self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = 0.0
        for _ in range(self.max_iter):
            yp = [_dot(X[i],w)+b for i in range(n)]
            gw = [0.0]*d; gb = 0.0
            for i in range(n):
                r = yp[i]-y[i]
                if abs_doz(r) <= self.epsilon:
                    dr = r
                else:
                    dr = self.epsilon * _sign(r)
                for j in range(d): gw[j] += dr*X[i][j]/n + self.alpha*w[j]
                gb += dr/n
            w = [w[j]-self.lr*gw[j] for j in range(d)]; b -= self.lr*gb
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── QuantileRegressor ─────────────────────────────────────────────────────────

class QuantileRegressor:
    """Regresión por cuantiles — minimiza pinball loss."""
    def __init__(self, quantile=0.5, alpha=1e-4, lr=0.01, max_iter=1000):
        self.quantile=quantile; self.alpha=alpha
        self.lr=lr; self.max_iter=int(max_iter)
        self.coef_=None; self.intercept_=0.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        w = [0.0]*d; b = 0.0; q = self.quantile
        for _ in range(self.max_iter):
            yp = [_dot(X[i],w)+b for i in range(n)]
            gw = [0.0]*d; gb = 0.0
            for i in range(n):
                r = y[i]-yp[i]
                rho = (q-1) if r < 0 else q
                for j in range(d): gw[j] -= rho*X[i][j]/n
                gw = [gw[j]+self.alpha*w[j] for j in range(d)]
                gb -= rho/n
            w = [w[j]-self.lr*gw[j] for j in range(d)]; b -= self.lr*gb
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)


# ── BayesianRidge ─────────────────────────────────────────────────────────────

class BayesianRidge:
    """Ridge bayesiano — estima alpha y lambda por evidencia (versión simplificada)."""
    def __init__(self, max_iter=300, tol=1e-3, lr=0.01):
        self.max_iter=int(max_iter); self.tol=tol; self.lr=lr
        self.coef_=None; self.intercept_=0.0
        self.alpha_=1.0; self.lambda_=1.0

    def fit(self, X, y):
        X = _to2d(X); n, d = len(X), len(X[0])
        # Gradient descent con regularización adaptativa (simplificado)
        w = [0.0]*d; b = 0.0
        for ep in range(self.max_iter):
            yp = [_dot(X[i],w)+b for i in range(n)]
            res = [yp[i]-y[i] for i in range(n)]
            gw = [(self.alpha_/n)*sum(res[i]*X[i][j] for i in range(n))+self.lambda_*w[j] for j in range(d)]
            gb = (self.alpha_/n)*sum(res)
            w_new = [w[j]-self.lr*gw[j] for j in range(d)]; b -= self.lr*gb
            # Actualizar hiperparámetros empíricamente cada 10 épocas
            if ep % 10 == 0:
                mse = sum(r**2 for r in res)/n
                self.alpha_ = max(1e-6, 1.0/(mse+1e-9))
                w2 = sum(wi**2 for wi in w)
                self.lambda_ = max(1e-6, d/(w2+1e-9)) if w2 > 0 else 1.0
            if max(abs_doz(w_new[j]-w[j]) for j in range(d)) < self.tol:
                w = w_new; break
            w = w_new
        self.coef_=w; self.intercept_=b; return self

    def predict(self, X):
        return [_dot(row,self.coef_)+self.intercept_ for row in _to2d(X)]

    def score(self, X, y): return LinearRegression.score(self, X, y)
