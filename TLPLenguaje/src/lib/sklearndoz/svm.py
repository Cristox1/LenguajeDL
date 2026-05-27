"""
sklearndoz.svm — Máquinas de soporte vectorial, Python puro.

Cubre: SVC (kernel lineal/poly/rbf via subgradiente), LinearSVC,
       SVR, LinearSVR, NuSVC, OneClassSVM.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz
from lib.randomdoz.randomdoz import random as _random, gauss as _gauss
from errors import TrezRuntimeError


def _dot(a, b): return sum(ai*bi for ai,bi in zip(a,b))
def _norm(v): return sqrt_doz(sum(x**2 for x in v)) or 1e-10
def _mean(v): return sum(v)/len(v) if v else 0.0


# ── Kernel functions ──────────────────────────────────────────────────────────

def _kernel_linear(a, b, **kw): return _dot(a, b)

def _kernel_poly(a, b, degree=3, coef0=1.0, gamma=1.0, **kw):
    return (gamma*_dot(a,b)+coef0)**degree

def _kernel_rbf(a, b, gamma=1.0, **kw):
    d2 = sum((ai-bi)**2 for ai,bi in zip(a,b))
    return exp_doz(-gamma*d2)

def _kernel_sigmoid(a, b, gamma=1.0, coef0=0.0, **kw):
    z = gamma*_dot(a,b)+coef0
    if z > 20: return 1.0
    if z < -20: return -1.0
    ep = exp_doz(2*z); return (ep-1)/(ep+1)

_KERNELS = {'linear': _kernel_linear, 'poly': _kernel_poly,
            'rbf': _kernel_rbf, 'sigmoid': _kernel_sigmoid}


# ── SVC ───────────────────────────────────────────────────────────────────────

class SVC:
    """SVM de clasificación. Kernels: linear, poly, rbf, sigmoid.
    Entrenamiento via subgradiente (aproximación)."""
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 coef0=0.0, max_iter=1000, tol=1e-4, probability=False):
        self.C=C; self.kernel=kernel; self.degree=int(degree)
        self.gamma=gamma; self.coef0=coef0; self.max_iter=int(max_iter)
        self.tol=tol; self.probability=probability
        self.support_vectors_=None; self.classes_=None
        self._alphas=None; self._sv=None; self._sv_y=None; self._b=0.0

    def _resolve_gamma(self, d):
        if isinstance(self.gamma, (int,float)): return float(self.gamma)
        if self.gamma == 'scale': return 1.0/(d or 1)
        return 1.0/(d or 1)  # 'auto'

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y))
        if len(self.classes_) != 2:
            raise TrezRuntimeError("SVC binario solo. Usa OneVsRestClassifier para multiclase.")
        d = len(X[0]); n = len(X)
        gam = self._resolve_gamma(d)
        kf = _KERNELS.get(self.kernel, _kernel_rbf)
        kw = {'degree': self.degree, 'gamma': gam, 'coef0': self.coef0}
        # Mapear etiquetas a {-1, +1}
        yn = [1.0 if yi==self.classes_[1] else -1.0 for yi in y]
        # Subgradiente dual simplificado: optimizar alphas
        alphas = [0.0]*n; b = 0.0
        lr = 0.01
        for it in range(self.max_iter):
            changed = False
            for i in range(n):
                # Decisión
                dec = sum(alphas[j]*yn[j]*kf(X[j],X[i],**kw) for j in range(n)) + b
                margin = yn[i]*dec
                if margin < 1.0:
                    alphas[i] = min(self.C, alphas[i]+lr)
                    b += lr*yn[i]
                    changed = True
                else:
                    delta = lr*0.01
                    if alphas[i] > 0:
                        alphas[i] = max(0.0, alphas[i]-delta)
            if not changed: break
        sv_mask = [alphas[i] > 1e-6 for i in range(n)]
        self._alphas = [alphas[i] for i in range(n) if sv_mask[i]]
        self._sv = [X[i] for i in range(n) if sv_mask[i]]
        self._sv_y = [yn[i] for i in range(n) if sv_mask[i]]
        self._b = b; self._kf = kf; self._kw = kw
        self.support_vectors_ = self._sv
        return self

    def decision_function(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [sum(self._alphas[j]*self._sv_y[j]*self._kf(self._sv[j],x,**self._kw)
                    for j in range(len(self._sv))) + self._b for x in X]

    def predict(self, X):
        df = self.decision_function(X)
        return [self.classes_[1] if v >= 0 else self.classes_[0] for v in df]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── LinearSVC ─────────────────────────────────────────────────────────────────

class LinearSVC:
    """SVM lineal con hinge loss y regularización L2 (SGD)."""
    def __init__(self, C=1.0, max_iter=1000, tol=1e-4, multi_class='ovr'):
        self.C=C; self.max_iter=int(max_iter); self.tol=tol; self.multi_class=multi_class
        self.coef_=None; self.intercept_=None; self.classes_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); K = len(self.classes_); d = len(X[0]); n = len(X)
        self.coef_ = []; self.intercept_ = []
        for c in self.classes_:
            yn = [1.0 if yi==c else -1.0 for yi in y]
            w = [0.0]*d; b = 0.0; lr = 0.01
            for it in range(self.max_iter):
                dw = [0.0]*d; db = 0.0
                for i in range(n):
                    if yn[i]*(_dot(w, X[i])+b) < 1:
                        for j in range(d): dw[j] -= yn[i]*X[i][j]/n
                        db -= yn[i]/n
                for j in range(d): w[j] = w[j]*(1-lr*2/n) - lr*dw[j]*self.C
                b -= lr*db*self.C
            self.coef_.append(w); self.intercept_.append(b)
        return self

    def decision_function(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [[_dot(self.coef_[k], x)+self.intercept_[k] for k in range(len(self.classes_))] for x in X]

    def predict(self, X):
        df = self.decision_function(X)
        return [self.classes_[scores.index(max(scores))] for scores in df]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── NuSVC ─────────────────────────────────────────────────────────────────────

class NuSVC:
    """SVC con parámetro nu (envuelve SVC con C equivalente)."""
    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0):
        self.nu=nu; self.kernel=kernel; self.degree=degree
        self.gamma=gamma; self.coef0=coef0
        self._svc=None; self.classes_=None

    def fit(self, X, y):
        C = 1.0/(self.nu+1e-10)
        self._svc = SVC(C=C, kernel=self.kernel, degree=self.degree,
                        gamma=self.gamma, coef0=self.coef0)
        self._svc.fit(X, y); self.classes_ = self._svc.classes_
        return self

    def predict(self, X): return self._svc.predict(X)
    def score(self, X, y): return self._svc.score(X, y)
    def decision_function(self, X): return self._svc.decision_function(X)


# ── SVR ───────────────────────────────────────────────────────────────────────

class SVR:
    """SVM de regresión con epsilon-insensitive loss (subgradiente)."""
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, max_iter=1000):
        self.C=C; self.epsilon=epsilon; self.kernel=kernel
        self.degree=int(degree); self.gamma=gamma; self.coef0=coef0
        self.max_iter=int(max_iter)
        self._alphas=None; self._sv=None; self._sv_y=None; self._b=0.0

    def _resolve_gamma(self, d):
        if isinstance(self.gamma, (int,float)): return float(self.gamma)
        return 1.0/(d or 1)

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); d = len(X[0])
        gam = self._resolve_gamma(d)
        kf = _KERNELS.get(self.kernel, _kernel_rbf)
        kw = {'degree': self.degree, 'gamma': gam, 'coef0': self.coef0}
        alphas_p = [0.0]*n; alphas_n = [0.0]*n; b = 0.0; lr = 0.01
        for _ in range(self.max_iter):
            for i in range(n):
                pred = sum((alphas_p[j]-alphas_n[j])*kf(X[j],X[i],**kw) for j in range(n)) + b
                err = pred - y[i]
                if err > self.epsilon:
                    alphas_n[i] = min(self.C, alphas_n[i]+lr)
                    b -= lr
                elif err < -self.epsilon:
                    alphas_p[i] = min(self.C, alphas_p[i]+lr)
                    b += lr
        self._alphas = [alphas_p[i]-alphas_n[i] for i in range(n)]
        self._sv = X; self._sv_y = list(y); self._b = b
        self._kf = kf; self._kw = kw
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [sum(self._alphas[j]*self._kf(self._sv[j],x,**self._kw) for j in range(len(self._sv))) + self._b for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((yi-pi)**2 for yi,pi in zip(y,preds))
        ss_tot = sum((yi-_mean(list(y)))**2 for yi in y) or 1e-10
        return 1-ss_res/ss_tot


# ── LinearSVR ─────────────────────────────────────────────────────────────────

class LinearSVR:
    """SVR lineal con epsilon-insensitive loss (SGD)."""
    def __init__(self, C=1.0, epsilon=0.0, max_iter=1000, tol=1e-4):
        self.C=C; self.epsilon=epsilon; self.max_iter=int(max_iter); self.tol=tol
        self.coef_=None; self.intercept_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); d = len(X[0]); w = [0.0]*d; b = 0.0; lr = 0.01
        for _ in range(self.max_iter):
            dw = [0.0]*d; db = 0.0
            for i in range(n):
                pred = _dot(w, X[i])+b; err = pred-y[i]
                if abs_doz(err) > self.epsilon:
                    s = 1.0 if err > 0 else -1.0
                    for j in range(d): dw[j] += s*X[i][j]/n
                    db += s/n
            for j in range(d): w[j] = w[j]-lr*(w[j]/n+self.C*dw[j])
            b -= lr*self.C*db
        self.coef_ = w; self.intercept_ = b
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [_dot(self.coef_, x)+self.intercept_ for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((yi-pi)**2 for yi,pi in zip(y,preds))
        ss_tot = sum((yi-_mean(list(y)))**2 for yi in y) or 1e-10
        return 1-ss_res/ss_tot


# ── OneClassSVM ───────────────────────────────────────────────────────────────

class OneClassSVM:
    """One-class SVM para detección de anomalías (nu-SVM aproximado)."""
    def __init__(self, nu=0.5, kernel='rbf', gamma='scale', degree=3, max_iter=500):
        self.nu=nu; self.kernel=kernel; self.gamma=gamma
        self.degree=int(degree); self.max_iter=int(max_iter)
        self._rho=0.0; self._alphas=None; self._sv=None

    def _resolve_gamma(self, d):
        if isinstance(self.gamma, (int,float)): return float(self.gamma)
        return 1.0/(d or 1)

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); d = len(X[0])
        gam = self._resolve_gamma(d)
        kf = _KERNELS.get(self.kernel, _kernel_rbf)
        kw = {'degree': self.degree, 'gamma': gam}
        C = 1.0/(self.nu*n+1e-10)
        alphas = [C]*n; lr = 0.01
        for _ in range(self.max_iter):
            for i in range(n):
                score = sum(alphas[j]*kf(X[j],X[i],**kw) for j in range(n))
                if score < self._rho:
                    alphas[i] = min(C, alphas[i]+lr)
                else:
                    alphas[i] = max(0.0, alphas[i]-lr)
            self._rho = _mean([sum(alphas[j]*kf(X[j],X[i],**kw) for j in range(n)) for i in range(n) if alphas[i]>1e-8]) if any(a>1e-8 for a in alphas) else 0.0
        self._alphas = alphas; self._sv = X; self._kf = kf; self._kw = kw
        return self

    def decision_function(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [sum(self._alphas[j]*self._kf(self._sv[j],x,**self._kw) for j in range(len(self._sv))) - self._rho for x in X]

    def predict(self, X):
        df = self.decision_function(X)
        return [1 if v >= 0 else -1 for v in df]

    def score(self, X):
        return sum(1 for v in self.predict(X) if v==1)/len(X)
