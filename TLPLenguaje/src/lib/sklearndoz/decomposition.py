"""
sklearndoz.decomposition — Reducción de dimensionalidad, Python puro.

Cubre: PCA, TruncatedSVD, NMF, FastICA, FactorAnalysis, LDA (LinearDiscriminantAnalysis).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz, log_doz
from lib.randomdoz.randomdoz import gauss as _gauss, random as _random
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0
def _dot(a, b): return sum(ai*bi for ai,bi in zip(a,b))
def _norm(v): return sqrt_doz(sum(x**2 for x in v)) or 1e-10

def _mat_mul(A, B):
    m, k = len(A), len(A[0]); n = len(B[0])
    return [[sum(A[i][p]*B[p][j] for p in range(k)) for j in range(n)] for i in range(m)]

def _transpose(M):
    if not M: return []
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]

def _col(X, j): return [row[j] for row in X]

def _center(X):
    d = len(X[0])
    means = [_mean(_col(X,j)) for j in range(d)]
    return [[row[j]-means[j] for j in range(d)] for row in X], means

def _project(X, V):
    """Proyecta X (n×d) sobre V (k×d), retorna (n×k)."""
    return [[_dot(row, v) for v in V] for row in X]

def _gram_schmidt(vectors):
    """Ortogonaliza una lista de vectores."""
    basis = []
    for v in vectors:
        v = list(v)
        for b in basis:
            proj = _dot(v, b)
            v = [vi - proj*bi for vi, bi in zip(v, b)]
        n = _norm(v)
        if n > 1e-10:
            basis.append([vi/n for vi in v])
    return basis

def _power_iteration(A, n_iter=100):
    """Encuentra el primer vector propio y valor propio de A (cuadrada) via potencias."""
    n = len(A)
    v = [_gauss(0, 1) for _ in range(n)]
    nv = _norm(v); v = [x/nv for x in v]
    for _ in range(n_iter):
        Av = [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        eigenval = _dot(v, Av)
        nAv = _norm(Av)
        if nAv < 1e-12: break
        v = [x/nAv for x in Av]
    eigenval = _dot(v, [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)])
    return eigenval, v

def _deflate(A, eigenval, eigenvec):
    n = len(A)
    return [[A[i][j] - eigenval*eigenvec[i]*eigenvec[j] for j in range(n)] for i in range(n)]

def _cov_matrix(X):
    n, d = len(X), len(X[0])
    means = [_mean(_col(X,j)) for j in range(d)]
    Xc = [[X[i][j]-means[j] for j in range(d)] for i in range(n)]
    Xt = _transpose(Xc)
    return [[_dot(Xt[r], _col(Xc, c))/n for c in range(d)] for r in range(d)]


# ── PCA ───────────────────────────────────────────────────────────────────────

class PCA:
    def __init__(self, n_components=2, n_iter=100):
        self.n_components=int(n_components); self.n_iter=n_iter
        self.components_=None; self.mean_=None; self.explained_variance_=None

    def fit(self, X):
        Xc, self.mean_ = _center(X)
        C = _cov_matrix(Xc)
        comps = []; evs = []
        for _ in range(self.n_components):
            ev, vec = _power_iteration(C, self.n_iter)
            comps.append(vec); evs.append(max(0.0, ev))
            C = _deflate(C, ev, vec)
        self.components_ = comps
        self.explained_variance_ = evs
        total = sum(evs) or 1e-10
        self.explained_variance_ratio_ = [e/total for e in evs]
        return self

    def transform(self, X):
        Xc = [[X[i][j]-self.mean_[j] for j in range(len(X[0]))] for i in range(len(X))]
        return _project(Xc, self.components_)

    def fit_transform(self, X): return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced):
        d = len(self.components_[0])
        return [[sum(X_reduced[i][k]*self.components_[k][j] for k in range(self.n_components))+self.mean_[j]
                 for j in range(d)] for i in range(len(X_reduced))]


# ── TruncatedSVD ──────────────────────────────────────────────────────────────

class TruncatedSVD:
    """SVD truncado via PCA sobre X·Xᵀ (para matrices densas pequeñas)."""
    def __init__(self, n_components=2, n_iter=100):
        self.n_components=int(n_components); self.n_iter=n_iter
        self.components_=None; self.singular_values_=None

    def fit(self, X):
        pca = PCA(self.n_components, self.n_iter)
        pca.mean_ = [0.0]*len(X[0])
        C = _cov_matrix(X)
        comps=[]; svs=[]
        for _ in range(self.n_components):
            ev, vec = _power_iteration(C, self.n_iter)
            comps.append(vec); svs.append(sqrt_doz(max(0.0, ev*len(X))))
            C = _deflate(C, ev, vec)
        self.components_=comps; self.singular_values_=svs
        return self

    def transform(self, X):
        return _project(X, self.components_)

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── NMF (Non-negative Matrix Factorization) ───────────────────────────────────

class NMF:
    def __init__(self, n_components=2, max_iter=200, tol=1e-4):
        self.n_components=int(n_components); self.max_iter=int(max_iter); self.tol=tol
        self.components_=None; self.reconstruction_err_=None

    def fit_transform(self, X):
        n, d = len(X), len(X[0]); k = self.n_components
        # Inicializar W y H con valores no negativos aleatorios
        W = [[max(0.0, _gauss(0.5, 0.1)) for _ in range(k)] for _ in range(n)]
        H = [[max(0.0, _gauss(0.5, 0.1)) for _ in range(d)] for _ in range(k)]
        eps = 1e-10
        for _ in range(self.max_iter):
            # Actualización multiplicativa (Lee & Seung)
            WH = _mat_mul(W, H)
            Ht = _transpose(H)
            # Update W: W ← W * (X Hᵀ) / (WH Hᵀ)
            XHt = _mat_mul(X, Ht)
            WHHt = _mat_mul(WH, Ht)
            W = [[W[i][j]*XHt[i][j]/(WHHt[i][j]+eps) for j in range(k)] for i in range(n)]
            # Update H: H ← H * (Wᵀ X) / (Wᵀ WH)
            Wt = _transpose(W); WH = _mat_mul(W, H)
            WtX = _mat_mul(Wt, X); WtWH = _mat_mul(Wt, WH)
            H = [[H[i][j]*WtX[i][j]/(WtWH[i][j]+eps) for j in range(d)] for i in range(k)]
        self.components_ = H
        self.reconstruction_err_ = sum((X[i][j]-sum(W[i][p]*H[p][j] for p in range(k)))**2
                                       for i in range(n) for j in range(d))
        return W

    def fit(self, X): self.fit_transform(X); return self

    def transform(self, X):
        n, d = len(X), len(X[0]); k = self.n_components; eps = 1e-10
        H = self.components_
        W = [[max(0.0, _gauss(0.5, 0.1)) for _ in range(k)] for _ in range(n)]
        for _ in range(50):
            WH = _mat_mul(W, H); Ht = _transpose(H)
            XHt = _mat_mul(X, Ht); WHHt = _mat_mul(WH, Ht)
            W = [[W[i][j]*XHt[i][j]/(WHHt[i][j]+eps) for j in range(k)] for i in range(n)]
        return W


# ── FastICA ───────────────────────────────────────────────────────────────────

class FastICA:
    """FastICA con función de contraste logcosh."""
    def __init__(self, n_components=2, max_iter=200, tol=1e-4):
        self.n_components=int(n_components); self.max_iter=int(max_iter); self.tol=tol
        self.components_=None; self.mean_=None

    def fit_transform(self, X):
        Xc, self.mean_ = _center(X)
        n, d = len(Xc), len(Xc[0])
        # Blanquear (whitening) via PCA
        pca = PCA(self.n_components)
        Xw = pca.fit_transform(Xc)
        # ICA deflacionaria
        W = []
        for p in range(self.n_components):
            w = [_gauss(0,1) for _ in range(self.n_components)]
            nw = _norm(w); w = [x/nw for x in w]
            for _ in range(self.max_iter):
                # logcosh: g(u) = tanh(u), g'(u) = 1-tanh²(u)
                def _tanh(z):
                    if z > 20: return 1.0
                    if z < -20: return -1.0
                    ep = exp_doz(2*z); return (ep-1)/(ep+1)
                wx = [_dot(w, row) for row in Xw]
                g = [_tanh(u) for u in wx]
                gp = [1-gi**2 for gi in g]
                w_new = [sum(g[i]*Xw[i][j] for i in range(n))/n - sum(gp)/n*w[j] for j in range(self.n_components)]
                # Decorrelate con W ya calculados
                for wb in W:
                    proj = _dot(w_new, wb)
                    w_new = [w_new[j]-proj*wb[j] for j in range(self.n_components)]
                nw = _norm(w_new)
                w_new = [x/nw for x in w_new] if nw > 1e-10 else w_new
                if abs_doz(abs_doz(_dot(w_new, w))-1) < self.tol:
                    w = w_new; break
                w = w_new
            W.append(w)
        self.components_ = W
        return [[_dot(w, row) for w in W] for row in Xw]

    def fit(self, X): self.fit_transform(X); return self


# ── FactorAnalysis ────────────────────────────────────────────────────────────

class FactorAnalysis:
    """Análisis factorial simplificado (versión EM básica)."""
    def __init__(self, n_components=2, max_iter=100, tol=1e-3):
        self.n_components=int(n_components); self.max_iter=int(max_iter); self.tol=tol
        self.components_=None; self.noise_variance_=None; self.mean_=None

    def fit(self, X):
        pca = PCA(self.n_components)
        pca.fit(X)
        self.components_ = pca.components_
        self.mean_ = pca.mean_
        Xc = [[X[i][j]-self.mean_[j] for j in range(len(X[0]))] for i in range(len(X))]
        scores = _project(Xc, self.components_)
        reconst = [[sum(scores[i][k]*self.components_[k][j]
                        for k in range(self.n_components))
                    for j in range(len(X[0]))] for i in range(len(X))]
        resid = [[Xc[i][j]-reconst[i][j] for j in range(len(X[0]))] for i in range(len(X))]
        d = len(X[0])
        self.noise_variance_ = [_mean([resid[i][j]**2 for i in range(len(X))]) for j in range(d)]
        return self

    def transform(self, X):
        Xc = [[X[i][j]-self.mean_[j] for j in range(len(X[0]))] for i in range(len(X))]
        return _project(Xc, self.components_)

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── LinearDiscriminantAnalysis ────────────────────────────────────────────────

class LinearDiscriminantAnalysis:
    def __init__(self, n_components=None):
        self.n_components=n_components
        self.scalings_=None; self.means_=None; self.classes_=None; self.priors_=None

    def fit(self, X, y):
        d = len(X[0]); self.classes_ = sorted(set(y))
        K = len(self.classes_); n = len(X)
        nc = min(K-1, d) if self.n_components is None else self.n_components
        self.priors_ = {c: sum(1 for v in y if v==c)/n for c in self.classes_}
        self.means_ = {c: [_mean([X[i][j] for i in range(n) if y[i]==c]) for j in range(d)]
                       for c in self.classes_}
        overall = [_mean(_col(X,j)) for j in range(d)]
        # Between-class scatter
        Sb = [[sum(self.priors_[c]*(self.means_[c][r]-overall[r])*(self.means_[c][s]-overall[s])
                   for c in self.classes_) for s in range(d)] for r in range(d)]
        # Within-class scatter
        Sw = [[0.0]*d for _ in range(d)]
        for i in range(n):
            c = y[i]; diff = [X[i][j]-self.means_[c][j] for j in range(d)]
            for r in range(d):
                for s in range(d): Sw[r][s] += diff[r]*diff[s]
        # Aproximación: proyectar usando diferencias de medias (simplificado)
        pca = PCA(nc)
        pca.fit(Sb)
        self.scalings_ = pca.components_
        return self

    def transform(self, X):
        return _project(X, self.scalings_)

    def fit_transform(self, X, y): return self.fit(X,y).transform(X)

    def predict(self, X):
        scores = self.transform(X)
        result = []
        for sc in scores:
            best_c = None; best_d = float('inf')
            tr_means = {c: _project([self.means_[c]], self.scalings_)[0] for c in self.classes_}
            for c in self.classes_:
                dist = sum((sc[j]-tr_means[c][j])**2 for j in range(len(sc)))
                if dist < best_d: best_d = dist; best_c = c
            result.append(best_c)
        return result

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)
