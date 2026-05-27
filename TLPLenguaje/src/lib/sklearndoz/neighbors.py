"""
sklearndoz.neighbors — Métodos basados en vecinos, Python puro.

Cubre: KNeighborsClassifier, KNeighborsRegressor,
       RadiusNeighborsClassifier, NearestCentroid,
       LocalOutlierFactor, NearestNeighbors.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz
from errors import TrezRuntimeError


def _dist(a, b):
    return sqrt_doz(sum((ai-bi)**2 for ai,bi in zip(a,b)))

def _mean(v): return sum(v)/len(v) if v else 0.0

def _mode(v):
    counts = {}
    for x in v: counts[x] = counts.get(x,0)+1
    return max(counts, key=lambda k: counts[k])


# ── NearestNeighbors ──────────────────────────────────────────────────────────

class NearestNeighbors:
    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors=int(n_neighbors); self.metric=metric
        self._X=None

    def fit(self, X):
        self._X = X if isinstance(X[0], list) else [[v] for v in X]
        return self

    def kneighbors(self, X, n_neighbors=None):
        k = n_neighbors or self.n_neighbors
        X = X if isinstance(X[0], list) else [[v] for v in X]
        dists_all=[]; idxs_all=[]
        for x in X:
            d = [(i, _dist(x, self._X[i])) for i in range(len(self._X))]
            d.sort(key=lambda t: t[1])
            top = d[:k]
            idxs_all.append([t[0] for t in top])
            dists_all.append([t[1] for t in top])
        return dists_all, idxs_all


# ── KNeighborsClassifier ──────────────────────────────────────────────────────

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors=int(n_neighbors); self.weights=weights; self.metric=metric
        self._X=None; self._y=None; self.classes_=None

    def fit(self, X, y):
        self._X = X if isinstance(X[0], list) else [[v] for v in X]
        self._y = list(y)
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            d = [(i, _dist(x, self._X[i])) for i in range(len(self._X))]
            d.sort(key=lambda t: t[1])
            top = d[:self.n_neighbors]
            if self.weights == 'distance':
                votes = {}
                for i, di in top:
                    w = 1.0/(di+1e-10)
                    votes[self._y[i]] = votes.get(self._y[i], 0)+w
                result.append(max(votes, key=lambda k: votes[k]))
            else:
                result.append(_mode([self._y[i] for i,_ in top]))
        return result

    def predict_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            d = [(i, _dist(x, self._X[i])) for i in range(len(self._X))]
            d.sort(key=lambda t: t[1])
            top = d[:self.n_neighbors]
            votes = {c: 0.0 for c in self.classes_}
            for i, di in top:
                w = 1.0/(di+1e-10) if self.weights=='distance' else 1.0
                votes[self._y[i]] += w
            total = sum(votes.values()) or 1e-10
            result.append([votes[c]/total for c in self.classes_])
        return result

    def score(self, X, y):
        preds = self.predict(X)
        return sum(1 for a,b in zip(y,preds) if a==b)/len(y)


# ── KNeighborsRegressor ───────────────────────────────────────────────────────

class KNeighborsRegressor:
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors=int(n_neighbors); self.weights=weights; self.metric=metric
        self._X=None; self._y=None

    def fit(self, X, y):
        self._X = X if isinstance(X[0], list) else [[v] for v in X]
        self._y = list(y)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            d = [(i, _dist(x, self._X[i])) for i in range(len(self._X))]
            d.sort(key=lambda t: t[1])
            top = d[:self.n_neighbors]
            if self.weights == 'distance':
                ws = [1.0/(di+1e-10) for _,di in top]
                total = sum(ws)
                result.append(sum(w*self._y[i]/total for (i,_),w in zip(top, ws)))
            else:
                result.append(_mean([self._y[i] for i,_ in top]))
        return result

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((yi-pi)**2 for yi,pi in zip(y,preds))
        ss_tot = sum((yi-_mean(list(y)))**2 for yi in y) or 1e-10
        return 1-ss_res/ss_tot


# ── RadiusNeighborsClassifier ─────────────────────────────────────────────────

class RadiusNeighborsClassifier:
    def __init__(self, radius=1.0, weights='uniform', outlier_label=-1):
        self.radius=radius; self.weights=weights; self.outlier_label=outlier_label
        self._X=None; self._y=None; self.classes_=None

    def fit(self, X, y):
        self._X = X if isinstance(X[0], list) else [[v] for v in X]
        self._y = list(y); self.classes_ = sorted(set(y))
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            nbrs = [(i, _dist(x, self._X[i])) for i in range(len(self._X)) if _dist(x, self._X[i]) <= self.radius]
            if not nbrs: result.append(self.outlier_label); continue
            if self.weights == 'distance':
                votes = {}
                for i,di in nbrs:
                    w = 1.0/(di+1e-10); votes[self._y[i]] = votes.get(self._y[i],0)+w
                result.append(max(votes, key=lambda k: votes[k]))
            else:
                result.append(_mode([self._y[i] for i,_ in nbrs]))
        return result

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── NearestCentroid ───────────────────────────────────────────────────────────

class NearestCentroid:
    def __init__(self, metric='euclidean', shrink_threshold=None):
        self.metric=metric; self.shrink_threshold=shrink_threshold
        self.centroids_=None; self.classes_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); d = len(X[0]); n = len(X)
        self.centroids_ = {}
        for c in self.classes_:
            pts = [X[i] for i in range(n) if y[i]==c]
            self.centroids_[c] = [_mean([p[j] for p in pts]) for j in range(d)]
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [min(self.classes_, key=lambda c: _dist(x, self.centroids_[c])) for x in X]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── LocalOutlierFactor ────────────────────────────────────────────────────────

class LocalOutlierFactor:
    """LOF — puntuación de anomalía local. novelty=False (transductivo)."""
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors=int(n_neighbors); self.contamination=contamination
        self.negative_outlier_factor_=None; self._X=None

    def fit_predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); k = min(self.n_neighbors, n-1)
        # k-distancia y vecindarios
        kdist = []; nbrs = []
        for i in range(n):
            d = sorted([(j, _dist(X[i], X[j])) for j in range(n) if j!=i], key=lambda t: t[1])
            kdist.append(d[k-1][1])
            nbrs.append([j for j,dist in d[:k]])
        # Reachability distance
        def reach_dist(i, j): return max(kdist[j], _dist(X[i], X[j]))
        # Local reachability density
        lrd = []
        for i in range(n):
            avg_rd = _mean([reach_dist(i, j) for j in nbrs[i]]) or 1e-10
            lrd.append(1.0/avg_rd)
        # LOF scores
        lof = [_mean([lrd[j] for j in nbrs[i]])/lrd[i] for i in range(n)]
        self.negative_outlier_factor_ = [-s for s in lof]
        threshold = sorted(lof)[int(n*(1-self.contamination))]
        return [1 if s < threshold else -1 for s in lof]

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.fit_predict(X); self._X = X; return self
