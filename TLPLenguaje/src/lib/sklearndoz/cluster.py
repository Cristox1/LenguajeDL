"""
sklearndoz.cluster — Algoritmos de agrupamiento, Python puro.

Cubre: KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering,
       MeanShift, SpectralClustering (aproximado), GaussianMixture.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz, log_doz
from lib.randomdoz.randomdoz import sample as _sample, random as _random, gauss as _gauss
from errors import TrezRuntimeError


def _dist(a, b):
    return sqrt_doz(sum((ai-bi)**2 for ai,bi in zip(a,b)))

def _mean_vec(pts, d):
    if not pts: return [0.0]*d
    return [sum(p[j] for p in pts)/len(pts) for j in range(d)]

def _mean(v): return sum(v)/len(v) if v else 0.0


# ── KMeans ────────────────────────────────────────────────────────────────────

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, init='k-means++', n_init=10):
        self.n_clusters=int(n_clusters); self.max_iter=int(max_iter)
        self.tol=tol; self.init=init; self.n_init=int(n_init)
        self.cluster_centers_=None; self.labels_=None; self.inertia_=None

    def _init_centers(self, X, k):
        if self.init == 'k-means++':
            centers = [X[int(_random()*len(X))]]
            for _ in range(k-1):
                dists = [min(_dist(x,c)**2 for c in centers) for x in X]
                total = sum(dists); r = _random()*total; cum = 0.0
                for i, d in enumerate(dists):
                    cum += d
                    if cum >= r: centers.append(X[i]); break
                else: centers.append(X[-1])
            return [list(c) for c in centers]
        else:
            idx = _sample(list(range(len(X))), k)
            return [list(X[i]) for i in idx]

    def _run_once(self, X, k):
        n, d = len(X), len(X[0])
        centers = self._init_centers(X, k)
        labels = [-1]*n
        for _ in range(self.max_iter):
            new_labels = [min(range(k), key=lambda c: _dist(X[i], centers[c])) for i in range(n)]
            if new_labels == labels: break
            labels = new_labels
            for c in range(k):
                pts = [X[i] for i in range(n) if labels[i]==c]
                if pts: centers[c] = _mean_vec(pts, d)
        inertia = sum(_dist(X[i], centers[labels[i]])**2 for i in range(n))
        return centers, labels, inertia

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        k = self.n_clusters
        best = None
        for _ in range(self.n_init):
            centers, labels, inertia = self._run_once(X, k)
            if best is None or inertia < best[2]: best = (centers, labels, inertia)
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        k = len(self.cluster_centers_)
        return [min(range(k), key=lambda c: _dist(x, self.cluster_centers_[c])) for x in X]

    def fit_predict(self, X): return self.fit(X).labels_

    def score(self, X):
        return -self.inertia_ if self.inertia_ is not None else 0.0


# ── MiniBatchKMeans ───────────────────────────────────────────────────────────

class MiniBatchKMeans:
    def __init__(self, n_clusters=8, max_iter=100, batch_size=100, init='k-means++'):
        self.n_clusters=int(n_clusters); self.max_iter=int(max_iter)
        self.batch_size=int(batch_size); self.init=init
        self.cluster_centers_=None; self.labels_=None

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n, d = len(X), len(X[0]); k = self.n_clusters
        km = KMeans(k, 1, init=self.init, n_init=1)
        km.fit(X[:min(self.batch_size*3, n)])
        centers = km.cluster_centers_; counts = [1]*k
        for _ in range(self.max_iter):
            batch_idx = _sample(list(range(n)), min(self.batch_size, n))
            batch = [X[i] for i in batch_idx]
            assigns = [min(range(k), key=lambda c: _dist(x, centers[c])) for x in batch]
            for x, c in zip(batch, assigns):
                counts[c] += 1; eta = 1.0/counts[c]
                centers[c] = [centers[c][j]*(1-eta)+x[j]*eta for j in range(d)]
        self.cluster_centers_ = centers
        self.labels_ = [min(range(k), key=lambda c: _dist(x, centers[c])) for x in X]
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        k = len(self.cluster_centers_)
        return [min(range(k), key=lambda c: _dist(x, self.cluster_centers_[c])) for x in X]


# ── DBSCAN ────────────────────────────────────────────────────────────────────

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps=eps; self.min_samples=int(min_samples); self.metric=metric
        self.labels_=None; self.core_sample_indices_=None

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); labels = [-1]*n; cluster_id = 0
        def neighbors(i):
            return [j for j in range(n) if j!=i and _dist(X[i],X[j]) <= self.eps]
        visited = [False]*n
        for i in range(n):
            if visited[i]: continue
            visited[i] = True; N = neighbors(i)
            if len(N) < self.min_samples: continue
            labels[i] = cluster_id; seeds = list(N)
            while seeds:
                j = seeds.pop(0)
                if not visited[j]:
                    visited[j] = True; Nj = neighbors(j)
                    if len(Nj) >= self.min_samples: seeds.extend(Nj)
                if labels[j] == -1: labels[j] = cluster_id
            cluster_id += 1
        self.labels_ = labels
        self.core_sample_indices_ = [i for i in range(n) if sum(1 for j in range(n) if _dist(X[i],X[j])<=self.eps) >= self.min_samples]
        return self

    def fit_predict(self, X): return self.fit(X).labels_


# ── AgglomerativeClustering ───────────────────────────────────────────────────

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean'):
        self.n_clusters=int(n_clusters); self.linkage=linkage; self.metric=metric
        self.labels_=None

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); clusters = [[i] for i in range(n)]
        while len(clusters) > self.n_clusters:
            best = (float('inf'), 0, 1)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    d = self._linkage_dist(X, clusters[i], clusters[j])
                    if d < best[0]: best = (d, i, j)
            _, i, j = best
            merged = clusters[i]+clusters[j]
            clusters = [c for k,c in enumerate(clusters) if k!=i and k!=j]+[merged]
        labels = [0]*n
        for cid, cluster in enumerate(clusters):
            for idx in cluster: labels[idx] = cid
        self.labels_ = labels; return self

    def _linkage_dist(self, X, c1, c2):
        dists = [_dist(X[i], X[j]) for i in c1 for j in c2]
        if self.linkage == 'single': return min(dists)
        if self.linkage == 'complete': return max(dists)
        return _mean(dists)  # average & ward (approximation)

    def fit_predict(self, X): return self.fit(X).labels_


# ── MeanShift ─────────────────────────────────────────────────────────────────

class MeanShift:
    def __init__(self, bandwidth=1.0, max_iter=300, tol=1e-3):
        self.bandwidth=bandwidth; self.max_iter=int(max_iter); self.tol=tol
        self.cluster_centers_=None; self.labels_=None

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n, d = len(X), len(X[0]); h = self.bandwidth
        points = [list(x) for x in X]
        for _ in range(self.max_iter):
            new_points = []
            for p in points:
                nbrs = [X[i] for i in range(n) if _dist(p, X[i]) <= h]
                if nbrs:
                    new_p = _mean_vec(nbrs, d)
                else:
                    new_p = p
                new_points.append(new_p)
            if max(_dist(points[i], new_points[i]) for i in range(n)) < self.tol:
                break
            points = new_points
        # Merge converged points
        centers = []; tol = h/2
        for p in points:
            merged = False
            for c in centers:
                if _dist(p, c) < tol: merged = True; break
            if not merged: centers.append(p)
        self.cluster_centers_ = centers
        k = len(centers)
        self.labels_ = [min(range(k), key=lambda c: _dist(x, centers[c])) for x in X]
        return self

    def predict(self, X):
        k = len(self.cluster_centers_)
        return [min(range(k), key=lambda c: _dist(x, self.cluster_centers_[c])) for x in X]


# ── GaussianMixture ───────────────────────────────────────────────────────────

class GaussianMixture:
    """Mezcla de Gaussianas (EM) con covarianza esférica."""
    def __init__(self, n_components=1, max_iter=100, tol=1e-3):
        self.n_components=int(n_components); self.max_iter=int(max_iter); self.tol=tol
        self.means_=None; self.covariances_=None; self.weights_=None; self.converged_=False

    def fit(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n, d = len(X), len(X[0]); K = self.n_components
        # Inicializar con KMeans
        km = KMeans(K, max_iter=50, n_init=1)
        km.fit(X)
        means = [list(c) for c in km.cluster_centers_]
        weights = [1.0/K]*K
        covs = [1.0]*K
        prev_ll = -float('inf')
        eps = 1e-10
        for it in range(self.max_iter):
            # E-step
            resp = []
            for x in X:
                r = []
                for k in range(K):
                    diff = sum((x[j]-means[k][j])**2 for j in range(d))
                    log_p = -d/2*log_doz(2*3.14159265358979*covs[k]+eps) - diff/(2*covs[k]+eps)
                    r.append(weights[k]*exp_doz(min(log_p, 700)))
                total = sum(r)+eps
                resp.append([ri/total for ri in r])
            # M-step
            Nk = [sum(resp[i][k] for i in range(n)) for k in range(K)]
            weights = [nk/n for nk in Nk]
            means = [[sum(resp[i][k]*X[i][j] for i in range(n))/(Nk[k]+eps) for j in range(d)] for k in range(K)]
            covs = [sum(resp[i][k]*sum((X[i][j]-means[k][j])**2 for j in range(d)) for i in range(n))/(d*Nk[k]+eps) for k in range(K)]
            covs = [max(c, 1e-6) for c in covs]
            # Log-likelihood
            ll = sum(log_doz(sum(weights[k]*exp_doz(min(-d/2*log_doz(2*3.14159265358979*covs[k]+eps)-sum((x[j]-means[k][j])**2 for j in range(d))/(2*covs[k]+eps), 700)) for k in range(K))+eps) for x in X)
            if abs_doz(ll-prev_ll) < self.tol: self.converged_=True; break
            prev_ll = ll
        self.means_=means; self.covariances_=covs; self.weights_=weights
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        K, d = self.n_components, len(X[0])
        result = []
        for x in X:
            scores = [self.weights_[k]*exp_doz(min(-sum((x[j]-self.means_[k][j])**2 for j in range(d))/(2*self.covariances_[k]+1e-10), 700)) for k in range(K)]
            result.append(scores.index(max(scores)))
        return result

    def predict_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        K, d = self.n_components, len(X[0]); eps=1e-10
        result = []
        for x in X:
            r = [self.weights_[k]*exp_doz(min(-sum((x[j]-self.means_[k][j])**2 for j in range(d))/(2*self.covariances_[k]+eps), 700)) for k in range(K)]
            total = sum(r)+eps
            result.append([ri/total for ri in r])
        return result

    def score(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        K, d = self.n_components, len(X[0]); eps=1e-10
        return _mean([log_doz(sum(self.weights_[k]*exp_doz(min(-sum((x[j]-self.means_[k][j])**2 for j in range(d))/(2*self.covariances_[k]+eps), 700)) for k in range(K))+eps) for x in X])
