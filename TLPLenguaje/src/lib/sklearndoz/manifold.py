"""
sklearndoz.manifold — Reducción no lineal de dimensionalidad, Python puro.

Cubre: TSNE (aproximado), MDS, Isomap (aproximado).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz, log_doz
from lib.randomdoz.randomdoz import gauss as _gauss, random as _random
from errors import TrezRuntimeError


def _dist(a, b): return sqrt_doz(sum((ai-bi)**2 for ai,bi in zip(a,b)))
def _mean(v): return sum(v)/len(v) if v else 0.0


# ── TSNE ──────────────────────────────────────────────────────────────────────

class TSNE:
    """t-SNE aproximado con perplexity fija (Barnes-Hut simplificado)."""
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=12.0, random_state=None):
        self.n_components=int(n_components); self.perplexity=perplexity
        self.learning_rate=learning_rate; self.n_iter=int(n_iter)
        self.early_exaggeration=early_exaggeration
        self.embedding_=None; self.kl_divergence_=None

    def fit_transform(self, X, y=None):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); k = self.n_components
        # Probabilidades de alta dimensión (matriz P simétrica)
        sigma = 1.0; target_entropy = log_doz(self.perplexity)
        P = [[0.0]*n for _ in range(n)]
        for i in range(n):
            dists = [_dist(X[i], X[j])**2 for j in range(n)]
            # Binary search for sigma (simplified: fixed sigma)
            for sig_iter in range(50):
                exps = [exp_doz(-dists[j]/(2*sigma**2)) if j!=i else 0.0 for j in range(n)]
                total = sum(exps) or 1e-10
                probs = [e/total for e in exps]
                entropy = -sum(p*log_doz(p+1e-10) for p in probs if p > 1e-10)
                if abs_doz(entropy - target_entropy) < 1e-3: break
                if entropy < target_entropy: sigma *= 1.5
                else: sigma *= 0.75
            for j in range(n): P[i][j] = probs[j]
        # Simetrizar P
        for i in range(n):
            for j in range(n):
                pij = (P[i][j]+P[j][i])/(2*n)
                P[i][j] = max(pij, 1e-12)
        # Inicializar embedding aleatorio
        Y = [[_gauss(0, 0.01) for _ in range(k)] for _ in range(n)]
        lr = self.learning_rate; mom = 0.5
        dY = [[0.0]*k for _ in range(n)]
        gains = [[1.0]*k for _ in range(n)]
        exag = self.early_exaggeration
        for it in range(self.n_iter):
            if it == 250: mom = 0.8; exag = 1.0
            # Q matrix (low-dim)
            Q = [[0.0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    d2 = sum((Y[i][d]-Y[j][d])**2 for d in range(k))
                    q = 1.0/(1.0+d2)
                    Q[i][j] = Q[j][i] = q
            q_total = sum(Q[i][j] for i in range(n) for j in range(n) if i!=j) or 1e-10
            for i in range(n):
                for j in range(n): Q[i][j] /= q_total
                for j in range(n): Q[i][j] = max(Q[i][j], 1e-12)
            # Gradientes
            for i in range(n):
                grad = [0.0]*k
                for j in range(n):
                    if j == i: continue
                    d2 = sum((Y[i][d]-Y[j][d])**2 for d in range(k))
                    q = 1.0/(1.0+d2)
                    factor = 4*(exag*P[i][j]-Q[i][j])*q
                    for d in range(k): grad[d] += factor*(Y[i][d]-Y[j][d])
                for d in range(k):
                    g = 1.0 if grad[d]*dY[i][d] > 0 else 0.2
                    gains[i][d] = max(0.01, (gains[i][d]+0.2) if g==0.2 else gains[i][d]*0.8)
                    dY[i][d] = mom*dY[i][d] - lr*gains[i][d]*grad[d]
                    Y[i][d] += dY[i][d]
            # Re-centrar
            for d in range(k):
                m = _mean([Y[i][d] for i in range(n)])
                for i in range(n): Y[i][d] -= m
        self.embedding_ = Y
        self.kl_divergence_ = 0.0
        return Y

    def fit(self, X, y=None):
        self.fit_transform(X, y); return self


# ── MDS ───────────────────────────────────────────────────────────────────────

class MDS:
    """Multidimensional Scaling (métrico, clásico)."""
    def __init__(self, n_components=2, metric=True, max_iter=300, eps=1e-3):
        self.n_components=int(n_components); self.metric=metric
        self.max_iter=int(max_iter); self.eps=eps
        self.embedding_=None; self.stress_=None

    def fit_transform(self, X, y=None):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); k = self.n_components
        # Matriz de distancias cuadradas
        D2 = [[_dist(X[i],X[j])**2 for j in range(n)] for i in range(n)]
        # Double centering
        row_means = [_mean(D2[i]) for i in range(n)]
        total_mean = _mean([v for row in D2 for v in row])
        B = [[-0.5*(D2[i][j]-row_means[i]-row_means[j]+total_mean) for j in range(n)] for i in range(n)]
        # Power iteration para primeros k eigenvectors
        from lib.sklearndoz.decomposition import _power_iteration, _deflate
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        vecs = []; evals = []
        Bc = [row[:] for row in B]
        for _ in range(k):
            ev, vec = _power_iteration(Bc, 100)
            evals.append(max(0.0, ev)); vecs.append(vec)
            Bc = _deflate(Bc, ev, vec)
        # Embedding = sqrt(eigenval) * eigenvec
        embed = [[sqrt_doz(evals[d])*vecs[d][i] for d in range(k)] for i in range(n)]
        self.embedding_ = embed
        self.stress_ = sum((_dist(X[i],X[j])-_dist(embed[i],embed[j]))**2 for i in range(n) for j in range(i+1,n))
        return embed

    def fit(self, X, y=None):
        self.fit_transform(X, y); return self


# ── Isomap ────────────────────────────────────────────────────────────────────

class Isomap:
    """Isomap aproximado: grafo k-NN + MDS sobre distancias geodésicas."""
    def __init__(self, n_neighbors=5, n_components=2, max_iter=300):
        self.n_neighbors=int(n_neighbors); self.n_components=int(n_components)
        self.max_iter=int(max_iter)
        self.embedding_=None; self._X=None

    def fit_transform(self, X, y=None):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); k = self.n_neighbors
        INF = float('inf')
        # Grafo k-NN (distancias euclídeas)
        G = [[INF]*n for _ in range(n)]
        for i in range(n): G[i][i] = 0.0
        for i in range(n):
            dists = sorted([(j, _dist(X[i],X[j])) for j in range(n) if j!=i], key=lambda t: t[1])
            for j, d in dists[:k]:
                G[i][j] = d; G[j][i] = d
        # Floyd-Warshall para distancias geodésicas
        for mid in range(n):
            for i in range(n):
                if G[i][mid] == INF: continue
                for j in range(n):
                    via = G[i][mid]+G[mid][j]
                    if via < G[i][j]: G[i][j] = via
        # MDS sobre matriz geodésica
        # Double centering
        D2 = [[G[i][j]**2 if G[i][j] < INF else 0.0 for j in range(n)] for i in range(n)]
        row_means = [_mean(D2[i]) for i in range(n)]
        total_mean = _mean([v for row in D2 for v in row])
        B = [[-0.5*(D2[i][j]-row_means[i]-row_means[j]+total_mean) for j in range(n)] for i in range(n)]
        from lib.sklearndoz.decomposition import _power_iteration, _deflate
        vecs = []; evals = []
        Bc = [row[:] for row in B]; kc = self.n_components
        for _ in range(kc):
            ev, vec = _power_iteration(Bc, 100)
            evals.append(max(0.0, ev)); vecs.append(vec)
            Bc = _deflate(Bc, ev, vec)
        embed = [[sqrt_doz(evals[d])*vecs[d][i] for d in range(kc)] for i in range(n)]
        self.embedding_ = embed; self._X = X
        return embed

    def fit(self, X, y=None):
        self.fit_transform(X, y); return self
