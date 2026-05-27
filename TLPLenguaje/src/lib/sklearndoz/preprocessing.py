"""
sklearndoz.preprocessing — Preprocesamiento, Python puro.

Cubre: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
       LabelEncoder, OrdinalEncoder, OneHotEncoder,
       PolynomialFeatures, SimpleImputer, Normalizer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz
from errors import TrezRuntimeError


def _col(X, j):
    return [row[j] for row in X]

def _mean(vals):
    return sum(vals)/len(vals) if vals else 0.0

def _median(vals):
    s = sorted(vals); n = len(s)
    return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]

def _std(vals):
    m = _mean(vals)
    return sqrt_doz(sum((v-m)**2 for v in vals)/len(vals)) if vals else 1.0

def _quantile(vals, q):
    s = sorted(vals); n = len(s)
    idx = q*(n-1); lo = int(idx); hi = min(lo+1, n-1)
    return s[lo]+(s[hi]-s[lo])*(idx-lo)


# ── StandardScaler ────────────────────────────────────────────────────────────

class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean=with_mean; self.with_std=with_std
        self.mean_=None; self.scale_=None

    def fit(self, X):
        d = len(X[0])
        self.mean_ = [_mean(_col(X,j)) for j in range(d)]
        self.scale_ = [max(_std(_col(X,j)), 1e-10) for j in range(d)]
        return self

    def transform(self, X):
        d = len(X[0])
        return [[(row[j]-(self.mean_[j] if self.with_mean else 0))/(self.scale_[j] if self.with_std else 1.0)
                 for j in range(d)] for row in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        d = len(X[0])
        return [[row[j]*self.scale_[j]+self.mean_[j] for j in range(d)] for row in X]


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.lo, self.hi = feature_range
        self.data_min_=None; self.data_max_=None

    def fit(self, X):
        d = len(X[0])
        self.data_min_ = [min(_col(X,j)) for j in range(d)]
        self.data_max_ = [max(_col(X,j)) for j in range(d)]
        return self

    def transform(self, X):
        d = len(X[0])
        return [[(row[j]-self.data_min_[j])/(max(self.data_max_[j]-self.data_min_[j], 1e-10))*(self.hi-self.lo)+self.lo
                 for j in range(d)] for row in X]

    def fit_transform(self, X): return self.fit(X).transform(X)

    def inverse_transform(self, X):
        d = len(X[0])
        return [[(row[j]-self.lo)/(self.hi-self.lo)*(self.data_max_[j]-self.data_min_[j])+self.data_min_[j]
                 for j in range(d)] for row in X]


# ── RobustScaler ──────────────────────────────────────────────────────────────

class RobustScaler:
    """Escala usando mediana e IQR — robusto a outliers."""
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
        self.with_centering=with_centering; self.with_scaling=with_scaling
        self.q_lo, self.q_hi = quantile_range[0]/100, quantile_range[1]/100
        self.center_=None; self.scale_=None

    def fit(self, X):
        d = len(X[0])
        self.center_ = [_median(_col(X,j)) for j in range(d)]
        self.scale_ = [max(_quantile(_col(X,j),self.q_hi)-_quantile(_col(X,j),self.q_lo), 1e-10)
                       for j in range(d)]
        return self

    def transform(self, X):
        d = len(X[0])
        return [[(row[j]-(self.center_[j] if self.with_centering else 0))/(self.scale_[j] if self.with_scaling else 1.0)
                 for j in range(d)] for row in X]

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── MaxAbsScaler ──────────────────────────────────────────────────────────────

class MaxAbsScaler:
    def __init__(self): self.scale_=None

    def fit(self, X):
        d = len(X[0])
        self.scale_ = [max(max(abs_doz(v) for v in _col(X,j)), 1e-10) for j in range(d)]
        return self

    def transform(self, X):
        return [[row[j]/self.scale_[j] for j in range(len(X[0]))] for row in X]

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """Normaliza cada fila (muestra) a norma unitaria."""
    def __init__(self, norm='l2'):
        self.norm = norm

    def fit(self, X): return self  # stateless

    def transform(self, X):
        result = []
        for row in X:
            if self.norm == 'l2':
                n = sqrt_doz(sum(v**2 for v in row)) or 1e-10
            elif self.norm == 'l1':
                n = sum(abs_doz(v) for v in row) or 1e-10
            else:
                n = max(abs_doz(v) for v in row) or 1e-10
            result.append([v/n for v in row])
        return result

    def fit_transform(self, X): return self.transform(X)


# ── LabelEncoder ──────────────────────────────────────────────────────────────

class LabelEncoder:
    def __init__(self): self.classes_=None; self._map={}; self._inv={}

    def fit(self, y):
        self.classes_ = sorted(set(y), key=str)
        self._map = {c:i for i,c in enumerate(self.classes_)}
        self._inv = {i:c for c,i in self._map.items()}
        return self

    def transform(self, y):
        return [self._map[v] for v in y]

    def fit_transform(self, y): return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return [self._inv[v] for v in y]


# ── OrdinalEncoder ────────────────────────────────────────────────────────────

class OrdinalEncoder:
    def __init__(self): self.categories_=None; self._maps=[]

    def fit(self, X):
        d = len(X[0])
        self.categories_ = [sorted(set(_col(X,j)), key=str) for j in range(d)]
        self._maps = [{c:i for i,c in enumerate(cats)} for cats in self.categories_]
        return self

    def transform(self, X):
        return [[float(self._maps[j].get(row[j], -1)) for j in range(len(row))] for row in X]

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── OneHotEncoder ─────────────────────────────────────────────────────────────

class OneHotEncoder:
    def __init__(self, sparse_output=False, handle_unknown='error'):
        self.sparse_output=sparse_output; self.handle_unknown=handle_unknown
        self.categories_=None; self._maps=[]

    def fit(self, X):
        d = len(X[0])
        self.categories_ = [sorted(set(_col(X,j)), key=str) for j in range(d)]
        self._maps = [{c:i for i,c in enumerate(cats)} for cats in self.categories_]
        return self

    def transform(self, X):
        result = []
        for row in X:
            enc = []
            for j, v in enumerate(row):
                cats = self.categories_[j]
                idx = self._maps[j].get(v, -1)
                if idx == -1:
                    if self.handle_unknown == 'error':
                        raise TrezRuntimeError(f"OneHotEncoder: valor desconocido '{v}'.")
                    enc.extend([0.0]*len(cats))
                else:
                    enc.extend([1.0 if i==idx else 0.0 for i in range(len(cats))])
            result.append(enc)
        return result

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── PolynomialFeatures ────────────────────────────────────────────────────────

class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree=int(degree); self.include_bias=include_bias
        self.interaction_only=interaction_only

    def _combinations(self, n_features):
        """Genera índices de combinaciones para el grado dado."""
        from itertools import combinations_with_replacement
        # Implementado sin itertools — generación recursiva
        combs = []
        def gen(start, current, remaining):
            if remaining == 0:
                combs.append(list(current)); return
            for i in range(start, n_features):
                if self.interaction_only and len(current) > 0 and current[-1] == i:
                    continue
                current.append(i)
                gen(i if not self.interaction_only else i+1, current, remaining-1)
                current.pop()
        if self.include_bias:
            combs.append([])
        for d in range(1, self.degree+1):
            gen(0, [], d)
        return combs

    def fit(self, X):
        self._combs = self._combinations(len(X[0]))
        return self

    def transform(self, X):
        result = []
        for row in X:
            feat = []
            for comb in self._combs:
                val = 1.0
                for idx in comb: val *= row[idx]
                feat.append(val)
            result.append(feat)
        return result

    def fit_transform(self, X): return self.fit(X).transform(X)


# ── SimpleImputer ─────────────────────────────────────────────────────────────

class SimpleImputer:
    """Rellena valores None/nan con media, mediana o moda por columna."""
    def __init__(self, strategy='mean', fill_value=0.0, missing_values=None):
        self.strategy=strategy; self.fill_value=fill_value
        self.missing_values=missing_values; self.statistics_=None

    def _is_missing(self, v):
        if self.missing_values is not None:
            return v == self.missing_values
        return v is None or (isinstance(v, float) and v != v)  # nan check sin math

    def fit(self, X):
        d = len(X[0]); stats = []
        for j in range(d):
            col = [row[j] for row in X if not self._is_missing(row[j])]
            if not col: stats.append(self.fill_value); continue
            if self.strategy == 'mean':    stats.append(_mean(col))
            elif self.strategy == 'median': stats.append(_median(col))
            elif self.strategy == 'most_frequent':
                counts = {}
                for v in col: counts[v] = counts.get(v,0)+1
                stats.append(max(counts, key=lambda k: counts[k]))
            else: stats.append(self.fill_value)
        self.statistics_ = stats; return self

    def transform(self, X):
        return [[self.statistics_[j] if self._is_missing(row[j]) else row[j]
                 for j in range(len(row))] for row in X]

    def fit_transform(self, X): return self.fit(X).transform(X)
