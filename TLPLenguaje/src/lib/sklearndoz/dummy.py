"""
sklearndoz.dummy — Clasificadores/regresores baseline, Python puro.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.randomdoz.randomdoz import random as _random
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0
def _mode(v):
    counts = {}
    for x in v: counts[x] = counts.get(x,0)+1
    return max(counts, key=lambda k: counts[k])


# ── DummyClassifier ───────────────────────────────────────────────────────────

class DummyClassifier:
    """Clasificador baseline: most_frequent, stratified, uniform, constant."""
    def __init__(self, strategy='most_frequent', constant=None):
        self.strategy=strategy; self.constant=constant
        self.classes_=None; self.class_prior_=None; self._most_frequent=None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = sorted(set(y)); n = len(y)
        self.class_prior_ = {c: sum(1 for yi in y if yi==c)/n for c in self.classes_}
        self._most_frequent = _mode(y)
        return self

    def predict(self, X):
        n = len(X)
        if self.strategy == 'most_frequent':
            return [self._most_frequent]*n
        elif self.strategy == 'constant':
            return [self.constant]*n
        elif self.strategy == 'uniform':
            return [self.classes_[int(_random()*len(self.classes_))] for _ in range(n)]
        elif self.strategy == 'stratified':
            # Sample proportional to prior
            result = []
            for _ in range(n):
                r = _random(); cum = 0.0
                for c in self.classes_:
                    cum += self.class_prior_[c]
                    if r <= cum: result.append(c); break
                else: result.append(self.classes_[-1])
            return result
        else:
            return [self._most_frequent]*n

    def predict_proba(self, X):
        K = len(self.classes_); n = len(X)
        if self.strategy == 'uniform':
            return [[1.0/K]*K for _ in range(n)]
        elif self.strategy == 'stratified' or self.strategy == 'most_frequent':
            prior = [self.class_prior_[c] for c in self.classes_]
            return [list(prior) for _ in range(n)]
        elif self.strategy == 'constant':
            idx = self.classes_.index(self.constant) if self.constant in self.classes_ else 0
            row = [0.0]*K; row[idx] = 1.0
            return [list(row) for _ in range(n)]
        return [[1.0/K]*K for _ in range(n)]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── DummyRegressor ────────────────────────────────────────────────────────────

class DummyRegressor:
    """Regresor baseline: mean, median, quantile, constant."""
    def __init__(self, strategy='mean', constant=None, quantile=None):
        self.strategy=strategy; self.constant=constant; self.quantile=quantile
        self._value=None

    def fit(self, X, y, sample_weight=None):
        y_list = list(y)
        if self.strategy == 'mean':
            self._value = _mean(y_list)
        elif self.strategy == 'median':
            s = sorted(y_list); n = len(s)
            self._value = (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]
        elif self.strategy == 'quantile':
            q = self.quantile or 0.5
            s = sorted(y_list); n = len(s)
            idx = q*(n-1); lo = int(idx); hi = min(lo+1, n-1)
            self._value = s[lo]+(s[hi]-s[lo])*(idx-lo)
        elif self.strategy == 'constant':
            self._value = self.constant
        else:
            self._value = _mean(y_list)
        return self

    def predict(self, X):
        return [self._value]*len(X)

    def score(self, X, y):
        preds = self.predict(X)
        m = _mean(list(y))
        ss_res = sum((yi-pi)**2 for yi,pi in zip(y,preds))
        ss_tot = sum((yi-m)**2 for yi in y) or 1e-10
        return 1-ss_res/ss_tot
