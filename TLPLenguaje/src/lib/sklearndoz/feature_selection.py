"""
sklearndoz.feature_selection — Selección de características, Python puro.

Cubre: SelectKBest, VarianceThreshold, RFE, SelectFromModel,
       SelectPercentile, chi2, f_classif, mutual_info_classif.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, log_doz
from errors import TrezRuntimeError
import copy as _copy


def _mean(v): return sum(v)/len(v) if v else 0.0
def _var(v): m = _mean(v); return sum((x-m)**2 for x in v)/len(v) if v else 0.0
def _col(X, j): return [row[j] for row in X]


# ── Scoring functions ─────────────────────────────────────────────────────────

def chi2(X, y):
    """Chi-cuadrado para características no negativas."""
    classes = sorted(set(y)); n = len(X); d = len(X[0])
    scores = []
    for j in range(d):
        col = _col(X, j)
        vals = sorted(set(col))
        chi = 0.0
        for c in classes:
            mask = [i for i in range(n) if y[i]==c]
            for v in vals:
                observed = sum(1 for i in mask if col[i]==v)
                expected = len(mask)*sum(1 for x in col if x==v)/n
                if expected > 0:
                    chi += (observed-expected)**2/expected
        scores.append(chi)
    return scores, [None]*d


def f_classif(X, y):
    """F-estadístico ANOVA para clasificación."""
    classes = sorted(set(y)); n = len(X); d = len(X[0]); K = len(classes)
    scores = []
    for j in range(d):
        col = _col(X, j); overall_mean = _mean(col)
        groups = [[col[i] for i in range(n) if y[i]==c] for c in classes]
        # Between groups variance
        ss_b = sum(len(g)*(_mean(g)-overall_mean)**2 for g in groups if g)
        # Within groups variance
        ss_w = sum(sum((x-_mean(g))**2 for x in g) for g in groups if g)
        df_b = K-1; df_w = n-K
        if ss_w < 1e-10 or df_b == 0 or df_w <= 0:
            scores.append(0.0); continue
        scores.append((ss_b/df_b)/(ss_w/df_w))
    return scores, [None]*d


def f_regression(X, y):
    """F-estadístico para regresión."""
    n = len(X); d = len(X[0])
    y_mean = _mean(y)
    ss_tot = sum((yi-y_mean)**2 for yi in y) or 1e-10
    scores = []
    for j in range(d):
        col = _col(X, j)
        x_mean = _mean(col)
        cov = sum((col[i]-x_mean)*(y[i]-y_mean) for i in range(n))
        var_x = sum((col[i]-x_mean)**2 for i in range(n)) or 1e-10
        beta = cov/var_x
        y_pred = [beta*(col[i]-x_mean)+y_mean for i in range(n)]
        ss_reg = sum((pi-y_mean)**2 for pi in y_pred)
        ss_res = sum((y[i]-y_pred[i])**2 for i in range(n)) or 1e-10
        f = (ss_reg/1)/(ss_res/(n-2)) if n > 2 else 0.0
        scores.append(f)
    return scores, [None]*d


def mutual_info_classif(X, y, n_bins=10):
    """Información mutua (discretizada) para clasificación."""
    n = len(X); d = len(X[0]); classes = sorted(set(y))
    scores = []
    for j in range(d):
        col = _col(X, j)
        lo, hi = min(col), max(col)
        if hi == lo: scores.append(0.0); continue
        def _bin(v): return min(int((v-lo)/(hi-lo)*n_bins), n_bins-1)
        binned = [_bin(v) for v in col]
        mi = 0.0
        for b in range(n_bins):
            px = sum(1 for v in binned if v==b)/n
            if px < 1e-10: continue
            for c in classes:
                pxy = sum(1 for i in range(n) if binned[i]==b and y[i]==c)/n
                py = sum(1 for yi in y if yi==c)/n
                if pxy > 1e-10 and py > 1e-10:
                    mi += pxy*log_doz(pxy/(px*py))
        scores.append(mi)
    return scores, [None]*d


# ── VarianceThreshold ─────────────────────────────────────────────────────────

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold=threshold
        self.variances_=None; self._mask=None

    def fit(self, X, y=None):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        d = len(X[0])
        self.variances_ = [_var(_col(X,j)) for j in range(d)]
        self._mask = [v > self.threshold for v in self.variances_]
        return self

    def transform(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [[row[j] for j in range(len(row)) if self._mask[j]] for row in X]

    def fit_transform(self, X, y=None): return self.fit(X).transform(X)

    def get_support(self): return list(self._mask)

    def inverse_transform(self, X):
        d = len(self._mask); result = []
        for row in X:
            full = [0.0]*d; idx = [j for j in range(d) if self._mask[j]]
            for k, j in enumerate(idx): full[j] = row[k] if k < len(row) else 0.0
            result.append(full)
        return result


# ── SelectKBest ───────────────────────────────────────────────────────────────

class SelectKBest:
    def __init__(self, score_func=f_classif, k=10):
        self.score_func=score_func; self.k=k
        self.scores_=None; self._mask=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        scores, _ = self.score_func(X, y)
        self.scores_ = scores
        k = min(self.k, len(scores)) if self.k != 'all' else len(scores)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        top = set(ranked[:k])
        self._mask = [j in top for j in range(len(scores))]
        return self

    def transform(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [[row[j] for j in range(len(row)) if self._mask[j]] for row in X]

    def fit_transform(self, X, y): return self.fit(X, y).transform(X)

    def get_support(self): return list(self._mask)


# ── SelectPercentile ──────────────────────────────────────────────────────────

class SelectPercentile:
    def __init__(self, score_func=f_classif, percentile=10):
        self.score_func=score_func; self.percentile=percentile
        self.scores_=None; self._mask=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        scores, _ = self.score_func(X, y)
        self.scores_ = scores
        k = max(1, int(len(scores)*self.percentile/100))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        top = set(ranked[:k])
        self._mask = [j in top for j in range(len(scores))]
        return self

    def transform(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [[row[j] for j in range(len(row)) if self._mask[j]] for row in X]

    def fit_transform(self, X, y): return self.fit(X, y).transform(X)
    def get_support(self): return list(self._mask)


# ── RFE (Recursive Feature Elimination) ──────────────────────────────────────

class RFE:
    def __init__(self, estimator, n_features_to_select=None, step=1):
        self.estimator=estimator; self.n_features_to_select=n_features_to_select
        self.step=int(step)
        self.support_=None; self.ranking_=None; self.estimator_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        d = len(X[0]); n_sel = self.n_features_to_select or max(1, d//2)
        active = list(range(d)); ranking = [0]*d
        rank = 1
        while len(active) > n_sel:
            Xsub = [[row[j] for j in active] for row in X]
            est = _copy.deepcopy(self.estimator); est.fit(Xsub, y)
            # Importancias: coef_ si existe, sino ones
            if hasattr(est, 'coef_'):
                coefs = est.coef_[0] if isinstance(est.coef_[0], list) else est.coef_
                importances = [abs_doz(c) for c in coefs]
            elif hasattr(est, 'feature_importances_'):
                importances = list(est.feature_importances_)
            else:
                importances = [1.0]*len(active)
            n_remove = min(self.step, len(active)-n_sel)
            ranked_local = sorted(range(len(active)), key=lambda i: importances[i])
            to_remove = [active[ranked_local[i]] for i in range(n_remove)]
            for j in to_remove:
                ranking[j] = rank; active.remove(j)
            rank += 1
        Xsub = [[row[j] for j in active] for row in X]
        self.estimator_ = _copy.deepcopy(self.estimator); self.estimator_.fit(Xsub, y)
        for j in active: ranking[j] = 1
        self.ranking_ = ranking
        self.support_ = [j in active for j in range(d)]
        return self

    def transform(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [[row[j] for j in range(len(row)) if self.support_[j]] for row in X]

    def fit_transform(self, X, y): return self.fit(X, y).transform(X)

    def predict(self, X):
        Xt = self.transform(X); return self.estimator_.predict(Xt)

    def score(self, X, y):
        Xt = self.transform(X); return self.estimator_.score(Xt, y)

    def get_support(self): return list(self.support_)


# ── SelectFromModel ───────────────────────────────────────────────────────────

class SelectFromModel:
    def __init__(self, estimator, threshold=None, max_features=None, prefit=False):
        self.estimator=estimator; self.threshold=threshold
        self.max_features=max_features; self.prefit=prefit
        self._mask=None; self.estimator_=None

    def fit(self, X, y=None):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        if self.prefit:
            self.estimator_ = self.estimator
        else:
            self.estimator_ = _copy.deepcopy(self.estimator)
            if y is not None: self.estimator_.fit(X, y)
            else: self.estimator_.fit(X)
        if hasattr(self.estimator_, 'feature_importances_'):
            importances = list(self.estimator_.feature_importances_)
        elif hasattr(self.estimator_, 'coef_'):
            c = self.estimator_.coef_
            importances = [abs_doz(v) for v in (c[0] if isinstance(c[0], list) else c)]
        else:
            importances = [1.0]*len(X[0])
        threshold = self.threshold
        if threshold is None or threshold == 'mean':
            threshold = sum(importances)/len(importances)
        elif threshold == 'median':
            s = sorted(importances); n = len(s)
            threshold = (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]
        self._mask = [imp >= threshold for imp in importances]
        if self.max_features is not None:
            ranked = sorted(range(len(importances)), key=lambda i: -importances[i])
            top = set(ranked[:self.max_features])
            self._mask = [j in top and self._mask[j] for j in range(len(importances))]
        return self

    def transform(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [[row[j] for j in range(len(row)) if self._mask[j]] for row in X]

    def fit_transform(self, X, y=None): return self.fit(X, y).transform(X)
    def get_support(self): return list(self._mask)
