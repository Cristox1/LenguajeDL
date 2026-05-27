"""
sklearndoz.naive_bayes — Clasificadores Naive Bayes, Python puro.

Cubre: GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, exp_doz, log_doz
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0
def _var(v):
    m = _mean(v)
    return sum((x-m)**2 for x in v)/len(v) if v else 1.0
def _col(X, j): return [row[j] for row in X]


# ── GaussianNB ────────────────────────────────────────────────────────────────

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing=var_smoothing
        self.classes_=None; self.class_prior_=None
        self.theta_=None; self.var_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); n = len(X); d = len(X[0])
        self.class_prior_ = {c: sum(1 for yi in y if yi==c)/n for c in self.classes_}
        self.theta_ = {}; self.var_ = {}
        for c in self.classes_:
            Xc = [X[i] for i in range(n) if y[i]==c]
            self.theta_[c] = [_mean(_col(Xc,j)) for j in range(d)]
            self.var_[c] = [max(_var(_col(Xc,j)), self.var_smoothing) for j in range(d)]
        return self

    def _log_likelihood(self, x, c):
        ll = 0.0
        for j in range(len(x)):
            mu = self.theta_[c][j]; sig2 = self.var_[c][j]
            ll += -0.5*log_doz(2*3.14159265358979*sig2) - (x[j]-mu)**2/(2*sig2)
        return ll

    def predict_log_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            lps = [log_doz(self.class_prior_[c])+self._log_likelihood(x,c) for c in self.classes_]
            result.append(lps)
        return result

    def predict_proba(self, X):
        lps = self.predict_log_proba(X)
        result = []
        for lp in lps:
            mx = max(lp)
            probs = [exp_doz(min(v-mx, 700)) for v in lp]
            total = sum(probs) or 1e-10
            result.append([p/total for p in probs])
        return result

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[lp.index(max(lp))] for lp in lps]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)

    def partial_fit(self, X, y, classes=None):
        return self.fit(X, y)


# ── MultinomialNB ─────────────────────────────────────────────────────────────

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha=alpha
        self.classes_=None; self.class_log_prior_=None; self.feature_log_prob_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); n = len(X); d = len(X[0])
        class_counts = {c: sum(1 for yi in y if yi==c) for c in self.classes_}
        self.class_log_prior_ = {c: log_doz(class_counts[c]/n) for c in self.classes_}
        self.feature_log_prob_ = {}
        for c in self.classes_:
            Xc = [X[i] for i in range(n) if y[i]==c]
            col_sums = [sum(_col(Xc,j))+self.alpha for j in range(d)]
            total = sum(col_sums)
            self.feature_log_prob_[c] = [log_doz(s/total) for s in col_sums]
        return self

    def predict_log_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            lps = [self.class_log_prior_[c] + sum(x[j]*self.feature_log_prob_[c][j] for j in range(len(x))) for c in self.classes_]
            result.append(lps)
        return result

    def predict_proba(self, X):
        lps = self.predict_log_proba(X)
        result = []
        for lp in lps:
            mx = max(lp)
            probs = [exp_doz(min(v-mx, 700)) for v in lp]
            total = sum(probs) or 1e-10
            result.append([p/total for p in probs])
        return result

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[lp.index(max(lp))] for lp in lps]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── BernoulliNB ───────────────────────────────────────────────────────────────

class BernoulliNB:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha=alpha; self.binarize=binarize
        self.classes_=None; self.class_log_prior_=None; self.feature_log_prob_=None

    def _binarize(self, X):
        return [[1.0 if v > self.binarize else 0.0 for v in row] for row in X]

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        X = self._binarize(X)
        self.classes_ = sorted(set(y)); n = len(X); d = len(X[0])
        self.class_log_prior_ = {c: log_doz(sum(1 for yi in y if yi==c)/n) for c in self.classes_}
        self.feature_log_prob_ = {}
        self.feature_log_prob_neg_ = {}
        for c in self.classes_:
            Xc = [X[i] for i in range(n) if y[i]==c]
            nc = len(Xc)
            ps = [(sum(_col(Xc,j))+self.alpha)/(nc+2*self.alpha) for j in range(d)]
            self.feature_log_prob_[c] = [log_doz(p) for p in ps]
            self.feature_log_prob_neg_[c] = [log_doz(1-p) for p in ps]
        return self

    def predict_log_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        X = self._binarize(X)
        result = []
        for x in X:
            lps = [self.class_log_prior_[c] +
                   sum(x[j]*self.feature_log_prob_[c][j] + (1-x[j])*self.feature_log_prob_neg_[c][j]
                       for j in range(len(x))) for c in self.classes_]
            result.append(lps)
        return result

    def predict_proba(self, X):
        lps = self.predict_log_proba(X)
        result = []
        for lp in lps:
            mx = max(lp)
            probs = [exp_doz(min(v-mx, 700)) for v in lp]
            total = sum(probs) or 1e-10
            result.append([p/total for p in probs])
        return result

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[lp.index(max(lp))] for lp in lps]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── ComplementNB ──────────────────────────────────────────────────────────────

class ComplementNB:
    """Complement Naive Bayes — mejor para datos desbalanceados."""
    def __init__(self, alpha=1.0, norm=False):
        self.alpha=alpha; self.norm=norm
        self.classes_=None; self.class_log_prior_=None; self.feature_log_prob_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); n = len(X); d = len(X[0])
        self.class_log_prior_ = {c: log_doz(sum(1 for yi in y if yi==c)/n) for c in self.classes_}
        self.feature_log_prob_ = {}
        all_feat_sums = [sum(_col(X,j))+self.alpha*len(self.classes_) for j in range(d)]
        for c in self.classes_:
            Xc = [X[i] for i in range(n) if y[i]==c]
            comp_sums = [all_feat_sums[j] - (sum(_col(Xc,j))+self.alpha) for j in range(d)]
            total = sum(comp_sums) or 1e-10
            weights = [log_doz(s/total) for s in comp_sums]
            if self.norm:
                nw = sqrt_doz(sum(w**2 for w in weights)) or 1e-10
                weights = [w/nw for w in weights]
            self.feature_log_prob_[c] = [-w for w in weights]
        return self

    def predict_log_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            lps = [self.class_log_prior_[c] + sum(x[j]*self.feature_log_prob_[c][j] for j in range(len(x))) for c in self.classes_]
            result.append(lps)
        return result

    def predict_proba(self, X):
        lps = self.predict_log_proba(X)
        result = []
        for lp in lps:
            mx = max(lp)
            probs = [exp_doz(min(v-mx, 700)) for v in lp]
            total = sum(probs) or 1e-10
            result.append([p/total for p in probs])
        return result

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[lp.index(max(lp))] for lp in lps]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── CategoricalNB ─────────────────────────────────────────────────────────────

class CategoricalNB:
    """NB para características categóricas."""
    def __init__(self, alpha=1.0):
        self.alpha=alpha
        self.classes_=None; self.class_log_prior_=None
        self._cats=None; self._log_probs=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y)); n = len(X); d = len(X[0])
        self.class_log_prior_ = {c: log_doz(sum(1 for yi in y if yi==c)/n) for c in self.classes_}
        self._cats = [sorted(set(_col(X,j))) for j in range(d)]
        self._log_probs = {c: {} for c in self.classes_}
        for c in self.classes_:
            Xc = [X[i] for i in range(n) if y[i]==c]; nc = len(Xc)
            for j in range(d):
                cats = self._cats[j]; ncat = len(cats)
                counts = {cat: sum(1 for row in Xc if row[j]==cat)+self.alpha for cat in cats}
                total = nc + self.alpha*ncat
                self._log_probs[c][j] = {cat: log_doz(counts[cat]/total) for cat in cats}
        return self

    def predict_log_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        result = []
        for x in X:
            lps = []
            for c in self.classes_:
                lp = self.class_log_prior_[c]
                for j, v in enumerate(x):
                    lp += self._log_probs[c][j].get(v, log_doz(self.alpha/(1+self.alpha*len(self._cats[j]))))
                lps.append(lp)
            result.append(lps)
        return result

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[lp.index(max(lp))] for lp in lps]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)
