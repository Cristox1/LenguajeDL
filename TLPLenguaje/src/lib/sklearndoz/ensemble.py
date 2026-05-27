"""
sklearndoz.ensemble — Métodos de ensemble, Python puro.

Cubre: RandomForestClassifier, RandomForestRegressor,
       ExtraTreesClassifier, ExtraTreesRegressor,
       AdaBoostClassifier, AdaBoostRegressor,
       GradientBoostingClassifier, GradientBoostingRegressor,
       BaggingClassifier, BaggingRegressor.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mldoz.tree import tree_clf_fit, tree_reg_fit, tree_predict, _predict_one
from lib.randomdoz.randomdoz import sample as _sample, randint as _randint, random as _random, shuffle as _shuffle
from lib.mathdoz.core_mathdoz import exp_doz, log_doz, sqrt_doz
from errors import TrezRuntimeError


def _moda(vals):
    counts = {}
    for v in vals: counts[v] = counts.get(v, 0) + 1
    return max(counts, key=lambda k: counts[k])

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _bootstrap(X, y):
    n = len(X)
    idx = [_randint(0, n-1) for _ in range(n)]
    return [X[i] for i in idx], [y[i] for i in idx]

def _feature_subset(X, max_features):
    d = len(X[0])
    k = max(1, int(max_features * d) if isinstance(max_features, float) else int(max_features))
    k = min(k, d)
    cols = _sample(list(range(d)), k)
    return [[row[c] for c in cols] for row in X], cols

def _apply_cols(X, cols):
    return [[row[c] for c in cols] for row in X]


# ── RandomForestClassifier ────────────────────────────────────────────────────

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt',
                 min_samples_split=2, min_samples_leaf=1):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth else 20
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.estimators_ = []
        self.feature_indices_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.estimators_ = []; self.feature_indices_ = []
        d = len(X[0])
        mf = int(sqrt_doz(d)) if self.max_features == 'sqrt' else (d if self.max_features in ('auto', None) else int(self.max_features))
        for _ in range(self.n_estimators):
            Xb, yb = _bootstrap(X, y)
            cols = _sample(list(range(d)), min(mf, d))
            Xs = _apply_cols(Xb, cols)
            tree = tree_clf_fit(Xs, yb, self.max_depth)
            self.estimators_.append(tree)
            self.feature_indices_.append(cols)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        votes = []
        for tree, cols in zip(self.estimators_, self.feature_indices_):
            preds = tree_predict(tree, _apply_cols(X, cols))
            votes.append(preds)
        return [_moda([votes[t][i] for t in range(self.n_estimators)]) for i in range(len(X))]

    def score(self, X, y):
        yp = self.predict(X)
        return sum(1 for a,b in zip(y,yp) if a==b)/len(y)


# ── RandomForestRegressor ─────────────────────────────────────────────────────

class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth else 20
        self.max_features = max_features
        self.estimators_ = []; self.feature_indices_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        d = len(X[0])
        mf = int(sqrt_doz(d)) if self.max_features == 'sqrt' else (d if self.max_features in ('auto', None) else int(self.max_features))
        for _ in range(self.n_estimators):
            Xb, yb = _bootstrap(X, y)
            cols = _sample(list(range(d)), min(mf, d))
            tree = tree_reg_fit(_apply_cols(Xb, cols), yb, self.max_depth)
            self.estimators_.append(tree)
            self.feature_indices_.append(cols)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        preds = []
        for tree, cols in zip(self.estimators_, self.feature_indices_):
            preds.append(tree_predict(tree, _apply_cols(X, cols)))
        return [_mean([preds[t][i] for t in range(self.n_estimators)]) for i in range(len(X))]

    def score(self, X, y):
        yp = self.predict(X); m = _mean(y)
        ss_res = sum((y[i]-yp[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i]-m)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot else 0.0


# ── ExtraTreesClassifier ──────────────────────────────────────────────────────
# Igual que RandomForest pero sin bootstrap (usa todo el dataset)

class ExtraTreesClassifier:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth else 20
        self.max_features = max_features
        self.estimators_ = []; self.feature_indices_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        d = len(X[0])
        mf = int(sqrt_doz(d)) if self.max_features == 'sqrt' else d
        for _ in range(self.n_estimators):
            cols = _sample(list(range(d)), min(mf, d))
            tree = tree_clf_fit(_apply_cols(X, cols), y, self.max_depth)
            self.estimators_.append(tree); self.feature_indices_.append(cols)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        votes = [[tree_predict(t, _apply_cols(X, c))[i]
                  for t, c in zip(self.estimators_, self.feature_indices_)]
                 for i in range(len(X))]
        return [_moda(v) for v in votes]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


class ExtraTreesRegressor:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt'):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth else 20
        self.max_features = max_features
        self.estimators_ = []; self.feature_indices_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        d = len(X[0])
        mf = int(sqrt_doz(d)) if self.max_features == 'sqrt' else d
        for _ in range(self.n_estimators):
            cols = _sample(list(range(d)), min(mf, d))
            tree = tree_reg_fit(_apply_cols(X, cols), y, self.max_depth)
            self.estimators_.append(tree); self.feature_indices_.append(cols)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        preds_all = [[tree_predict(t, _apply_cols(X, c))[i]
                      for t, c in zip(self.estimators_, self.feature_indices_)]
                     for i in range(len(X))]
        return [_mean(p) for p in preds_all]

    def score(self, X, y):
        yp = self.predict(X); m = _mean(y)
        ss_res = sum((y[i]-yp[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i]-m)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot else 0.0


# ── AdaBoostClassifier ────────────────────────────────────────────────────────

class AdaBoostClassifier:
    """AdaBoost SAMME con árboles de profundidad 1 (decision stumps)."""
    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=1):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.max_depth = int(max_depth)
        self.estimators_ = []; self.alphas_ = []; self.classes_ = None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X)
        self.classes_ = sorted(set(y))
        K = len(self.classes_)
        w = [1.0/n]*n
        self.estimators_ = []; self.alphas_ = []
        for _ in range(self.n_estimators):
            # Muestra ponderada: replica índices proporcional a w
            total = sum(w); w_norm = [wi/total for wi in w]
            idx = []
            for _ in range(n):
                r = _random(); cum = 0.0
                for j, wj in enumerate(w_norm):
                    cum += wj
                    if r <= cum: idx.append(j); break
                else: idx.append(n-1)
            Xs = [X[i] for i in idx]; ys = [y[i] for i in idx]
            tree = tree_clf_fit(Xs, ys, self.max_depth)
            preds = tree_predict(tree, X)
            err = sum(w[i] for i in range(n) if preds[i] != y[i])
            err = max(err, 1e-10)
            if err >= 1 - 1.0/K: break
            alpha = self.learning_rate * (log_doz((1-err)/err) + log_doz(K-1))
            w = [w[i]*exp_doz(alpha*(0 if preds[i]==y[i] else 1)) for i in range(n)]
            self.estimators_.append(tree); self.alphas_.append(alpha)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        scores = {c: [0.0]*len(X) for c in self.classes_}
        for tree, alpha in zip(self.estimators_, self.alphas_):
            preds = tree_predict(tree, X)
            for i, p in enumerate(preds): scores[p][i] += alpha
        return [max(self.classes_, key=lambda c: scores[c][i]) for i in range(len(X))]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


# ── AdaBoostRegressor ─────────────────────────────────────────────────────────

class AdaBoostRegressor:
    """AdaBoost.R2 para regresión."""
    def __init__(self, n_estimators=50, learning_rate=1.0, max_depth=3):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.max_depth = int(max_depth)
        self.estimators_ = []; self.weights_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); w = [1.0/n]*n
        self.estimators_ = []; self.weights_ = []
        for _ in range(self.n_estimators):
            total = sum(w); w_norm = [wi/total for wi in w]
            idx = []
            for _ in range(n):
                r = _random(); cum = 0.0
                for j, wj in enumerate(w_norm):
                    cum += wj
                    if r <= cum: idx.append(j); break
                else: idx.append(n-1)
            Xs = [X[i] for i in idx]; ys = [y[i] for i in idx]
            tree = tree_reg_fit(Xs, ys, self.max_depth)
            preds = tree_predict(tree, X)
            max_err = max(abs(preds[i]-y[i]) for i in range(n)) or 1e-10
            loss = [abs(preds[i]-y[i])/max_err for i in range(n)]
            err = sum(w[i]*loss[i] for i in range(n))
            if err >= 0.5: break
            beta = err/(1-err+1e-10)
            w = [w[i]*beta**(self.learning_rate*(1-loss[i])) for i in range(n)]
            self.estimators_.append(tree); self.weights_.append(log_doz(1.0/(beta+1e-10)))
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        if not self.estimators_: return [0.0]*len(X)
        all_preds = [tree_predict(t, X) for t in self.estimators_]
        total_w = sum(self.weights_)
        return [sum(self.weights_[t]*all_preds[t][i] for t in range(len(self.estimators_)))/total_w for i in range(len(X))]

    def score(self, X, y):
        yp = self.predict(X); m = _mean(y)
        ss_res = sum((y[i]-yp[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i]-m)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot else 0.0


# ── GradientBoostingClassifier ────────────────────────────────────────────────

class GradientBoostingClassifier:
    """Gradient Boosting binario (log loss) con árboles de regresión."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = int(n_estimators)
        self.lr = learning_rate
        self.max_depth = int(max_depth)
        self.trees_ = []; self.classes_ = None; self.F0_ = 0.0

    def _sigmoid(self, x):
        if x >= 0: return 1.0/(1.0+exp_doz(-x))
        ex = exp_doz(x); return ex/(1.0+ex)

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y))
        if len(self.classes_) != 2:
            raise TrezRuntimeError("GradientBoostingClassifier: solo binario por ahora.")
        cm = {self.classes_[0]:0.0, self.classes_[1]:1.0}
        yb = [cm[v] for v in y]; n = len(X)
        p = sum(yb)/n; self.F0_ = log_doz(p/(1-p+1e-10)+1e-10)
        F = [self.F0_]*n; self.trees_ = []
        for _ in range(self.n_estimators):
            probs = [self._sigmoid(f) for f in F]
            resid = [yb[i]-probs[i] for i in range(n)]
            tree = tree_reg_fit(X, resid, self.max_depth)
            upd = tree_predict(tree, X)
            F = [F[i]+self.lr*upd[i] for i in range(n)]
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        F = [self.F0_]*len(X)
        for tree in self.trees_:
            upd = tree_predict(tree, X)
            F = [F[i]+self.lr*upd[i] for i in range(len(X))]
        p1 = [self._sigmoid(f) for f in F]
        return [[1-p, p] for p in p1]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[1] if p[1]>=0.5 else self.classes_[0] for p in proba]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


# ── GradientBoostingRegressor ─────────────────────────────────────────────────

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = int(n_estimators)
        self.lr = learning_rate
        self.max_depth = int(max_depth)
        self.trees_ = []; self.F0_ = 0.0

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        n = len(X); self.F0_ = _mean(y)
        F = [self.F0_]*n; self.trees_ = []
        for _ in range(self.n_estimators):
            resid = [y[i]-F[i] for i in range(n)]
            tree = tree_reg_fit(X, resid, self.max_depth)
            upd = tree_predict(tree, X)
            F = [F[i]+self.lr*upd[i] for i in range(n)]
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        F = [self.F0_]*len(X)
        for tree in self.trees_:
            upd = tree_predict(tree, X)
            F = [F[i]+self.lr*upd[i] for i in range(len(X))]
        return F

    def score(self, X, y):
        yp = self.predict(X); m = _mean(y)
        ss_res = sum((y[i]-yp[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i]-m)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot else 0.0


# ── BaggingClassifier / BaggingRegressor ─────────────────────────────────────

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.estimators_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            Xb, yb = _bootstrap(X, y)
            self.estimators_.append(tree_clf_fit(Xb, yb, self.max_depth))
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        votes = [tree_predict(t, X) for t in self.estimators_]
        return [_moda([votes[t][i] for t in range(self.n_estimators)]) for i in range(len(X))]

    def score(self, X, y):
        return sum(1 for a,b in zip(y,self.predict(X)) if a==b)/len(y)


class BaggingRegressor:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.estimators_ = []

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            Xb, yb = _bootstrap(X, y)
            self.estimators_.append(tree_reg_fit(Xb, yb, self.max_depth))
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        preds = [tree_predict(t, X) for t in self.estimators_]
        return [_mean([preds[t][i] for t in range(self.n_estimators)]) for i in range(len(X))]

    def score(self, X, y):
        yp = self.predict(X); m = _mean(y)
        ss_res = sum((y[i]-yp[i])**2 for i in range(len(y)))
        ss_tot = sum((y[i]-m)**2 for i in range(len(y)))
        return 1 - ss_res/ss_tot if ss_tot else 0.0
