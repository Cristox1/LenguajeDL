"""
sklearndoz.model_selection — Selección de modelos, Python puro.

Cubre: train_test_split, KFold, StratifiedKFold, LeaveOneOut,
       ShuffleSplit, cross_val_score, cross_validate,
       GridSearchCV, RandomizedSearchCV,
       learning_curve, validation_curve.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.randomdoz.randomdoz import shuffle as _shuffle, sample as _sample, random as _random
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0

def _score(estimator, X, y):
    return estimator.score(X, y)


# ── train_test_split ──────────────────────────────────────────────────────────

def train_test_split(*arrays, test_size=0.2, shuffle=True, random_state=None, stratify=None):
    if not arrays: raise TrezRuntimeError("train_test_split: se necesita al menos un array.")
    n = len(arrays[0])
    idx = list(range(n))
    if shuffle: _shuffle(idx)
    n_test = max(1, int(n*test_size))
    n_train = n - n_test
    train_idx = idx[:n_train]; test_idx = idx[n_train:]
    result = []
    for arr in arrays:
        result.append([arr[i] for i in train_idx])
        result.append([arr[i] for i in test_idx])
    return result


# ── KFold ─────────────────────────────────────────────────────────────────────

class KFold:
    def __init__(self, n_splits=5, shuffle=False):
        self.n_splits=int(n_splits); self.shuffle=shuffle

    def split(self, X):
        n = len(X); idx = list(range(n))
        if self.shuffle: _shuffle(idx)
        fold_size = n // self.n_splits
        folds = []
        for k in range(self.n_splits):
            start = k*fold_size
            end = start+fold_size if k<self.n_splits-1 else n
            test = idx[start:end]
            train = idx[:start]+idx[end:]
            folds.append((train, test))
        return folds

    def get_n_splits(self): return self.n_splits


# ── StratifiedKFold ───────────────────────────────────────────────────────────

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False):
        self.n_splits=int(n_splits); self.shuffle=shuffle

    def split(self, X, y):
        classes = sorted(set(y))
        class_idx = {c: [i for i,v in enumerate(y) if v==c] for c in classes}
        if self.shuffle:
            for idx in class_idx.values(): _shuffle(idx)
        # Distribuir cada clase entre los folds
        folds = [[] for _ in range(self.n_splits)]
        for c in classes:
            cIdx = class_idx[c]; n = len(cIdx)
            for k in range(self.n_splits):
                start = k*n//self.n_splits
                end = (k+1)*n//self.n_splits
                folds[k].extend(cIdx[start:end])
        result = []
        for k in range(self.n_splits):
            test = folds[k]
            train = [i for j,f in enumerate(folds) if j!=k for i in f]
            result.append((train, test))
        return result

    def get_n_splits(self): return self.n_splits


# ── LeaveOneOut ───────────────────────────────────────────────────────────────

class LeaveOneOut:
    def split(self, X):
        n = len(X)
        return [([j for j in range(n) if j!=i], [i]) for i in range(n)]

    def get_n_splits(self, X): return len(X)


# ── ShuffleSplit ──────────────────────────────────────────────────────────────

class ShuffleSplit:
    def __init__(self, n_splits=10, test_size=0.1):
        self.n_splits=int(n_splits); self.test_size=test_size

    def split(self, X):
        n = len(X); n_test = max(1, int(n*self.test_size))
        result = []
        for _ in range(self.n_splits):
            idx = list(range(n)); _shuffle(idx)
            result.append((idx[n_test:], idx[:n_test]))
        return result


# ── TimeSeriesSplit ───────────────────────────────────────────────────────────

class TimeSeriesSplit:
    def __init__(self, n_splits=5, gap=0):
        self.n_splits=int(n_splits); self.gap=int(gap)

    def split(self, X):
        n = len(X); fold_size = n//(self.n_splits+1)
        result = []
        for k in range(1, self.n_splits+1):
            train_end = k*fold_size
            test_start = train_end+self.gap
            test_end = test_start+fold_size
            if test_end > n: test_end = n
            train = list(range(train_end))
            test = list(range(test_start, test_end))
            if test: result.append((train, test))
        return result


# ── cross_val_score ───────────────────────────────────────────────────────────

def cross_val_score(estimator, X, y, cv=5, scoring=None):
    if isinstance(cv, int):
        splitter = KFold(n_splits=cv, shuffle=True)
        splits = splitter.split(X)
    else:
        splits = cv.split(X, y) if hasattr(cv, 'split') else cv.split(X)

    scores = []
    for train_idx, test_idx in splits:
        X_tr = [X[i] for i in train_idx]; y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx];  y_te = [y[i] for i in test_idx]
        import copy
        est = copy.deepcopy(estimator)
        est.fit(X_tr, y_tr)
        if scoring is None:
            scores.append(est.score(X_te, y_te))
        else:
            preds = est.predict(X_te)
            scores.append(scoring(y_te, preds))
    return scores


def cross_validate(estimator, X, y, cv=5, scoring=None):
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
    return {'test_score': scores, 'mean': _mean(scores)}


# ── ParameterGrid ─────────────────────────────────────────────────────────────

class ParameterGrid:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        keys = list(self.param_grid.keys())
        vals = [self.param_grid[k] for k in keys]
        def _product(lists):
            if not lists: yield []; return
            for item in lists[0]:
                for rest in _product(lists[1:]):
                    yield [item]+rest
        for combo in _product(vals):
            yield dict(zip(keys, combo))

    def __len__(self):
        n = 1
        for v in self.param_grid.values(): n *= len(v)
        return n


# ── GridSearchCV ──────────────────────────────────────────────────────────────

class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True):
        self.estimator=estimator; self.param_grid=param_grid
        self.cv=cv; self.scoring=scoring; self.refit=refit
        self.best_params_=None; self.best_score_=None; self.best_estimator_=None
        self.cv_results_=[]

    def fit(self, X, y):
        import copy
        best_score = -float('inf'); best_params = None
        for params in ParameterGrid(self.param_grid):
            est = copy.deepcopy(self.estimator)
            for k,v in params.items(): setattr(est, k, v)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_s = _mean(scores)
            self.cv_results_.append({'params': params, 'mean_test_score': mean_s, 'scores': scores})
            if mean_s > best_score:
                best_score = mean_s; best_params = params
        self.best_score_ = best_score; self.best_params_ = best_params
        if self.refit:
            self.best_estimator_ = copy.deepcopy(self.estimator)
            for k,v in best_params.items(): setattr(self.best_estimator_, k, v)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X): return self.best_estimator_.predict(X)
    def score(self, X, y): return self.best_estimator_.score(X, y)


# ── RandomizedSearchCV ────────────────────────────────────────────────────────

class RandomizedSearchCV:
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring=None, refit=True):
        self.estimator=estimator; self.param_distributions=param_distributions
        self.n_iter=int(n_iter); self.cv=cv; self.scoring=scoring; self.refit=refit
        self.best_params_=None; self.best_score_=None; self.best_estimator_=None
        self.cv_results_=[]

    def _sample_params(self):
        params = {}
        for k, v in self.param_distributions.items():
            if isinstance(v, list):
                params[k] = v[int(_random()*len(v))]
            elif isinstance(v, tuple) and len(v)==2:
                lo, hi = v
                if isinstance(lo, int) and isinstance(hi, int):
                    params[k] = lo+int(_random()*(hi-lo+1))
                else:
                    params[k] = lo+_random()*(hi-lo)
            else:
                params[k] = v
        return params

    def fit(self, X, y):
        import copy
        best_score = -float('inf'); best_params = None
        for _ in range(self.n_iter):
            params = self._sample_params()
            est = copy.deepcopy(self.estimator)
            for k,v in params.items(): setattr(est, k, v)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_s = _mean(scores)
            self.cv_results_.append({'params': params, 'mean_test_score': mean_s})
            if mean_s > best_score:
                best_score = mean_s; best_params = params
        self.best_score_ = best_score; self.best_params_ = best_params
        if self.refit:
            self.best_estimator_ = copy.deepcopy(self.estimator)
            for k,v in best_params.items(): setattr(self.best_estimator_, k, v)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X): return self.best_estimator_.predict(X)
    def score(self, X, y): return self.best_estimator_.score(X, y)


# ── learning_curve ────────────────────────────────────────────────────────────

def learning_curve(estimator, X, y, train_sizes=None, cv=5, scoring=None):
    import copy
    if train_sizes is None:
        train_sizes = [0.1, 0.33, 0.55, 0.78, 1.0]
    n = len(X)
    splitter = KFold(n_splits=cv, shuffle=True)
    splits = splitter.split(X)
    train_scores = []; val_scores = []; sizes = []
    for frac in train_sizes:
        size = max(1, int(n*frac)) if frac<=1.0 else int(frac)
        ts_fold = []; vs_fold = []
        for train_idx, test_idx in splits:
            sub = train_idx[:size]
            X_tr=[X[i] for i in sub]; y_tr=[y[i] for i in sub]
            X_te=[X[i] for i in test_idx]; y_te=[y[i] for i in test_idx]
            est = copy.deepcopy(estimator); est.fit(X_tr, y_tr)
            ts_fold.append(est.score(X_tr, y_tr))
            vs_fold.append(est.score(X_te, y_te))
        sizes.append(size)
        train_scores.append(ts_fold)
        val_scores.append(vs_fold)
    return sizes, train_scores, val_scores


# ── validation_curve ──────────────────────────────────────────────────────────

def validation_curve(estimator, X, y, param_name, param_range, cv=5, scoring=None):
    import copy
    train_scores = []; val_scores = []
    splitter = KFold(n_splits=cv, shuffle=True)
    splits = splitter.split(X)
    for val in param_range:
        ts_fold = []; vs_fold = []
        for train_idx, test_idx in splits:
            X_tr=[X[i] for i in train_idx]; y_tr=[y[i] for i in train_idx]
            X_te=[X[i] for i in test_idx]; y_te=[y[i] for i in test_idx]
            est = copy.deepcopy(estimator); setattr(est, param_name, val)
            est.fit(X_tr, y_tr)
            ts_fold.append(est.score(X_tr, y_tr))
            vs_fold.append(est.score(X_te, y_te))
        train_scores.append(ts_fold)
        val_scores.append(vs_fold)
    return train_scores, val_scores
