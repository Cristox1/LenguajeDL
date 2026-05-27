"""
sklearndoz.pipeline — Pipeline, FeatureUnion, ColumnTransformer, Python puro.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError
import copy as _copy


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, steps, memory=None):
        self.steps = steps  # list of (name, estimator)
        self.memory = memory
        self.named_steps = {name: est for name, est in steps}

    def _final_estimator(self): return self.steps[-1][1]
    def _transformers(self): return self.steps[:-1]

    def fit(self, X, y=None):
        Xt = X
        for name, step in self._transformers():
            if y is not None and hasattr(step, 'fit_transform'):
                try: Xt = step.fit_transform(Xt, y)
                except TypeError: Xt = step.fit_transform(Xt)
            elif hasattr(step, 'fit_transform'):
                Xt = step.fit_transform(Xt)
            else:
                step.fit(Xt, y) if y is not None else step.fit(Xt)
                Xt = step.transform(Xt)
        fe = self._final_estimator()
        if y is not None: fe.fit(Xt, y)
        else: fe.fit(Xt)
        return self

    def fit_transform(self, X, y=None):
        Xt = X
        for name, step in self.steps:
            if y is not None and hasattr(step, 'fit_transform'):
                try: Xt = step.fit_transform(Xt, y)
                except TypeError: Xt = step.fit_transform(Xt)
            elif hasattr(step, 'fit_transform'):
                Xt = step.fit_transform(Xt)
            else:
                step.fit(Xt); Xt = step.transform(Xt)
        return Xt

    def transform(self, X):
        Xt = X
        for name, step in self._transformers():
            Xt = step.transform(Xt)
        fe = self._final_estimator()
        if hasattr(fe, 'transform'): return fe.transform(Xt)
        raise TrezRuntimeError("Pipeline: último paso no es transformador.")

    def predict(self, X):
        Xt = X
        for name, step in self._transformers():
            Xt = step.transform(Xt)
        return self._final_estimator().predict(Xt)

    def predict_proba(self, X):
        Xt = X
        for name, step in self._transformers():
            Xt = step.transform(Xt)
        return self._final_estimator().predict_proba(Xt)

    def score(self, X, y):
        Xt = X
        for name, step in self._transformers():
            Xt = step.transform(Xt)
        return self._final_estimator().score(Xt, y)

    def set_params(self, **params):
        for k, v in params.items():
            if '__' in k:
                step_name, param = k.split('__', 1)
                setattr(self.named_steps[step_name], param, v)
            else:
                setattr(self, k, v)
        return self

    def get_params(self, deep=True):
        params = {}
        for name, step in self.steps:
            params[name] = step
            if deep and hasattr(step, '__dict__'):
                for k, v in step.__dict__.items():
                    params[f"{name}__{k}"] = v
        return params

    def __getitem__(self, key):
        if isinstance(key, str): return self.named_steps[key]
        return self.steps[key][1]


def make_pipeline(*steps):
    named = [(f"step{i}", s) for i, s in enumerate(steps)]
    return Pipeline(named)


# ── FeatureUnion ──────────────────────────────────────────────────────────────

class FeatureUnion:
    def __init__(self, transformer_list, weights=None):
        self.transformer_list = transformer_list
        self.weights = weights
        self.named_transformers_ = {n: t for n, t in transformer_list}

    def fit(self, X, y=None):
        for name, t in self.transformer_list:
            if y is not None:
                try: t.fit(X, y)
                except TypeError: t.fit(X)
            else: t.fit(X)
        return self

    def transform(self, X):
        results = []
        for i, (name, t) in enumerate(self.transformer_list):
            Xt = t.transform(X)
            if self.weights and i < len(self.weights):
                w = self.weights[i]
                Xt = [[v*w for v in row] for row in Xt]
            results.append(Xt)
        # Concatenar columnas
        if not results: return X
        return [sum([r[i] for r in results], []) for i in range(len(results[0]))]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ── ColumnTransformer ─────────────────────────────────────────────────────────

class ColumnTransformer:
    """Aplica transformaciones diferentes a subconjuntos de columnas."""
    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3):
        self.transformers = transformers  # list of (name, transformer, cols)
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self._fitted = []

    def _get_cols(self, X, cols):
        if isinstance(cols, list):
            return [[X[i][c] for c in cols] for i in range(len(X))]
        return [[X[i][cols]] for i in range(len(X))]

    def fit(self, X, y=None):
        self._fitted = []
        all_cols = set()
        for name, t, cols in self.transformers:
            cols_list = cols if isinstance(cols, list) else [cols]
            for c in cols_list: all_cols.add(c)
            Xsub = self._get_cols(X, cols)
            t2 = _copy.deepcopy(t)
            if y is not None:
                try: t2.fit(Xsub, y)
                except TypeError: t2.fit(Xsub)
            else: t2.fit(Xsub)
            self._fitted.append((name, t2, cols))
        d = len(X[0])
        if self.remainder == 'passthrough':
            self._remainder_cols = [j for j in range(d) if j not in all_cols]
        else:
            self._remainder_cols = []
        return self

    def transform(self, X):
        parts = []
        for name, t, cols in self._fitted:
            Xsub = self._get_cols(X, cols)
            parts.append(t.transform(Xsub))
        if self._remainder_cols:
            parts.append(self._get_cols(X, self._remainder_cols))
        if not parts: return [[] for _ in X]
        return [sum([p[i] for p in parts], []) for i in range(len(parts[0]))]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ── make_column_transformer ───────────────────────────────────────────────────

def make_column_transformer(*transformers, **kwargs):
    named = [(f"transformer{i}", t, c) for i, (t, c) in enumerate(transformers)]
    return ColumnTransformer(named, **kwargs)
