"""
sklearndoz.multiclass — Estrategias multiclase, Python puro.

Cubre: OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError
import copy as _copy


# ── OneVsRestClassifier ───────────────────────────────────────────────────────

class OneVsRestClassifier:
    def __init__(self, estimator):
        self.estimator=estimator
        self.classes_=None; self.estimators_=[]

    def fit(self, X, y):
        self.classes_ = sorted(set(y)); self.estimators_ = []
        for c in self.classes_:
            yb = [1 if yi==c else 0 for yi in y]
            est = _copy.deepcopy(self.estimator)
            est.fit(X, yb)
            self.estimators_.append(est)
        return self

    def predict(self, X):
        # Puntaje para cada clase, elige max
        scores = []
        for est in self.estimators_:
            if hasattr(est, 'predict_proba'):
                prob = est.predict_proba(X)
                # Probabilidad de clase positiva
                pos_idx = list(est.classes_).index(1) if hasattr(est, 'classes_') else 1
                scores.append([p[pos_idx] if isinstance(p, list) else p for p in prob])
            elif hasattr(est, 'decision_function'):
                df = est.decision_function(X)
                scores.append(df if isinstance(df[0], (int,float)) else [d[0] for d in df])
            else:
                preds = est.predict(X)
                scores.append([float(p) for p in preds])
        n = len(X)
        return [self.classes_[max(range(len(self.classes_)), key=lambda k: scores[k][i])] for i in range(n)]

    def predict_proba(self, X):
        probs = []
        for est in self.estimators_:
            if hasattr(est, 'predict_proba'):
                prob = est.predict_proba(X)
                pos_idx = list(est.classes_).index(1) if hasattr(est, 'classes_') else 1
                probs.append([p[pos_idx] if isinstance(p, list) else p for p in prob])
            else:
                preds = est.predict(X)
                probs.append([float(p) for p in preds])
        n = len(X); K = len(self.classes_)
        result = []
        for i in range(n):
            raw = [probs[k][i] for k in range(K)]
            total = sum(raw) or 1e-10
            result.append([r/total for r in raw])
        return result

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── OneVsOneClassifier ────────────────────────────────────────────────────────

class OneVsOneClassifier:
    def __init__(self, estimator):
        self.estimator=estimator
        self.classes_=None; self._pairs=[]; self._estimators=[]

    def fit(self, X, y):
        self.classes_ = sorted(set(y)); K = len(self.classes_)
        self._pairs = []; self._estimators = []
        for i in range(K):
            for j in range(i+1, K):
                ci, cj = self.classes_[i], self.classes_[j]
                mask = [k for k in range(len(y)) if y[k]==ci or y[k]==cj]
                Xsub = [X[k] for k in mask]; ysub = [y[k] for k in mask]
                est = _copy.deepcopy(self.estimator); est.fit(Xsub, ysub)
                self._pairs.append((ci, cj)); self._estimators.append(est)
        return self

    def predict(self, X):
        n = len(X); K = len(self.classes_)
        votes = [{c: 0 for c in self.classes_} for _ in range(n)]
        for (ci, cj), est in zip(self._pairs, self._estimators):
            preds = est.predict(X)
            for i, p in enumerate(preds):
                if p == ci: votes[i][ci] += 1
                elif p == cj: votes[i][cj] += 1
        return [max(self.classes_, key=lambda c: votes[i][c]) for i in range(n)]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)


# ── OutputCodeClassifier ──────────────────────────────────────────────────────

class OutputCodeClassifier:
    """Error-correcting output codes (código aleatorio)."""
    def __init__(self, estimator, code_size=1.5, random_state=None):
        self.estimator=estimator; self.code_size=code_size
        self.classes_=None; self._code_book=[]; self._estimators=[]

    def fit(self, X, y):
        from lib.randomdoz.randomdoz import random as _rng
        self.classes_ = sorted(set(y)); K = len(self.classes_)
        n_codes = max(int(K*self.code_size), K)
        # Código aleatorio binario
        self._code_book = [[1 if _rng()>0.5 else 0 for _ in range(n_codes)] for _ in range(K)]
        cls_idx = {c:i for i,c in enumerate(self.classes_)}
        self._estimators = []
        for bit in range(n_codes):
            yb = [self._code_book[cls_idx[yi]][bit] for yi in y]
            est = _copy.deepcopy(self.estimator); est.fit(X, yb)
            self._estimators.append(est)
        return self

    def predict(self, X):
        n = len(X); n_codes = len(self._estimators)
        codes = []
        for est in self._estimators:
            preds = est.predict(X)
            codes.append([int(p) for p in preds])
        result = []
        for i in range(n):
            pred_code = [codes[b][i] for b in range(n_codes)]
            # Distancia de Hamming al código más cercano
            best_c = self.classes_[0]; best_d = float('inf')
            for k, c in enumerate(self.classes_):
                d = sum(pred_code[b]!=self._code_book[k][b] for b in range(n_codes))
                if d < best_d: best_d = d; best_c = c
            result.append(best_c)
        return result

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)
