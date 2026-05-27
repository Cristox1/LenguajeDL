"""
sklearndoz.calibration — Calibración de probabilidades, Python puro.

Cubre: CalibratedClassifierCV (isotónica y sigmoid/Platt).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import exp_doz, log_doz
from errors import TrezRuntimeError
import copy as _copy


def _mean(v): return sum(v)/len(v) if v else 0.0
def _logistic(z):
    if z >= 0: return 1.0/(1.0+exp_doz(-z))
    ez = exp_doz(z); return ez/(1.0+ez)


class _PlattScaler:
    """Escala de Platt: calibración sigmoid sobre scores."""
    def __init__(self): self.a = 0.0; self.b = 0.0

    def fit(self, scores, y, max_iter=100, lr=0.01):
        a, b = 0.0, 0.0
        for _ in range(max_iter):
            da = db = 0.0
            for s, yi in zip(scores, y):
                p = _logistic(a*s+b); err = p-yi
                da += err*s/len(scores); db += err/len(scores)
            a -= lr*da; b -= lr*db
        self.a = a; self.b = b

    def predict_proba(self, scores):
        return [[1-_logistic(self.a*s+self.b), _logistic(self.a*s+self.b)] for s in scores]


class _IsotonicScaler:
    """Calibración isotónica (PAVA)."""
    def fit(self, scores, y):
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        xs = [scores[i] for i in order]; ys = [float(y[i]) for i in order]
        # PAVA
        val_blocks = [[ys[0]]]; x_blocks = [[xs[0]]]
        for i in range(1, len(ys)):
            val_blocks.append([ys[i]]); x_blocks.append([xs[i]])
            while len(val_blocks) > 1:
                m1 = _mean(val_blocks[-2]); m2 = _mean(val_blocks[-1])
                if m1 <= m2: break
                val_blocks[-2] = val_blocks[-2]+val_blocks[-1]; x_blocks[-2] = x_blocks[-2]+x_blocks[-1]
                val_blocks.pop(); x_blocks.pop()
        self._xs = [x for xb in x_blocks for x in xb]
        self._ys = [_mean(vb) for vb,xb in zip(val_blocks,x_blocks) for _ in xb]

    def predict_proba(self, scores):
        result = []
        for s in scores:
            if s <= self._xs[0]: p = self._ys[0]
            elif s >= self._xs[-1]: p = self._ys[-1]
            else:
                p = self._ys[-1]
                for i in range(len(self._xs)-1):
                    if self._xs[i] <= s <= self._xs[i+1]:
                        dx = self._xs[i+1]-self._xs[i]
                        if dx < 1e-10: p = self._ys[i]; break
                        t = (s-self._xs[i])/dx
                        p = self._ys[i]*(1-t)+self._ys[i+1]*t; break
            p = min(max(p, 0.0), 1.0)
            result.append([1-p, p])
        return result


class CalibratedClassifierCV:
    """Calibración de probabilidades con CV o prefit."""
    def __init__(self, estimator, method='sigmoid', cv=5):
        self.estimator=estimator; self.method=method; self.cv=int(cv)
        self.calibrated_classifiers_=[]

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y))
        # Entrenar el estimador base
        base = _copy.deepcopy(self.estimator); base.fit(X, y)
        # Obtener scores para calibración (cross-val simplificada: usar todo)
        if hasattr(base, 'decision_function'):
            raw = base.decision_function(X)
            scores = raw if isinstance(raw[0], (int,float)) else [r[0] for r in raw]
        elif hasattr(base, 'predict_proba'):
            probs = base.predict_proba(X)
            K = len(self.classes_)
            scores = [probs[i][1] if K==2 else max(probs[i]) for i in range(len(X))]
        else:
            preds = base.predict(X)
            cls_idx = {c:i for i,c in enumerate(self.classes_)}
            scores = [float(cls_idx.get(p,0)) for p in preds]
        yb = [1 if yi==self.classes_[-1] else 0 for yi in y]
        if self.method == 'sigmoid':
            cal = _PlattScaler(); cal.fit(scores, yb)
        else:
            cal = _IsotonicScaler(); cal.fit(scores, yb)
        self._base = base; self._cal = cal
        return self

    def predict_proba(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        if hasattr(self._base, 'decision_function'):
            raw = self._base.decision_function(X)
            scores = raw if isinstance(raw[0], (int,float)) else [r[0] for r in raw]
        elif hasattr(self._base, 'predict_proba'):
            probs = self._base.predict_proba(X)
            K = len(self.classes_)
            scores = [probs[i][1] if K==2 else max(probs[i]) for i in range(len(X))]
        else:
            preds = self._base.predict(X)
            cls_idx = {c:i for i,c in enumerate(self.classes_)}
            scores = [float(cls_idx.get(p,0)) for p in preds]
        return self._cal.predict_proba(scores)

    def predict(self, X):
        probs = self.predict_proba(X)
        return [self.classes_[p.index(max(p))] for p in probs]

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)
