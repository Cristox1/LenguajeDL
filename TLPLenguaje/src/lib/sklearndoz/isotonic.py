"""
sklearndoz.isotonic — Regresión isotónica, Python puro.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0


class IsotonicRegression:
    """Regresión isotónica via Pool Adjacent Violators (PAVA)."""
    def __init__(self, increasing=True, out_of_bounds='nan'):
        self.increasing=increasing; self.out_of_bounds=out_of_bounds
        self.X_thresholds_=None; self.y_thresholds_=None

    def fit(self, X, y, sample_weight=None):
        X_list = list(X); y_list = list(y)
        # Ordenar por X
        order = sorted(range(len(X_list)), key=lambda i: X_list[i])
        Xs = [X_list[i] for i in order]
        ys = [y_list[i] for i in order]
        ws = [1.0]*len(ys) if sample_weight is None else [sample_weight[i] for i in order]
        if not self.increasing:
            ys = list(reversed(ys)); ws = list(reversed(ws)); Xs = list(reversed(Xs))
        # PAVA
        blocks = [[ys[0]], [ws[0]], [Xs[0]]]  # values, weights, x_vals stored as lists
        val_blocks = [[ys[0]]]; w_blocks = [[ws[0]]]; x_blocks = [[Xs[0]]]
        for i in range(1, len(ys)):
            val_blocks.append([ys[i]]); w_blocks.append([ws[i]]); x_blocks.append([Xs[i]])
            while len(val_blocks) > 1:
                wm1 = sum(w_blocks[-2]); wm2 = sum(w_blocks[-1])
                m1 = sum(v*w for v,w in zip(val_blocks[-2],w_blocks[-2]))/wm1
                m2 = sum(v*w for v,w in zip(val_blocks[-1],w_blocks[-1]))/wm2
                if m1 <= m2: break
                val_blocks[-2] = val_blocks[-2]+val_blocks[-1]
                w_blocks[-2] = w_blocks[-2]+w_blocks[-1]
                x_blocks[-2] = x_blocks[-2]+x_blocks[-1]
                val_blocks.pop(); w_blocks.pop(); x_blocks.pop()
        # Build thresholds
        xs_out = []; ys_out = []
        for vb, wb, xb in zip(val_blocks, w_blocks, x_blocks):
            wt = sum(wb)
            val = sum(v*w for v,w in zip(vb,wb))/wt
            for x in xb: xs_out.append(x); ys_out.append(val)
        if not self.increasing:
            xs_out = list(reversed(xs_out)); ys_out = list(reversed(ys_out))
        self.X_thresholds_ = xs_out; self.y_thresholds_ = ys_out
        return self

    def predict(self, X):
        result = []
        for x in X:
            if x <= self.X_thresholds_[0]: result.append(self.y_thresholds_[0]); continue
            if x >= self.X_thresholds_[-1]: result.append(self.y_thresholds_[-1]); continue
            # Interpolación lineal
            for i in range(len(self.X_thresholds_)-1):
                if self.X_thresholds_[i] <= x <= self.X_thresholds_[i+1]:
                    dx = self.X_thresholds_[i+1]-self.X_thresholds_[i]
                    if dx < 1e-10:
                        result.append(self.y_thresholds_[i]); break
                    t = (x-self.X_thresholds_[i])/dx
                    result.append(self.y_thresholds_[i]*(1-t)+self.y_thresholds_[i+1]*t); break
            else: result.append(self.y_thresholds_[-1])
        return result

    def fit_transform(self, X, y, sample_weight=None):
        return self.fit(X, y, sample_weight).predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        m = _mean(list(y))
        ss_res = sum((yi-pi)**2 for yi,pi in zip(y,preds))
        ss_tot = sum((yi-m)**2 for yi in y) or 1e-10
        return 1-ss_res/ss_tot
