"""
sklearndoz.tree — Árboles de decisión con interfaz sklearn, Python puro.

Cubre: DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor.
Envuelve mldoz.tree con interfaz sklearn-compatible.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mldoz.tree import tree_clf_fit, tree_reg_fit, tree_predict
from errors import TrezRuntimeError


def _mean(v): return sum(v)/len(v) if v else 0.0
def _col(X, j): return [row[j] for row in X]


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', max_features=None):
        self.max_depth=max_depth; self.min_samples_split=int(min_samples_split)
        self.min_samples_leaf=int(min_samples_leaf); self.criterion=criterion
        self.max_features=max_features
        self.tree_=None; self.classes_=None; self.feature_importances_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        self.classes_ = sorted(set(y))
        depth = self.max_depth if self.max_depth is not None else 20
        self.tree_ = tree_clf_fit(X, list(y), depth)
        self.feature_importances_ = [1.0/len(X[0])]*len(X[0])
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [tree_predict(self.tree_, x) for x in X]

    def predict_proba(self, X):
        preds = self.predict(X); K = len(self.classes_)
        cls_idx = {c:i for i,c in enumerate(self.classes_)}
        result = []
        for p in preds:
            row = [0.0]*K; row[cls_idx.get(p, 0)] = 1.0; result.append(row)
        return result

    def score(self, X, y):
        return sum(1 for a,b in zip(y, self.predict(X)) if a==b)/len(y)

    def fit_predict(self, X, y): return self.fit(X, y).predict(X)


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion='mse', max_features=None):
        self.max_depth=max_depth; self.min_samples_split=int(min_samples_split)
        self.min_samples_leaf=int(min_samples_leaf); self.criterion=criterion
        self.max_features=max_features
        self.tree_=None; self.feature_importances_=None

    def fit(self, X, y):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        depth = self.max_depth if self.max_depth is not None else 20
        self.tree_ = tree_reg_fit(X, list(y), depth)
        self.feature_importances_ = [1.0/len(X[0])]*len(X[0])
        return self

    def predict(self, X):
        X = X if isinstance(X[0], list) else [[v] for v in X]
        return [tree_predict(self.tree_, x) for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        ss_res = sum((yi-pi)**2 for yi,pi in zip(y,preds))
        ss_tot = sum((yi-_mean(list(y)))**2 for yi in y) or 1e-10
        return 1-ss_res/ss_tot


# ExtraTree = árbol con threshold aleatorio (envuelve el mismo código)
class ExtraTreeClassifier(DecisionTreeClassifier):
    pass

class ExtraTreeRegressor(DecisionTreeRegressor):
    pass
