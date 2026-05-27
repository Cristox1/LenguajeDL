"""
sklearndoz.metrics — Métricas de evaluación, Python puro.

Cubre: accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
       balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
       confusion_matrix, classification_report,
       mean_squared_error, root_mean_squared_error, mean_absolute_error,
       mean_absolute_percentage_error, median_absolute_error, max_error,
       r2_score, explained_variance_score,
       roc_curve, roc_auc_score, log_loss, hinge_loss, brier_score_loss,
       adjusted_rand_score, silhouette_score, calinski_harabasz_score,
       davies_bouldin_score,
       euclidean_distances, cosine_similarity, pairwise_distances.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import sqrt_doz, abs_doz, log_doz, exp_doz
from errors import TrezRuntimeError


def _check(a, b):
    if len(a) != len(b): raise TrezRuntimeError("metrics: listas de distinto tamaño.")

def _mean(v): return sum(v)/len(v) if v else 0.0


# ── Clasificación ─────────────────────────────────────────────────────────────

def accuracy_score(y_true, y_pred, normalize=True):
    _check(y_true, y_pred)
    n = sum(1 for a,b in zip(y_true,y_pred) if a==b)
    return n/len(y_true) if normalize else n

def _tp_fp_fn_tn(y_true, y_pred, cls):
    tp=fp=fn=tn=0
    for a,b in zip(y_true,y_pred):
        if b==cls and a==cls:  tp+=1
        elif b==cls and a!=cls: fp+=1
        elif b!=cls and a==cls: fn+=1
        else: tn+=1
    return tp,fp,fn,tn

def precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0):
    _check(y_true, y_pred)
    classes = sorted(set(y_true)|set(y_pred))
    if average == 'binary':
        tp,fp,fn,tn = _tp_fp_fn_tn(y_true,y_pred,pos_label)
        return tp/(tp+fp) if tp+fp else zero_division
    scores = []
    weights = [] if average == 'weighted' else None
    for c in classes:
        tp,fp,fn,tn = _tp_fp_fn_tn(y_true,y_pred,c)
        scores.append(tp/(tp+fp) if tp+fp else zero_division)
        if weights is not None: weights.append(sum(1 for v in y_true if v==c))
    if average == 'macro': return _mean(scores)
    if average == 'weighted': total=sum(weights); return sum(s*w for s,w in zip(scores,weights))/total if total else 0.0
    return _mean(scores)

def recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0):
    _check(y_true, y_pred)
    classes = sorted(set(y_true)|set(y_pred))
    if average == 'binary':
        tp,fp,fn,tn = _tp_fp_fn_tn(y_true,y_pred,pos_label)
        return tp/(tp+fn) if tp+fn else zero_division
    scores = []; weights = [] if average == 'weighted' else None
    for c in classes:
        tp,fp,fn,tn = _tp_fp_fn_tn(y_true,y_pred,c)
        scores.append(tp/(tp+fn) if tp+fn else zero_division)
        if weights is not None: weights.append(sum(1 for v in y_true if v==c))
    if average == 'macro': return _mean(scores)
    if average == 'weighted': total=sum(weights); return sum(s*w for s,w in zip(scores,weights))/total if total else 0.0
    return _mean(scores)

def f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0):
    return fbeta_score(y_true, y_pred, beta=1.0, average=average, pos_label=pos_label, zero_division=zero_division)

def fbeta_score(y_true, y_pred, beta=1.0, average='binary', pos_label=1, zero_division=0):
    _check(y_true, y_pred)
    b2 = beta**2
    classes = sorted(set(y_true)|set(y_pred))
    def _fb(cls):
        tp,fp,fn,_ = _tp_fp_fn_tn(y_true,y_pred,cls)
        denom = (1+b2)*tp + b2*fn + fp
        return (1+b2)*tp/denom if denom else zero_division
    if average == 'binary': return _fb(pos_label)
    scores = [_fb(c) for c in classes]
    if average == 'macro': return _mean(scores)
    weights = [sum(1 for v in y_true if v==c) for c in classes]
    total = sum(weights)
    if average == 'weighted': return sum(s*w for s,w in zip(scores,weights))/total if total else 0.0
    return _mean(scores)

def balanced_accuracy_score(y_true, y_pred):
    classes = sorted(set(y_true))
    recalls = []
    for c in classes:
        tp,fp,fn,tn = _tp_fp_fn_tn(y_true,y_pred,c)
        recalls.append(tp/(tp+fn) if tp+fn else 0.0)
    return _mean(recalls)

def cohen_kappa_score(y_true, y_pred):
    _check(y_true,y_pred)
    n = len(y_true)
    classes = sorted(set(y_true)|set(y_pred))
    K = len(classes); idx={c:i for i,c in enumerate(classes)}
    M = [[0]*K for _ in range(K)]
    for a,b in zip(y_true,y_pred): M[idx[a]][idx[b]] += 1
    p_obs = sum(M[i][i] for i in range(K))/n
    row_sums = [sum(M[i]) for i in range(K)]
    col_sums = [sum(M[r][j] for r in range(K)) for j in range(K)]
    p_exp = sum(row_sums[i]*col_sums[i] for i in range(K))/(n*n)
    return (p_obs-p_exp)/(1-p_exp) if 1-p_exp else 0.0

def matthews_corrcoef(y_true, y_pred):
    tp,fp,fn,tn = _tp_fp_fn_tn(y_true,y_pred,1)
    denom = sqrt_doz((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return (tp*tn-fp*fn)/denom if denom else 0.0

def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    _check(y_true,y_pred)
    classes = labels if labels else sorted(set(y_true)|set(y_pred))
    K = len(classes); idx={c:i for i,c in enumerate(classes)}
    M = [[0]*K for _ in range(K)]
    for a,b in zip(y_true,y_pred):
        if a in idx and b in idx: M[idx[a]][idx[b]] += 1
    if normalize == 'true':
        M = [[M[i][j]/(sum(M[i]) or 1) for j in range(K)] for i in range(K)]
    elif normalize == 'pred':
        col_s = [sum(M[r][j] for r in range(K)) for j in range(K)]
        M = [[M[i][j]/(col_s[j] or 1) for j in range(K)] for i in range(K)]
    elif normalize == 'all':
        t = sum(M[i][j] for i in range(K) for j in range(K)) or 1
        M = [[M[i][j]/t for j in range(K)] for i in range(K)]
    return M

def classification_report(y_true, y_pred, labels=None, output_dict=False):
    classes = labels if labels else sorted(set(y_true)|set(y_pred))
    lines = []; report = {}
    header = f"{'':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines.append(header)
    for c in classes:
        p = precision_score(y_true,y_pred,average='binary',pos_label=c)
        r = recall_score(y_true,y_pred,average='binary',pos_label=c)
        f = f1_score(y_true,y_pred,average='binary',pos_label=c)
        s = sum(1 for v in y_true if v==c)
        lines.append(f"{str(c):>10} {p:>10.2f} {r:>10.2f} {f:>10.2f} {s:>10}")
        report[c] = {'precision':p,'recall':r,'f1-score':f,'support':s}
    acc = accuracy_score(y_true,y_pred)
    lines.append(f"\n{'accuracy':>10} {acc:>31.2f} {len(y_true):>10}")
    report['accuracy'] = acc
    if output_dict: return report
    return '\n'.join(lines)

def hamming_loss(y_true, y_pred):
    return sum(1 for a,b in zip(y_true,y_pred) if a!=b)/len(y_true)

def zero_one_loss(y_true, y_pred, normalize=True):
    n = sum(1 for a,b in zip(y_true,y_pred) if a!=b)
    return n/len(y_true) if normalize else n

def log_loss(y_true, y_pred_proba, eps=1e-15):
    total = 0.0
    for yt, yp in zip(y_true, y_pred_proba):
        p = max(eps, min(1-eps, yp if isinstance(yp,(int,float)) else yp[int(yt)]))
        total += -log_doz(p)
    return total/len(y_true)

def hinge_loss(y_true, pred_decision, labels=None):
    return _mean([max(0.0, 1-yt*pd) for yt,pd in zip(y_true, pred_decision)])

def brier_score_loss(y_true, y_prob):
    return _mean([(yt-yp)**2 for yt,yp in zip(y_true,y_prob)])


# ── Curvas ROC / PR ───────────────────────────────────────────────────────────

def roc_curve(y_true, y_score, pos_label=1):
    pairs = sorted(zip(y_score, y_true), reverse=True)
    P = sum(1 for v in y_true if v==pos_label)
    N = len(y_true)-P
    tp=fp=0; tprs=[0.0]; fprs=[0.0]; thresholds=[]
    for score, label in pairs:
        if label==pos_label: tp+=1
        else: fp+=1
        tprs.append(tp/P if P else 0.0)
        fprs.append(fp/N if N else 0.0)
        thresholds.append(score)
    return fprs, tprs, thresholds

def roc_auc_score(y_true, y_score, pos_label=1):
    fprs, tprs, _ = roc_curve(y_true, y_score, pos_label)
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i]-fprs[i-1])*(tprs[i]+tprs[i-1])/2
    return auc

def precision_recall_curve(y_true, y_score, pos_label=1):
    pairs = sorted(zip(y_score, y_true), reverse=True)
    P = sum(1 for v in y_true if v==pos_label)
    tp=0; fp=0; precs=[]; recs=[]
    for score, label in pairs:
        if label==pos_label: tp+=1
        else: fp+=1
        precs.append(tp/(tp+fp) if tp+fp else 0.0)
        recs.append(tp/P if P else 0.0)
    precs.append(1.0); recs.append(0.0)
    return precs, recs

def average_precision_score(y_true, y_score, pos_label=1):
    precs, recs = precision_recall_curve(y_true, y_score, pos_label)
    ap = 0.0
    for i in range(1, len(recs)):
        ap += (recs[i-1]-recs[i])*precs[i]
    return ap


# ── Regresión ─────────────────────────────────────────────────────────────────

def mean_squared_error(y_true, y_pred, squared=True):
    _check(y_true,y_pred)
    mse = _mean([(a-b)**2 for a,b in zip(y_true,y_pred)])
    return mse if squared else sqrt_doz(mse)

def root_mean_squared_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def mean_absolute_error(y_true, y_pred):
    _check(y_true,y_pred)
    return _mean([abs_doz(a-b) for a,b in zip(y_true,y_pred)])

def mean_absolute_percentage_error(y_true, y_pred):
    _check(y_true,y_pred)
    return _mean([abs_doz((a-b)/(a if a else 1e-10)) for a,b in zip(y_true,y_pred)])

def median_absolute_error(y_true, y_pred):
    _check(y_true,y_pred)
    errs = sorted(abs_doz(a-b) for a,b in zip(y_true,y_pred))
    n = len(errs)
    return (errs[n//2-1]+errs[n//2])/2 if n%2==0 else errs[n//2]

def max_error(y_true, y_pred):
    return max(abs_doz(a-b) for a,b in zip(y_true,y_pred))

def r2_score(y_true, y_pred):
    _check(y_true,y_pred)
    m = _mean(y_true)
    ss_res = sum((a-b)**2 for a,b in zip(y_true,y_pred))
    ss_tot = sum((a-m)**2 for a in y_true)
    return 1-ss_res/ss_tot if ss_tot else 0.0

def explained_variance_score(y_true, y_pred):
    _check(y_true,y_pred)
    resid = [a-b for a,b in zip(y_true,y_pred)]
    var_resid = _mean([(r-_mean(resid))**2 for r in resid])
    var_true = _mean([(a-_mean(y_true))**2 for a in y_true])
    return 1-var_resid/var_true if var_true else 0.0

def mean_pinball_loss(y_true, y_pred, alpha=0.5):
    def _pinball(yt, yp):
        r = yt-yp
        return alpha*r if r>=0 else (alpha-1)*r
    return _mean([_pinball(a,b) for a,b in zip(y_true,y_pred)])


# ── Distancias ────────────────────────────────────────────────────────────────

def euclidean_distances(X, Y=None, squared=False):
    if Y is None: Y = X
    def d2(a,b): return sum((ai-bi)**2 for ai,bi in zip(a,b))
    M = [[d2(a,b) for b in Y] for a in X]
    if not squared: M = [[sqrt_doz(v) for v in row] for row in M]
    return M

def manhattan_distances(X, Y=None):
    if Y is None: Y = X
    return [[sum(abs_doz(ai-bi) for ai,bi in zip(a,b)) for b in Y] for a in X]

def cosine_similarity(X, Y=None):
    if Y is None: Y = X
    def _cos(a, b):
        dot = sum(ai*bi for ai,bi in zip(a,b))
        na = sqrt_doz(sum(ai**2 for ai in a)) or 1e-10
        nb = sqrt_doz(sum(bi**2 for bi in b)) or 1e-10
        return dot/(na*nb)
    return [[_cos(a,b) for b in Y] for a in X]

def cosine_distances(X, Y=None):
    sim = cosine_similarity(X, Y)
    return [[1-v for v in row] for row in sim]

def pairwise_distances(X, Y=None, metric='euclidean'):
    if metric == 'euclidean': return euclidean_distances(X, Y)
    if metric == 'manhattan': return manhattan_distances(X, Y)
    if metric == 'cosine':    return cosine_distances(X, Y)
    raise TrezRuntimeError(f"pairwise_distances: métrica '{metric}' no soportada.")


# ── Clustering ────────────────────────────────────────────────────────────────

def adjusted_rand_score(labels_true, labels_pred):
    n = len(labels_true)
    classes = set(labels_true); clusters = set(labels_pred)
    # Tabla de contingencia
    cont = {}
    for a,b in zip(labels_true,labels_pred):
        cont[(a,b)] = cont.get((a,b),0)+1
    def _c2(x): return x*(x-1)//2
    sum_comb = sum(_c2(v) for v in cont.values())
    row_sums = {}
    for (a,_),v in cont.items(): row_sums[a]=row_sums.get(a,0)+v
    col_sums = {}
    for (_,b),v in cont.items(): col_sums[b]=col_sums.get(b,0)+v
    sum_rows = sum(_c2(v) for v in row_sums.values())
    sum_cols = sum(_c2(v) for v in col_sums.values())
    total_pairs = _c2(n)
    expected = sum_rows*sum_cols/total_pairs if total_pairs else 0
    max_index = (sum_rows+sum_cols)/2
    denom = max_index-expected
    return (sum_comb-expected)/denom if denom else 1.0

def silhouette_score(X, labels, metric='euclidean'):
    if metric == 'euclidean':
        def dist(a,b): return sqrt_doz(sum((ai-bi)**2 for ai,bi in zip(a,b)))
    elif metric == 'manhattan':
        def dist(a,b): return sum(abs_doz(ai-bi) for ai,bi in zip(a,b))
    else: raise TrezRuntimeError(f"silhouette_score: métrica '{metric}' no soportada.")
    n = len(X); unique = list(set(labels))
    if len(unique) < 2: return 0.0
    scores = []
    for i in range(n):
        same = [j for j in range(n) if j!=i and labels[j]==labels[i]]
        a = sum(dist(X[i],X[j]) for j in same)/len(same) if same else 0.0
        b_vals = []
        for c in unique:
            if c == labels[i]: continue
            other = [j for j in range(n) if labels[j]==c]
            if other: b_vals.append(sum(dist(X[i],X[j]) for j in other)/len(other))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a,b)
        scores.append((b-a)/denom if denom else 0.0)
    return _mean(scores)

def calinski_harabasz_score(X, labels):
    n = len(X); d = len(X[0])
    unique = list(set(labels)); K = len(unique)
    if K < 2: return 0.0
    def _mean_vec(pts):
        return [_mean([p[j] for p in pts]) for j in range(d)]
    overall = _mean_vec(X)
    clusters = {c: [X[i] for i in range(n) if labels[i]==c] for c in unique}
    centroids = {c: _mean_vec(pts) for c,pts in clusters.items()}
    bss = sum(len(clusters[c])*sum((centroids[c][j]-overall[j])**2 for j in range(d)) for c in unique)
    wss = sum(sum((X[i][j]-centroids[labels[i]][j])**2 for j in range(d)) for i in range(n))
    return (bss/(K-1))/(wss/(n-K)) if wss and n>K else 0.0

def davies_bouldin_score(X, labels):
    n = len(X); d = len(X[0])
    unique = list(set(labels)); K = len(unique)
    if K < 2: return 0.0
    def _mean_vec(pts): return [_mean([p[j] for p in pts]) for j in range(d)]
    def _edist(a,b): return sqrt_doz(sum((ai-bi)**2 for ai,bi in zip(a,b)))
    clusters = {c:[X[i] for i in range(n) if labels[i]==c] for c in unique}
    centroids = {c:_mean_vec(pts) for c,pts in clusters.items()}
    s = {c:_mean([_edist(p,centroids[c]) for p in clusters[c]]) for c in unique}
    db = 0.0
    for i, ci in enumerate(unique):
        best = max((s[ci]+s[cj])/_edist(centroids[ci],centroids[cj])
                   for j,cj in enumerate(unique) if i!=j and _edist(centroids[ci],centroids[cj])>0)
        db += best
    return db/K

def homogeneity_score(labels_true, labels_pred):
    return _v_measure(labels_true, labels_pred)[0]

def completeness_score(labels_true, labels_pred):
    return _v_measure(labels_true, labels_pred)[1]

def v_measure_score(labels_true, labels_pred):
    h, c = _v_measure(labels_true, labels_pred)
    return 2*h*c/(h+c) if h+c else 0.0

def _v_measure(labels_true, labels_pred):
    def _entropy(labels):
        n = len(labels); counts = {}
        for v in labels: counts[v]=counts.get(v,0)+1
        return -sum((c/n)*log_doz(c/n+1e-10) for c in counts.values())
    def _conditional_entropy(lt, lp):
        n = len(lt); joint = {}
        for a,b in zip(lt,lp): joint[(a,b)]=joint.get((a,b),0)+1
        pred_counts = {}
        for (_,b),v in joint.items(): pred_counts[b]=pred_counts.get(b,0)+v
        return -sum((v/n)*log_doz((v/n)/(pred_counts[b]/n)+1e-10) for (a,b),v in joint.items())
    ht = _entropy(labels_true); hp = _entropy(labels_pred)
    h_tk = _conditional_entropy(labels_true, labels_pred)
    h_kt = _conditional_entropy(labels_pred, labels_true)
    homogeneity = 1-h_kt/hp if hp else 1.0
    completeness = 1-h_tk/ht if ht else 1.0
    return homogeneity, completeness

def mutual_info_score(labels_true, labels_pred):
    n = len(labels_true)
    joint = {}
    for a,b in zip(labels_true,labels_pred): joint[(a,b)]=joint.get((a,b),0)+1
    row_s = {}; col_s = {}
    for (a,b),v in joint.items():
        row_s[a]=row_s.get(a,0)+v; col_s[b]=col_s.get(b,0)+v
    return sum((v/n)*log_doz((v*n)/(row_s[a]*col_s[b]+1e-10)+1e-10) for (a,b),v in joint.items())

def normalized_mutual_info_score(labels_true, labels_pred):
    mi = mutual_info_score(labels_true, labels_pred)
    from lib.mathdoz.core_mathdoz import sqrt_doz as _sq
    def _ent(l):
        n=len(l); c={}
        for v in l: c[v]=c.get(v,0)+1
        return -sum((x/n)*log_doz(x/n+1e-10) for x in c.values())
    denom = _sq(_ent(labels_true)*_ent(labels_pred))
    return mi/denom if denom else 0.0
