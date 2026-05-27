import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from lib.mathdoz.core_mathdoz import exp_doz, log_doz, sqrt_doz
from errors import TrezRuntimeError

def mse(y_true, y_pred):
    """Calcula el Error Cuadrático Medio entre dos tensores o listas de números."""
    if isinstance(y_true, list) and isinstance(y_pred, list):
        if len(y_true) != len(y_pred):
            raise TrezRuntimeError("Loss mse() mismatch: Las dimensiones de y_true e y_pred no coinciden.")
        return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    elif isinstance(y_true, (int, float)) and isinstance(y_pred, (int, float)):
        return (y_true - y_pred) ** 2
    else:
        raise TrezRuntimeError("Tipos no soportados para mse(). Se requieren números o listas planas.")

def mse_grad(y_true, y_pred):
    """Calcula el gradiente (derivada) del MSE con respecto a y_pred."""
    if isinstance(y_true, list) and isinstance(y_pred, list):
        if len(y_true) != len(y_pred):
            raise TrezRuntimeError("Loss mse_grad() mismatch: Las dimensiones de y_true e y_pred no coinciden.")
        n = len(y_true)
        # La derivada del MSE es 2 * (y_pred - y_true) / N
        return [2 * (p - t) / n for t, p in zip(y_true, y_pred)]
    elif isinstance(y_true, (int, float)) and isinstance(y_pred, (int, float)):
        return 2 * (y_pred - y_true)
    else:
        raise TrezRuntimeError("Tipos no soportados para mse_grad().")


def cross_entropy(logits, targets):
    """
    Entropía cruzada con softmax numericamente estable incorporado.
    logits:  lista de listas (batch x n_clases) — valores crudos de la red
    targets: lista de índices enteros (clase correcta por muestra)
    Retorna el loss promedio como escalar.
    """
    if not isinstance(logits, list) or not logits:
        raise TrezRuntimeError("cross_entropy: logits debe ser lista de listas no vacía.")
    if not isinstance(targets, list) or len(targets) != len(logits):
        raise TrezRuntimeError("cross_entropy: targets debe tener el mismo largo que logits.")

    batch_size = len(logits)
    loss_sum = 0.0
    for i, (row, t) in enumerate(zip(logits, targets)):
        if not isinstance(row, list):
            raise TrezRuntimeError(f"cross_entropy: logits[{i}] debe ser lista de números.")
        if not isinstance(t, int) or t < 0 or t >= len(row):
            raise TrezRuntimeError(f"cross_entropy: target {t} fuera de rango para {len(row)} clases.")
        max_val = max(row)
        shifted = [x - max_val for x in row]
        exp_vals = [exp_doz(x) for x in shifted]
        sum_exp = sum(exp_vals)
        log_softmax_t = shifted[t] - log_doz(sum_exp)
        loss_sum -= log_softmax_t

    return loss_sum / batch_size


def cross_entropy_grad(logits, targets):
    """
    Gradiente de cross_entropy respecto a logits.
    Retorna lista de listas con misma forma que logits.
    grad[i][j] = softmax(logits[i])[j] - 1{j == targets[i]}
    """
    if not isinstance(logits, list) or not logits:
        raise TrezRuntimeError("cross_entropy_grad: logits debe ser lista de listas no vacía.")
    if not isinstance(targets, list) or len(targets) != len(logits):
        raise TrezRuntimeError("cross_entropy_grad: targets debe tener el mismo largo que logits.")

    batch_size = len(logits)
    grads = []
    for i, (row, t) in enumerate(zip(logits, targets)):
        max_val = max(row)
        shifted = [x - max_val for x in row]
        exp_vals = [exp_doz(x) for x in shifted]
        sum_exp = sum(exp_vals)
        softmax = [e / sum_exp for e in exp_vals]
        softmax[t] -= 1.0
        grads.append([g / batch_size for g in softmax])
    return grads


# ── Binary Cross-Entropy ──────────────────────────────────────────────────────

def bce_loss(y_pred, y_true):
    """BCE para salidas sigmoid. y_pred: lista de escalares en (0,1). y_true: {0,1}."""
    if len(y_pred) != len(y_true):
        raise TrezRuntimeError("bce_loss: y_pred e y_true deben tener el mismo largo.")
    eps = 1e-12
    n = len(y_pred)
    return -sum(y_true[i]*log_doz(y_pred[i]+eps) + (1-y_true[i])*log_doz(1-y_pred[i]+eps)
                for i in range(n)) / n

def bce_grad(y_pred, y_true):
    """Gradiente de BCE respecto a y_pred (antes de sigmoid)."""
    eps = 1e-12; n = len(y_pred)
    return [-(y_true[i]/(y_pred[i]+eps) - (1-y_true[i])/(1-y_pred[i]+eps))/n
            for i in range(n)]


# ── Contrastive Loss (redes siamesas) ─────────────────────────────────────────

def contrastive_loss(dist, label, margin=1.0):
    """
    Contrastive loss de Hadsell et al.
    dist:   distancia euclidiana entre los dos embeddings (escalar o lista).
    label:  0 = par similar, 1 = par disímil (escalar o lista).
    margin: margen mínimo para pares disímiles.
    Retorna loss promedio.
    """
    if isinstance(dist, (int, float)):
        dist = [dist]; label = [label]
    n = len(dist)
    loss = 0.0
    for d, l in zip(dist, label):
        sim_term  = (1 - l) * d * d
        diff_term = l * max(0.0, margin - d) ** 2
        loss += 0.5 * (sim_term + diff_term)
    return loss / n

def contrastive_grad(emb_a, emb_b, label, margin=1.0):
    """
    Gradientes de contrastive_loss respecto a emb_a y emb_b.
    emb_a, emb_b: vectores de embedding (listas de listas: batch × dim).
    label: lista de {0,1} por par.
    Retorna (grad_a, grad_b).
    """
    if isinstance(emb_a[0], (int, float)):
        emb_a = [emb_a]; emb_b = [emb_b]; label = [label]
    n = len(emb_a); d = len(emb_a[0])
    grad_a = [[0.0]*d for _ in range(n)]
    grad_b = [[0.0]*d for _ in range(n)]
    for i in range(n):
        diff = [emb_a[i][j] - emb_b[i][j] for j in range(d)]
        dist = sqrt_doz(sum(x*x for x in diff)) + 1e-12
        l = label[i]
        if l == 0:
            # Par similar: empuja embeddings juntos
            scale = (1.0 - 0.0) / n  # d(d²)/d(emb_a) = 2*(emb_a-emb_b)
            for j in range(d):
                grad_a[i][j] =  diff[j] / n
                grad_b[i][j] = -diff[j] / n
        else:
            # Par disímil: empuja si dist < margin
            push = max(0.0, margin - dist)
            scale = -push / (dist * n)
            for j in range(d):
                grad_a[i][j] =  scale * diff[j]
                grad_b[i][j] = -scale * diff[j]
    return grad_a, grad_b


# ── Triplet Loss ──────────────────────────────────────────────────────────────

def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Triplet loss: max(0, d(a,p) - d(a,n) + margin).
    anchor, positive, negative: listas de embeddings (batch × dim) o vectores 1D.
    Retorna loss promedio.
    """
    if isinstance(anchor[0], (int, float)):
        anchor = [anchor]; positive = [positive]; negative = [negative]
    n = len(anchor); loss = 0.0
    for i in range(n):
        d_pos = sqrt_doz(sum((anchor[i][j]-positive[i][j])**2 for j in range(len(anchor[i]))))
        d_neg = sqrt_doz(sum((anchor[i][j]-negative[i][j])**2 for j in range(len(anchor[i]))))
        loss += max(0.0, d_pos - d_neg + margin)
    return loss / n

def triplet_grad(anchor, positive, negative, margin=0.2):
    """
    Gradientes de triplet_loss respecto a anchor, positive, negative.
    Retorna (grad_anchor, grad_positive, grad_negative).
    """
    if isinstance(anchor[0], (int, float)):
        anchor = [anchor]; positive = [positive]; negative = [negative]
    n = len(anchor); d = len(anchor[0])
    ga = [[0.0]*d for _ in range(n)]
    gp = [[0.0]*d for _ in range(n)]
    gn = [[0.0]*d for _ in range(n)]
    for i in range(n):
        diff_ap = [anchor[i][j]-positive[i][j] for j in range(d)]
        diff_an = [anchor[i][j]-negative[i][j] for j in range(d)]
        d_pos = sqrt_doz(sum(x*x for x in diff_ap)) + 1e-12
        d_neg = sqrt_doz(sum(x*x for x in diff_an)) + 1e-12
        if d_pos - d_neg + margin > 0:
            for j in range(d):
                ga[i][j] = (diff_ap[j]/d_pos - diff_an[j]/d_neg) / n
                gp[i][j] = -diff_ap[j] / (d_pos * n)
                gn[i][j] =  diff_an[j] / (d_neg * n)
    return ga, gp, gn


# ── KL Divergence (VAE) ───────────────────────────────────────────────────────

def kl_divergence(mu, log_var):
    """
    KL(q || N(0,1)) para VAE: -0.5 * sum(1 + log_var - mu² - exp(log_var)).
    mu, log_var: listas de escalares (latent dim) o listas de listas (batch × dim).
    Retorna loss promedio por muestra.
    """
    if isinstance(mu[0], (int, float)):
        mu = [mu]; log_var = [log_var]
    n = len(mu); total = 0.0
    for i in range(n):
        for j in range(len(mu[i])):
            total += -0.5 * (1 + log_var[i][j] - mu[i][j]**2 - exp_doz(log_var[i][j]))
    return total / n

def kl_grad(mu, log_var):
    """Gradientes de KL respecto a mu y log_var."""
    if isinstance(mu[0], (int, float)):
        mu = [mu]; log_var = [log_var]
    n = len(mu); d = len(mu[0])
    g_mu  = [[mu[i][j] / n for j in range(d)] for i in range(n)]
    g_lv  = [[0.5 * (exp_doz(log_var[i][j]) - 1) / n for j in range(d)] for i in range(n)]
    return g_mu, g_lv


# ── Huber Loss ────────────────────────────────────────────────────────────────

def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss — robusto a outliers."""
    n = len(y_pred)
    total = 0.0
    for p, t in zip(y_pred, y_true):
        e = abs(p - t)
        total += 0.5*e*e if e <= delta else delta*(e - 0.5*delta)
    return total / n

def huber_grad(y_pred, y_true, delta=1.0):
    n = len(y_pred)
    return [(p-t)/n if abs(p-t) <= delta else delta*(1 if p>t else -1)/n
            for p, t in zip(y_pred, y_true)]
