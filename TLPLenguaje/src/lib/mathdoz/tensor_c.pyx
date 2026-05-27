# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
tensor_c.pyx — Motor de álgebra lineal compilado con Cython/C.

Operaciones implementadas (todas sin NumPy):
  matmul(a, b)          — multiplicación matricial (M×K) × (K×N) → (M×N)
  transpose(m)          — transposición (M×N) → (N×M)
  add_mat(a, b)         — suma elemento a elemento
  sub_mat(a, b)         — resta elemento a elemento
  scale_mat(a, s)       — escala todos los elementos por escalar s
  dot_vec(a, b)         — producto punto de dos vectores
  zeros(rows, cols)     — matriz de ceros
  ones(rows, cols)      — matriz de unos
  relu_mat(a)           — ReLU elemento a elemento, retorna (out, mask)
  sigmoid_mat(a)        — Sigmoid elemento a elemento
  flatten(a)            — aplana matriz a vector
  reshape(v, rows, cols)— convierte vector en matriz
"""

from libc.math cimport exp
from libc.stdlib cimport malloc, free


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos de forma
# ─────────────────────────────────────────────────────────────────────────────

cdef int _rows(list m):
    return len(m)

cdef int _cols(list m):
    if not m:
        return 0
    if isinstance(m[0], list):
        return len(m[0])
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# zeros / ones
# ─────────────────────────────────────────────────────────────────────────────

def zeros(int rows, int cols):
    return [[0.0] * cols for _ in range(rows)]

def ones(int rows, int cols):
    return [[1.0] * cols for _ in range(rows)]


# ─────────────────────────────────────────────────────────────────────────────
# matmul — (M×K) × (K×N) → (M×N)
# ─────────────────────────────────────────────────────────────────────────────

def matmul(list a, list b):
    cdef int M = len(a)
    cdef int K, N, i, j, k
    cdef double s

    if not a or not b:
        raise ValueError("matmul: matrices vacías")

    b_is_vec = not isinstance(b[0], list)
    if b_is_vec:
        K = len(b)
        N = 1
        if len(a[0]) != K:
            raise ValueError(f"matmul: shapes incompatibles ({M}x{len(a[0])}) x ({K},)")
        result = []
        for i in range(M):
            s = 0.0
            row_a = a[i]
            for k in range(K):
                s += row_a[k] * b[k]
            result.append(s)
        return result

    K = len(b)
    N = len(b[0])
    if len(a[0]) != K:
        raise ValueError(f"matmul: shapes incompatibles ({M}x{len(a[0])}) x ({K}x{N})")

    result = []
    for i in range(M):
        row_a = a[i]
        row_out = []
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += row_a[k] * b[k][j]
            row_out.append(s)
        result.append(row_out)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# transpose — (M×N) → (N×M)
# ─────────────────────────────────────────────────────────────────────────────

def transpose(list m):
    cdef int rows = len(m)
    cdef int cols, i, j

    if not m:
        return []
    if not isinstance(m[0], list):
        return [[v] for v in m]

    cols = len(m[0])
    result = []
    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(m[i][j])
        result.append(row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# add_mat / sub_mat / scale_mat
# ─────────────────────────────────────────────────────────────────────────────

def add_mat(list a, list b):
    cdef int i, j, rows = len(a)
    cdef double va, vb

    if not isinstance(a[0], list):
        return [a[i] + b[i] for i in range(rows)]

    result = []
    for i in range(rows):
        row = []
        ai = a[i]; bi = b[i]
        for j in range(len(ai)):
            row.append(ai[j] + bi[j])
        result.append(row)
    return result


def sub_mat(list a, list b):
    cdef int i, j, rows = len(a)

    if not isinstance(a[0], list):
        return [a[i] - b[i] for i in range(rows)]

    result = []
    for i in range(rows):
        row = []
        ai = a[i]; bi = b[i]
        for j in range(len(ai)):
            row.append(ai[j] - bi[j])
        result.append(row)
    return result


def scale_mat(list a, double s):
    cdef int i, j, rows = len(a)

    if not isinstance(a[0], list):
        return [a[i] * s for i in range(rows)]

    result = []
    for i in range(rows):
        row = []
        ai = a[i]
        for j in range(len(ai)):
            row.append(ai[j] * s)
        result.append(row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# dot_vec — producto punto de dos vectores planos
# ─────────────────────────────────────────────────────────────────────────────

def dot_vec(list a, list b):
    cdef int i, n = len(a)
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# relu_mat — retorna (out, mask)
# ─────────────────────────────────────────────────────────────────────────────

def relu_mat(list a):
    cdef int i, j, rows = len(a)
    cdef double v

    if not isinstance(a[0], list):
        out  = []
        mask = []
        for i in range(rows):
            v = a[i]
            out.append(v if v > 0.0 else 0.0)
            mask.append(1.0 if v > 0.0 else 0.0)
        return out, mask

    out_mat  = []
    mask_mat = []
    for i in range(rows):
        ai = a[i]
        out_row  = []
        mask_row = []
        for j in range(len(ai)):
            v = ai[j]
            out_row.append(v if v > 0.0 else 0.0)
            mask_row.append(1.0 if v > 0.0 else 0.0)
        out_mat.append(out_row)
        mask_mat.append(mask_row)
    return out_mat, mask_mat


# ─────────────────────────────────────────────────────────────────────────────
# sigmoid_mat — σ(x) = 1 / (1 + e^(-x))
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid_mat(list a):
    cdef int i, j, rows = len(a)
    cdef double v

    if not isinstance(a[0], list):
        return [1.0 / (1.0 + exp(-a[i])) for i in range(rows)]

    result = []
    for i in range(rows):
        ai = a[i]
        row = []
        for j in range(len(ai)):
            row.append(1.0 / (1.0 + exp(-ai[j])))
        result.append(row)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# flatten / reshape
# ─────────────────────────────────────────────────────────────────────────────

def flatten(list m):
    if not m:
        return []
    if not isinstance(m[0], list):
        return list(m)
    result = []
    for row in m:
        result.extend(row)
    return result


def reshape(list v, int rows, int cols):
    if len(v) != rows * cols:
        raise ValueError(f"reshape: {len(v)} elementos no caben en ({rows},{cols})")
    result = []
    idx = 0
    for _ in range(rows):
        result.append(v[idx:idx+cols])
        idx += cols
    return result
