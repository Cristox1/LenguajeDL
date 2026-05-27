import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError, ShapeMismatchError, TypeMismatchError

# Intentar módulo C compilado con Cython; si no está disponible usar Python puro.
try:
    sys.path.insert(0, os.path.dirname(__file__))
    import tensor_c as _C
    _USE_C = True
except ImportError:
    _USE_C = False


def _shape(t):
    if isinstance(t, list) and t and isinstance(t[0], list):
        return f"[{len(t)}x{len(t[0])}]"
    if isinstance(t, list):
        return f"[{len(t)}]"
    return "escalar"


# ── dot ──────────────────────────────────────────────────────────────────────

def dot(a, b):
    """Producto punto / multiplicación matricial. Usa backend C si disponible."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a * b

    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeMismatchError("dot", "lista o número", type(a).__name__)
    if not a or not b:
        raise TrezRuntimeError("dot() no acepta tensores vacíos.")

    a_is_vec = isinstance(a[0], (int, float))
    b_is_vec = isinstance(b[0], (int, float))
    a_is_mat = isinstance(a[0], list)

    if a_is_vec and b_is_vec:
        if len(a) != len(b):
            raise ShapeMismatchError("dot", _shape(a), _shape(b))
        return _C.dot_vec(a, b) if _USE_C else sum(x * y for x, y in zip(a, b))

    if a_is_mat:
        cols_a = len(a[0])
        rows_b = len(b)
        expected_k = rows_b
        if cols_a != expected_k:
            raise ShapeMismatchError("dot", _shape(a), _shape(b))
        return _C.matmul(a, b) if _USE_C else _py_matmul(a, b)

    raise TypeMismatchError("dot", "vector o matriz", f"{_shape(a)} y {_shape(b)}")


def _py_matmul(a, b):
    b_is_vec = isinstance(b[0], (int, float))
    if b_is_vec:
        return [sum(a[i][k] * b[k] for k in range(len(b))) for i in range(len(a))]
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


# ── transpose ────────────────────────────────────────────────────────────────

def transpose(m):
    if not isinstance(m, list) or not m:
        raise TypeMismatchError("transpose", "matriz 2D", type(m).__name__)
    if not isinstance(m[0], list):
        raise TypeMismatchError("transpose", "matriz 2D", "vector 1D")
    return _C.transpose(m) if _USE_C else [[m[r][c] for r in range(len(m))] for c in range(len(m[0]))]


# ── reshape / flatten ────────────────────────────────────────────────────────

def reshape(m, rows, cols):
    """Convierte vector o matriz a shape (rows, cols)."""
    flat = flatten(m)
    if len(flat) != rows * cols:
        raise TrezRuntimeError(f"reshape: {len(flat)} elementos no caben en ({rows},{cols})")
    return _C.reshape(flat, rows, cols) if _USE_C else [flat[i*cols:(i+1)*cols] for i in range(rows)]


def flatten(m):
    """Aplana cualquier tensor a vector 1D."""
    if not isinstance(m, list):
        return [m]
    if _USE_C:
        return _C.flatten(m)
    result = []
    for item in m:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


# ── add / sub / scale ────────────────────────────────────────────────────────

def add(a, b):
    """Suma elemento a elemento de dos tensores de igual forma."""
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeMismatchError("add", "listas", f"{type(a).__name__}, {type(b).__name__}")
    if len(a) != len(b):
        raise ShapeMismatchError("add", _shape(a), _shape(b))
    return _C.add_mat(a, b) if _USE_C else _py_elemwise(a, b, lambda x, y: x + y)


def sub(a, b):
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeMismatchError("sub", "listas", f"{type(a).__name__}, {type(b).__name__}")
    if len(a) != len(b):
        raise ShapeMismatchError("sub", _shape(a), _shape(b))
    return _C.sub_mat(a, b) if _USE_C else _py_elemwise(a, b, lambda x, y: x - y)


def scale(m, s):
    """Multiplica todo el tensor por escalar s."""
    if not isinstance(m, list):
        raise TypeMismatchError("scale", "lista", type(m).__name__)
    return _C.scale_mat(m, float(s)) if _USE_C else _py_scale(m, s)


def _py_elemwise(a, b, op):
    if not isinstance(a[0], list):
        return [op(a[i], b[i]) for i in range(len(a))]
    return [_py_elemwise(a[i], b[i], op) for i in range(len(a))]


def _py_scale(m, s):
    if not isinstance(m, list):
        return m * s
    return [_py_scale(x, s) for x in m]


# ── zeros / ones ─────────────────────────────────────────────────────────────

def zeros(rows, cols):
    return _C.zeros(rows, cols) if _USE_C else [[0.0] * cols for _ in range(rows)]


def ones(rows, cols):
    return _C.ones(rows, cols) if _USE_C else [[1.0] * cols for _ in range(rows)]


# ── concat ───────────────────────────────────────────────────────────────────

def concat(a, b, axis=0):
    """Concatena dos matrices. axis=0: apila filas. axis=1: apila columnas."""
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeMismatchError("concat", "listas", f"{type(a).__name__}, {type(b).__name__}")
    if axis == 0:
        if isinstance(a[0], list) and isinstance(b[0], list) and len(a[0]) != len(b[0]):
            raise ShapeMismatchError("concat(axis=0)", _shape(a), _shape(b))
        return a + b
    if axis == 1:
        if len(a) != len(b):
            raise ShapeMismatchError("concat(axis=1)", _shape(a), _shape(b))
        return [a[i] + b[i] for i in range(len(a))]
    raise TrezRuntimeError(f"concat: axis debe ser 0 o 1, recibido {axis}")


# ── mul_elem (Hadamard) ──────────────────────────────────────────────────────

def mul_elem(a, b):
    """Producto Hadamard: multiplicación elemento a elemento."""
    if not isinstance(a, list) or not isinstance(b, list):
        raise TypeMismatchError("mul_elem", "listas", f"{type(a).__name__}, {type(b).__name__}")
    if len(a) != len(b):
        raise ShapeMismatchError("mul_elem", _shape(a), _shape(b))
    return _py_elemwise(a, b, lambda x, y: x * y)


# ── mean ─────────────────────────────────────────────────────────────────────

def mean(t, axis=None):
    """Promedio. axis=None: media global. axis=0: por columna. axis=1: por fila."""
    if not isinstance(t, list) or not t:
        raise TypeMismatchError("mean", "lista no vacía", _shape(t))
    if axis is None:
        flat = flatten(t)
        return sum(flat) / len(flat)
    if not isinstance(t[0], list):
        raise TrezRuntimeError("mean(axis): requiere matriz 2D.")
    rows, cols = len(t), len(t[0])
    if axis == 0:
        return [sum(t[r][c] for r in range(rows)) / rows for c in range(cols)]
    if axis == 1:
        return [sum(row) / cols for row in t]
    raise TrezRuntimeError(f"mean: axis debe ser None, 0 o 1, recibido {axis}")


# ── argmax ───────────────────────────────────────────────────────────────────

def argmax(v):
    """Índice del valor máximo de un vector 1D, o lista de índices por fila si es 2D."""
    if not isinstance(v, list) or not v:
        raise TypeMismatchError("argmax", "lista no vacía", _shape(v))
    if isinstance(v[0], list):
        return [argmax(row) for row in v]
    best_i, best_v = 0, v[0]
    for i in range(1, len(v)):
        if v[i] > best_v:
            best_v, best_i = v[i], i
    return best_i


# ── eye ──────────────────────────────────────────────────────────────────────

def eye(n):
    """Matriz identidad n×n."""
    n = int(n)
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
