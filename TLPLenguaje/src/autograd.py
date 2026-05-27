import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lib.mathdoz.core_mathdoz import sqrt_doz, exp_doz, log_doz
from errors import TrezRuntimeError


def _zeros_like(data):
    if not isinstance(data, list):
        return 0.0
    return [_zeros_like(x) for x in data]


def _add_elem(a, b):
    if not isinstance(a, list) and not isinstance(b, list):
        return a + b
    if not isinstance(a, list):
        a = [a] * len(b)
    if not isinstance(b, list):
        b = [b] * len(a)
    return [_add_elem(ai, bi) for ai, bi in zip(a, b)]


def _mul_elem(a, b):
    if not isinstance(a, list) and not isinstance(b, list):
        return a * b
    if not isinstance(a, list):
        a = [a] * len(b)
    if not isinstance(b, list):
        b = [b] * len(a)
    return [_mul_elem(ai, bi) for ai, bi in zip(a, b)]


def _scale(data, scalar):
    if not isinstance(data, list):
        return data * scalar
    return [_scale(x, scalar) for x in data]


def _sum_all(data):
    if not isinstance(data, list):
        return data
    return sum(_sum_all(x) for x in data)


def _transpose(m):
    if not isinstance(m[0], list):
        return [[v] for v in m]
    rows, cols = len(m), len(m[0])
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def _matmul(a, b):
    """a: (m,n)  b: (n,p)  →  (m,p). Ambos deben ser listas de listas."""
    m, n = len(a), len(a[0])
    b_cols = len(b[0]) if isinstance(b[0], list) else 1
    if isinstance(b[0], list):
        p = b_cols
        return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
    else:
        return [sum(a[i][k] * b[k] for k in range(n)) for i in range(m)]


class Tensor:
    """
    Tensor con soporte para autograd sobre escalares y matrices (listas anidadas).
    data puede ser un número, una lista 1D o una lista de listas 2D.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = None          # None hasta que backward() lo inicialice
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # ------------------------------------------------------------------
    # Forma
    # ------------------------------------------------------------------

    @property
    def shape(self):
        def _s(d):
            if not isinstance(d, list):
                return ()
            return (len(d),) + _s(d[0])
        return _s(self.data)

    # ------------------------------------------------------------------
    # Operaciones aritméticas escalares y vectoriales
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(_add_elem(self.data, other.data), (self, other), '+')

        def _backward():
            _accum(self, out.grad)
            _accum(other, out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(_add_elem(self.data, _scale(other.data, -1.0)), (self, other), '-')

        def _backward():
            _accum(self, out.grad)
            _accum(other, _scale(out.grad, -1.0))
        out._backward = _backward
        return out

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(_mul_elem(self.data, other.data), (self, other), '*')

        def _backward():
            _accum(self, _mul_elem(other.data, out.grad))
            _accum(other, _mul_elem(self.data, out.grad))
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * Tensor(-1.0)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __pow__(self, power):
        power = float(power)

        def _pow_data(d, p):
            if not isinstance(d, list):
                return d ** p
            return [_pow_data(x, p) for x in d]

        out = Tensor(_pow_data(self.data, power), (self,), f'**{power}')

        def _backward():
            def _pow_grad(d, g, p):
                if not isinstance(d, list):
                    return p * (d ** (p - 1)) * g
                return [_pow_grad(di, gi, p) for di, gi in zip(d, g)]
            _accum(self, _pow_grad(self.data, out.grad, power))
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Álgebra lineal
    # ------------------------------------------------------------------

    def matmul(self, other):
        """Multiplicación matricial (m,n) @ (n,p) → (m,p) con backward."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(_matmul(self.data, other.data), (self, other), 'matmul')

        def _backward():
            # grad_self  = out.grad @ other.T
            # grad_other = self.T  @ out.grad
            g = out.grad if isinstance(out.grad, list) else [[out.grad]]
            _accum(self, _matmul(g, _transpose(other.data)))
            _accum(other, _matmul(_transpose(self.data), g))
        out._backward = _backward
        return out

    def T(self):
        """Transposición 2D con backward."""
        out = Tensor(_transpose(self.data), (self,), 'T')

        def _backward():
            _accum(self, _transpose(out.grad))
        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activaciones
    # ------------------------------------------------------------------

    def relu(self):
        def _relu(d):
            if not isinstance(d, list):
                return max(0.0, d)
            return [_relu(x) for x in d]

        out = Tensor(_relu(self.data), (self,), 'relu')

        def _backward():
            def _relu_grad(d, g):
                if not isinstance(d, list):
                    return g if d > 0 else 0.0
                return [_relu_grad(di, gi) for di, gi in zip(d, g)]
            _accum(self, _relu_grad(self.data, out.grad))
        out._backward = _backward
        return out

    def sigmoid(self):
        def _sig(d):
            if not isinstance(d, list):
                return 1.0/(1.0+exp_doz(-d)) if d >= 0 else (exp_doz(d)/(1.0+exp_doz(d)))
            return [_sig(x) for x in d]
        s = _sig(self.data)
        out = Tensor(s, (self,), 'sigmoid')
        def _backward():
            def _sig_grad(sv, g):
                if not isinstance(sv, list):
                    return g * sv * (1.0 - sv)
                return [_sig_grad(si, gi) for si, gi in zip(sv, g)]
            _accum(self, _sig_grad(out.data, out.grad))
        out._backward = _backward
        return out

    def tanh(self):
        def _tanh(d):
            if not isinstance(d, list):
                if d > 20: return 1.0
                if d < -20: return -1.0
                e = exp_doz(2*d); return (e-1)/(e+1)
            return [_tanh(x) for x in d]
        t = _tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            def _tanh_grad(tv, g):
                if not isinstance(tv, list):
                    return g * (1.0 - tv*tv)
                return [_tanh_grad(ti, gi) for ti, gi in zip(tv, g)]
            _accum(self, _tanh_grad(out.data, out.grad))
        out._backward = _backward
        return out

    def log(self):
        def _log(d):
            if not isinstance(d, list):
                return log_doz(d) if d > 0 else log_doz(1e-12)
            return [_log(x) for x in d]
        out = Tensor(_log(self.data), (self,), 'log')
        def _backward():
            def _log_grad(d, g):
                if not isinstance(d, list):
                    return g / (d if d > 0 else 1e-12)
                return [_log_grad(di, gi) for di, gi in zip(d, g)]
            _accum(self, _log_grad(self.data, out.grad))
        out._backward = _backward
        return out

    def exp(self):
        def _exp(d):
            if not isinstance(d, list):
                return exp_doz(min(d, 700))
            return [_exp(x) for x in d]
        out = Tensor(_exp(self.data), (self,), 'exp')
        def _backward():
            def _exp_grad(ev, g):
                if not isinstance(ev, list):
                    return g * ev
                return [_exp_grad(ei, gi) for ei, gi in zip(ev, g)]
            _accum(self, _exp_grad(out.data, out.grad))
        out._backward = _backward
        return out

    def sum(self):
        """Reduce todo el tensor a escalar."""
        s = _sum_all(self.data)
        out = Tensor(s, (self,), 'sum')
        def _backward():
            _accum(self, _ones_like(self.data) if not isinstance(out.grad, list)
                   else out.grad)
        out._backward = _backward
        return out

    def mean(self):
        n = _count_elems(self.data)
        return self.sum() * (1.0/n)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        # Gradiente inicial: 1 para escalar, matriz de unos para tensor
        self.grad = _ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = None
        for child in self._prev:
            child.zero_grad()


# ------------------------------------------------------------------
# Helpers de acumulación de gradientes
# ------------------------------------------------------------------

def _ones_like(data):
    if not isinstance(data, list):
        return 1.0
    return [_ones_like(x) for x in data]


def _count_elems(data):
    if not isinstance(data, list):
        return 1
    return sum(_count_elems(x) for x in data)


def _accum(tensor, grad_val):
    """Acumula grad_val en tensor.grad (suma si ya existe)."""
    if tensor.grad is None:
        tensor.grad = grad_val
    else:
        tensor.grad = _add_elem(tensor.grad, grad_val)


# ------------------------------------------------------------------
# Constructor helpers (expuestos al visitor como Autodoz.*)
# ------------------------------------------------------------------

def tensor(data):
    """Crea un Tensor desde datos Python (número, lista, lista de listas)."""
    return Tensor(data)

def tensor_zeros(rows, cols=None):
    if cols is None:
        return Tensor([0.0]*rows)
    return Tensor([[0.0]*cols for _ in range(rows)])

def tensor_ones(rows, cols=None):
    if cols is None:
        return Tensor([1.0]*rows)
    return Tensor([[1.0]*cols for _ in range(rows)])
