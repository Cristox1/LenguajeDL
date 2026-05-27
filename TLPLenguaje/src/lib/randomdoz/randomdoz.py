"""
Randomdoz — generador pseudoaleatorio nativo, sin dependencias externas.

Implementación: Xorshift64 (Marsaglia, 2003). Estado de 64 bits, periodo 2^64 - 1,
calidad estadística suficiente para inicialización de pesos y barajado de datos.

Toda la API es deterministica tras `seed(s)`: la misma semilla produce la misma
secuencia. Esto es esencial para reproducibilidad de tests y entrenamiento.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


_MASK64 = 0xFFFFFFFFFFFFFFFF
_state = 88172645463325252  # semilla por defecto, debe ser distinta de cero


def seed(s):
    """Fija la semilla del generador. s ≠ 0 (si es 0 se reemplaza por 1)."""
    global _state
    s = int(s)
    if s == 0:
        s = 1
    _state = s & _MASK64


def _next_u64():
    """Avanza el estado un paso de Xorshift64 y retorna el nuevo entero de 64 bits."""
    global _state
    x = _state
    x ^= (x << 13) & _MASK64
    x ^= (x >> 7)
    x ^= (x << 17) & _MASK64
    _state = x & _MASK64
    return _state


def random():
    """Float uniforme en [0, 1). Toma los 53 bits altos del estado para precisión IEEE-754."""
    return (_next_u64() >> 11) * (1.0 / (1 << 53))


def uniform(a, b):
    """Float uniforme en [a, b)."""
    return a + (b - a) * random()


def randint(a, b):
    """Entero en [a, b] inclusive."""
    a, b = int(a), int(b)
    if b < a:
        raise TrezRuntimeError(f"Randomdoz.randint: b ({b}) < a ({a}).")
    return a + int(random() * (b - a + 1))


def choice(seq):
    """Un elemento aleatorio de seq."""
    if not seq:
        raise TrezRuntimeError("Randomdoz.choice: secuencia vacía.")
    return seq[int(random() * len(seq))]


def sample(seq, k):
    """k elementos sin reemplazo. Fisher-Yates parcial."""
    n = len(seq)
    k = int(k)
    if k > n:
        raise TrezRuntimeError(f"Randomdoz.sample: k={k} > len={n}.")
    if k < 0:
        raise TrezRuntimeError(f"Randomdoz.sample: k={k} negativo.")
    pool = list(seq)
    for i in range(k):
        j = i + int(random() * (n - i))
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:k]


def shuffle(seq):
    """Fisher-Yates in-place. Retorna la lista barajada (también muta seq)."""
    n = len(seq)
    for i in range(n - 1, 0, -1):
        j = int(random() * (i + 1))
        seq[i], seq[j] = seq[j], seq[i]
    return seq


def gauss(mu=0.0, sigma=1.0):
    """
    Variable normal N(mu, sigma) por método Box-Muller.
    Usa sqrt_doz, log_doz, cos_doz y PI_DOZ de core_mathdoz — sin librerías externas.
    """
    from lib.mathdoz.core_mathdoz import sqrt_doz, log_doz, cos_doz, PI_DOZ
    u1 = random()
    if u1 < 1e-12:
        u1 = 1e-12
    u2 = random()
    z = sqrt_doz(-2.0 * log_doz(u1)) * cos_doz(2.0 * PI_DOZ * u2)
    return mu + sigma * z
