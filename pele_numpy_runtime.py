def _is_tensor(x):
    return isinstance(x, dict) and x.get("__np__") is True and "shape" in x and "ndim" in x and "data" in x


def _is_number(x):
    return (isinstance(x, int) or isinstance(x, float)) and not isinstance(x, bool)


_RNG_M = 4294967296
_RNG_A = 1664525
_RNG_C = 1013904223
_rng_state = 1


def _coerce_seed(seed):
    if isinstance(seed, bool):
        return 1 if seed else 0
    if isinstance(seed, int):
        return seed
    if isinstance(seed, float):
        if seed != seed:
            raise Exception("np_seed: seed no puede ser NaN")
        return int(seed)
    raise Exception("np_seed: seed debe ser int, float o bool")


def _rng_next_unit():
    global _rng_state
    _rng_state = (_RNG_A * _rng_state + _RNG_C) % _RNG_M
    return _rng_state / _RNG_M


def _product(vals):
    p = 1
    for v in vals:
        p *= v
    return p


def _make_tensor(shape, data):
    if not isinstance(shape, list):
        raise Exception("shape debe ser una lista de enteros")
    for d in shape:
        if not isinstance(d, int):
            raise Exception("shape debe contener enteros")
        if d < 0:
            raise Exception("shape no puede tener dimensiones negativas")
    expected = _product(shape) if len(shape) > 0 else 1
    if len(data) != expected:
        raise Exception("cantidad de datos no coincide con shape")
    return {"__np__": True, "shape": shape[:], "ndim": len(shape), "data": data[:]}


def _validate_tensor(t):
    if not _is_tensor(t):
        raise Exception("valor no es un tensor mini-numpy")
    shape = t["shape"]
    if not isinstance(shape, list):
        raise Exception("tensor inválido: shape debe ser lista")
    for d in shape:
        if not isinstance(d, int) or d < 0:
            raise Exception("tensor inválido: shape debe contener enteros >= 0")
    if t["ndim"] != len(shape):
        raise Exception("tensor inválido: ndim inconsistente con shape")
    data = t["data"]
    if not isinstance(data, list):
        raise Exception("tensor inválido: data debe ser lista")
    expected = _product(shape) if len(shape) > 0 else 1
    if len(data) != expected:
        raise Exception("tensor inválido: tamaño de data inconsistente con shape")
    for v in data:
        if not _is_number(v):
            raise Exception("tensor inválido: data debe contener números")
    return t


def _shape_from_input(shape_like):
    if isinstance(shape_like, int):
        shape = [shape_like]
    else:
        shape = shape_like
    if not isinstance(shape, list):
        raise Exception("shape debe ser una lista de enteros")
    out = []
    for d in shape:
        if not _is_number(d):
            raise Exception("shape debe contener números enteros")
        di = int(d)
        if di != d:
            raise Exception("shape debe contener números enteros")
        if di < 0:
            raise Exception("shape no puede tener dimensiones negativas")
        out.append(di)
    return out


def np_array(x):
    if _is_tensor(x):
        return _validate_tensor(x)
    if _is_number(x):
        return _make_tensor([], [x])
    if not isinstance(x, list):
        raise Exception("np_array espera escalar numérico, lista 1D o lista 2D")

    if len(x) == 0:
        return _make_tensor([0], [])

    has_list = False
    has_non_list = False
    for e in x:
        if isinstance(e, list):
            has_list = True
        else:
            has_non_list = True

    if has_list and has_non_list:
        raise Exception("np_array: lista con mezcla inválida de escalares y sublistas")

    if has_non_list:
        data = []
        for v in x:
            if not _is_number(v):
                raise Exception("np_array: lista 1D debe contener solo números")
            data.append(v)
        return _make_tensor([len(x)], data)

    rows = len(x)
    cols = None
    data = []
    for row in x:
        if not isinstance(row, list):
            raise Exception("np_array: lista 2D inválida")
        if cols is None:
            cols = len(row)
        elif len(row) != cols:
            raise Exception("np_array: lista 2D debe ser rectangular")
        for v in row:
            if not _is_number(v):
                raise Exception("np_array: lista 2D debe contener solo números")
            data.append(v)
    if cols is None:
        cols = 0
    return _make_tensor([rows, cols], data)


def np_shape(a):
    t = np_array(a)
    return t["shape"][:]


def np_zeros(shape_list):
    shape = _shape_from_input(shape_list)
    size = _product(shape) if len(shape) > 0 else 1
    return _make_tensor(shape, [0 for _ in range(size)])


def np_ones(shape_list):
    shape = _shape_from_input(shape_list)
    size = _product(shape) if len(shape) > 0 else 1
    return _make_tensor(shape, [1 for _ in range(size)])


def np_seed(seed):
    global _rng_state
    seed_int = _coerce_seed(seed)
    _rng_state = seed_int % _RNG_M
    return _rng_state


def np_rand(shape):
    shape_list = _shape_from_input(shape)
    size = _product(shape_list) if len(shape_list) > 0 else 1
    out = []
    for _ in range(size):
        out.append(_rng_next_unit())
    return _make_tensor(shape_list, out)


def np_uniform(low, high, shape):
    if not _is_number(low) or not _is_number(high):
        raise Exception("np_uniform: low y high deben ser escalares numéricos")
    if high <= low:
        raise Exception("np_uniform: high debe ser mayor que low")
    shape_list = _shape_from_input(shape)
    size = _product(shape_list) if len(shape_list) > 0 else 1
    span = high - low
    out = []
    for _ in range(size):
        out.append(low + _rng_next_unit() * span)
    return _make_tensor(shape_list, out)


def _elementwise_binary(a, b, op_name):
    ta = np_array(a)
    tb = np_array(b)

    def apply_op(x, y):
        if op_name == "add":
            return x + y
        if op_name == "sub":
            return x - y
        if op_name == "mul":
            return x * y
        if op_name == "div":
            if y == 0:
                raise Exception("np_div: división por cero")
            return x / y
        raise Exception("operación no soportada")

    if ta["ndim"] == 0 and tb["ndim"] == 0:
        return _make_tensor([], [apply_op(ta["data"][0], tb["data"][0])])

    if ta["ndim"] == 0:
        s = ta["data"][0]
        return _make_tensor(tb["shape"], [apply_op(s, v) for v in tb["data"]])

    if tb["ndim"] == 0:
        s = tb["data"][0]
        return _make_tensor(ta["shape"], [apply_op(v, s) for v in ta["data"]])

    if ta["shape"] != tb["shape"]:
        raise Exception("operación elemento a elemento requiere shapes iguales")
    return _make_tensor(ta["shape"], [apply_op(x, y) for x, y in zip(ta["data"], tb["data"])])


def np_add(a, b):
    return _elementwise_binary(a, b, "add")


def np_sub(a, b):
    return _elementwise_binary(a, b, "sub")


def np_mul(a, b):
    return _elementwise_binary(a, b, "mul")


def np_div(a, b):
    return _elementwise_binary(a, b, "div")


def np_scalar_mul(s, a):
    if not _is_number(s):
        raise Exception("np_scalar_mul: primer argumento debe ser escalar numérico")
    return np_mul(s, a)


def np_transpose(a):
    t = np_array(a)
    if t["ndim"] != 2:
        raise Exception("np_transpose: solo soporta tensores 2D")
    rows = t["shape"][0]
    cols = t["shape"][1]
    out = []
    for c in range(cols):
        for r in range(rows):
            out.append(t["data"][r * cols + c])
    return _make_tensor([cols, rows], out)


def np_matmul(a, b):
    ta = np_array(a)
    tb = np_array(b)
    na = ta["ndim"]
    nb = tb["ndim"]

    if na == 0 or nb == 0:
        raise Exception("np_matmul: no soporta tensores escalares (0D)")

    if na not in [1, 2] or nb not in [1, 2]:
        raise Exception("np_matmul: solo soporta combinaciones 1D/2D")

    if na == 1 and nb == 1:
        n = ta["shape"][0]
        if tb["shape"][0] != n:
            raise Exception("np_matmul: dimensiones incompatibles para vector·vector")
        acc = 0
        for i in range(n):
            acc += ta["data"][i] * tb["data"][i]
        return _make_tensor([], [acc])

    if na == 2 and nb == 1:
        m = ta["shape"][0]
        k = ta["shape"][1]
        if tb["shape"][0] != k:
            raise Exception("np_matmul: dimensiones incompatibles para matriz·vector")
        out = []
        for i in range(m):
            acc = 0
            base = i * k
            for p in range(k):
                acc += ta["data"][base + p] * tb["data"][p]
            out.append(acc)
        return _make_tensor([m], out)

    if na == 1 and nb == 2:
        k = ta["shape"][0]
        kb = tb["shape"][0]
        n = tb["shape"][1]
        if kb != k:
            raise Exception("np_matmul: dimensiones incompatibles para vector·matriz")
        out = []
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += ta["data"][p] * tb["data"][p * n + j]
            out.append(acc)
        return _make_tensor([n], out)

    m = ta["shape"][0]
    k = ta["shape"][1]
    kb = tb["shape"][0]
    n = tb["shape"][1]
    if kb != k:
        raise Exception("np_matmul: dimensiones incompatibles para matriz·matriz")
    out = []
    for i in range(m):
        for j in range(n):
            acc = 0
            for p in range(k):
                acc += ta["data"][i * k + p] * tb["data"][p * n + j]
            out.append(acc)
    return _make_tensor([m, n], out)


def np_sum(a, axis):
    t = np_array(a)
    if not _is_number(axis):
        raise Exception("np_sum: axis debe ser entero")
    ax = int(axis)
    if ax != axis:
        raise Exception("np_sum: axis debe ser entero")

    if t["ndim"] == 0:
        if ax not in [-1, 0]:
            raise Exception("np_sum: axis inválido para tensor 0D (usar -1 o 0)")
        return _make_tensor([], [t["data"][0]])

    if t["ndim"] == 1:
        if ax not in [-1, 0]:
            raise Exception("np_sum: axis inválido para tensor 1D (usar -1 o 0)")
        acc = 0
        for v in t["data"]:
            acc += v
        return _make_tensor([], [acc])

    if t["ndim"] == 2:
        rows = t["shape"][0]
        cols = t["shape"][1]
        if ax == -1:
            acc = 0
            for v in t["data"]:
                acc += v
            return _make_tensor([], [acc])
        if ax == 0:
            out = [0 for _ in range(cols)]
            for r in range(rows):
                base = r * cols
                for c in range(cols):
                    out[c] += t["data"][base + c]
            return _make_tensor([cols], out)
        if ax == 1:
            out = []
            for r in range(rows):
                base = r * cols
                acc = 0
                for c in range(cols):
                    acc += t["data"][base + c]
                out.append(acc)
            return _make_tensor([rows], out)
        raise Exception("np_sum: axis inválido para tensor 2D (usar -1, 0 o 1)")

    raise Exception("np_sum: solo soporta tensores 0D, 1D o 2D")


def np_argmax(a, axis):
    t = np_array(a)
    if not _is_number(axis):
        raise Exception("np_argmax: axis debe ser entero")
    ax = int(axis)
    if ax != axis:
        raise Exception("np_argmax: axis debe ser entero")

    if t["ndim"] == 0:
        raise Exception("np_argmax: no está definido para tensores escalares (0D)")

    if t["ndim"] == 1:
        if ax not in [-1, 0]:
            raise Exception("np_argmax: axis inválido para tensor 1D (usar -1 o 0)")
        if len(t["data"]) == 0:
            raise Exception("np_argmax: no definido para vector vacío")
        best_i = 0
        best_v = t["data"][0]
        for i in range(1, len(t["data"])):
            if t["data"][i] > best_v:
                best_v = t["data"][i]
                best_i = i
        return _make_tensor([], [best_i])

    if t["ndim"] == 2:
        rows = t["shape"][0]
        cols = t["shape"][1]
        if rows == 0 or cols == 0:
            raise Exception("np_argmax: no definido para matriz vacía")
        if ax == 0:
            out = []
            for c in range(cols):
                best_r = 0
                best_v = t["data"][c]
                for r in range(1, rows):
                    v = t["data"][r * cols + c]
                    if v > best_v:
                        best_v = v
                        best_r = r
                out.append(best_r)
            return _make_tensor([cols], out)
        if ax == 1:
            out = []
            for r in range(rows):
                base = r * cols
                best_c = 0
                best_v = t["data"][base]
                for c in range(1, cols):
                    v = t["data"][base + c]
                    if v > best_v:
                        best_v = v
                        best_c = c
                out.append(best_c)
            return _make_tensor([rows], out)
        raise Exception("np_argmax: axis inválido para tensor 2D (usar 0 o 1)")

    raise Exception("np_argmax: solo soporta tensores 1D o 2D")
