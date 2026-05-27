"""
Plotdoz — gráficos nativos en SVG y ASCII. Cero dependencias externas.

Cada función pública genera un archivo .svg (texto plano, abre en cualquier
navegador) y retorna la ruta del archivo generado.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── SVG helpers ───────────────────────────────────────────────────────────────

_SVG_COLORS = [
    "#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#b07aa1",
    "#9c755f", "#ff9da7", "#bab0ac", "#edc948", "#76b7b2",
]

def _svg_open(w, h):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'

def _svg_close():
    return '</svg>'

def _rect(x, y, w, h, fill, stroke="none", sw=1):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'

def _line(x1, y1, x2, y2, color="#aaaaaa", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'

def _circle(cx, cy, r, fill, opacity=0.75):
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" opacity="{opacity}"/>'

def _text(x, y, s, anchor="middle", size=11, fill="#333333"):
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{fill}" font-family="sans-serif">{s}</text>'

def _polyline(pts, color, width=2):
    pts_str = " ".join(f"{px},{py}" for px, py in pts)
    return f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="{width}"/>'

def _scale_vals(vals, lo, hi):
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return [(lo + hi) / 2.0 for _ in vals]
    return [lo + (v - vmin) * (hi - lo) / (vmax - vmin) for v in vals]

def _write_svg(lines, path):
    content = "\n".join(lines)
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _canvas_frame(W, H, pad, title, xlabel, ylabel, elements):
    """Retorna lista de líneas SVG con fondo, ejes, labels y los elementos dados."""
    lines = [_svg_open(W, H)]
    # Fondo
    lines.append(_rect(0, 0, W, H, "#ffffff"))
    # Área de plot
    lines.append(_rect(pad, pad, W - 2 * pad, H - 2 * pad, "#f9f9f9", "#cccccc"))
    # Título
    lines.append(_text(W // 2, pad - 6, title, size=13, fill="#222222"))
    # Etiquetas ejes
    lines.append(_text(W // 2, H - 4, xlabel, size=10))
    lines.append(
        f'<text x="14" y="{H // 2}" text-anchor="middle" font-size="10" '
        f'fill="#333333" font-family="sans-serif" '
        f'transform="rotate(-90,14,{H // 2})">{ylabel}</text>'
    )
    lines.extend(elements)
    lines.append(_svg_close())
    return lines


# ── learning_curve ───────────────────────────────────────────────────────────

def learning_curve(train_losses, val_losses=None, title="Learning Curve",
                   xlabel="Epoch", ylabel="Loss", output_file="learning_curve.svg"):
    if not train_losses:
        raise TrezRuntimeError("Plotdoz.learning_curve(): train_losses vacío.")
    output_file = _ensure_svg(output_file)

    W, H, pad = 600, 360, 50
    pw, ph = W - 2 * pad, H - 2 * pad

    all_vals = [float(v) for v in train_losses]
    if val_losses:
        all_vals += [float(v) for v in val_losses]

    train_f = [float(v) for v in train_losses]
    n = len(train_f)
    xs_px = [pad + i * pw / max(n - 1, 1) for i in range(n)]
    ys_px = _scale_vals(all_vals, pad + ph, pad)
    train_ys = ys_px[:n]

    elems = []
    # Grid horizontal
    for frac in [0.25, 0.5, 0.75]:
        gy = pad + ph * (1 - frac)
        elems.append(_line(pad, gy, pad + pw, gy))

    train_pts = list(zip(xs_px, train_ys))
    elems.append(_polyline(train_pts, "#4e79a7", 2))

    if val_losses:
        val_f = [float(v) for v in val_losses]
        nv = len(val_f)
        xs_v = [pad + i * pw / max(nv - 1, 1) for i in range(nv)]
        val_ys_all = _scale_vals(all_vals, pad + ph, pad)
        val_ys = val_ys_all[n:]
        if len(val_ys) == nv:
            elems.append(_polyline(list(zip(xs_v, val_ys)), "#e15759", 2))

    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] learning_curve guardado en: {output_file}")
    return output_file


# ── multi_curve ──────────────────────────────────────────────────────────────

def multi_curve(series, labels=None, title="Curvas", xlabel="X", ylabel="Y",
                output_file="multi_curve.svg"):
    if not series:
        raise TrezRuntimeError("Plotdoz.multi_curve(): series vacía.")
    output_file = _ensure_svg(output_file)

    W, H, pad = 600, 360, 50
    pw, ph = W - 2 * pad, H - 2 * pad

    all_vals = [float(v) for s in series for v in s]
    vmin, vmax = min(all_vals), max(all_vals)
    if vmax == vmin:
        vmax = vmin + 1

    elems = []
    for gi in [0.25, 0.5, 0.75]:
        gy = pad + ph * (1 - gi)
        elems.append(_line(pad, gy, pad + pw, gy))

    for idx, s in enumerate(series):
        sf = [float(v) for v in s]
        n = len(sf)
        xs = [pad + i * pw / max(n - 1, 1) for i in range(n)]
        ys = [pad + ph - (v - vmin) / (vmax - vmin) * ph for v in sf]
        color = _SVG_COLORS[idx % len(_SVG_COLORS)]
        elems.append(_polyline(list(zip(xs, ys)), color, 2))
        if labels and idx < len(labels):
            lx = pad + pw - 5
            ly = ys[-1]
            elems.append(_text(lx, ly, str(labels[idx]), anchor="end", size=9, fill=color))

    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] multi_curve guardado en: {output_file}")
    return output_file


# ── histogram ────────────────────────────────────────────────────────────────

def histogram(values, title="Histograma", xlabel="Valor", ylabel="Frecuencia",
              bins=20, output_file="histograma.svg"):
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        raise TrezRuntimeError("Plotdoz.histogram(): no hay valores numéricos.")
    output_file = _ensure_svg(output_file)

    bins = int(bins)
    vmin, vmax = min(numeric), max(numeric)
    if vmax == vmin:
        vmax = vmin + 1
    width = (vmax - vmin) / bins
    counts = [0] * bins
    for v in numeric:
        b = int((v - vmin) / width)
        if b >= bins:
            b = bins - 1
        counts[b] += 1

    W, H, pad = 600, 360, 50
    pw, ph = W - 2 * pad, H - 2 * pad
    max_c = max(counts) if counts else 1

    elems = []
    bar_w = pw / bins
    for i, c in enumerate(counts):
        bh = (c / max_c) * ph
        bx = pad + i * bar_w
        by = pad + ph - bh
        elems.append(_rect(bx + 1, by, bar_w - 2, bh, "#4e79a7", "#ffffff", 0.5))

    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] histogram guardado en: {output_file}")
    return output_file


# ── bar_chart ────────────────────────────────────────────────────────────────

def bar_chart(labels, values, title="Barras", xlabel="", ylabel="",
              output_file="barras.svg"):
    if len(labels) != len(values):
        raise TrezRuntimeError("Plotdoz.bar_chart(): labels y values deben tener el mismo tamaño.")
    output_file = _ensure_svg(output_file)

    W, H, pad = 600, 380, 60
    pw, ph = W - 2 * pad, H - 2 * pad
    n = len(values)
    floats = [float(v) for v in values]
    vmin = min(0.0, min(floats))
    vmax = max(floats) if floats else 1.0
    if vmax == vmin:
        vmax = vmin + 1

    elems = []
    bar_w = pw / n * 0.7
    gap = pw / n

    for i, (lbl, val) in enumerate(zip(labels, floats)):
        bh = abs(val - vmin) / (vmax - vmin) * ph
        bx = pad + i * gap + (gap - bar_w) / 2
        by = pad + ph - bh
        color = _SVG_COLORS[i % len(_SVG_COLORS)]
        elems.append(_rect(bx, by, bar_w, bh, color))
        elems.append(_text(bx + bar_w / 2, H - pad + 14, str(lbl), size=9))

    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] bar_chart guardado en: {output_file}")
    return output_file


# ── scatter ──────────────────────────────────────────────────────────────────

def scatter(x_vals, y_vals, title="Scatter", xlabel="X", ylabel="Y",
            color="steelblue", output_file="scatter.svg"):
    if len(x_vals) != len(y_vals):
        raise TrezRuntimeError("Plotdoz.scatter(): x_vals y y_vals deben tener el mismo tamaño.")
    output_file = _ensure_svg(output_file)

    W, H, pad = 600, 360, 50
    pw, ph = W - 2 * pad, H - 2 * pad
    xf = [float(v) for v in x_vals]
    yf = [float(v) for v in y_vals]

    xs_px = _scale_vals(xf, pad, pad + pw)
    ys_px = _scale_vals(yf, pad + ph, pad)

    elems = [_circle(xs_px[i], ys_px[i], 3, "#4e79a7") for i in range(len(xf))]
    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] scatter guardado en: {output_file}")
    return output_file


# ── scatter_classes ───────────────────────────────────────────────────────────

def scatter_classes(x_vals, y_vals, class_ids, class_names=None,
                    title="Scatter por Clase", xlabel="X", ylabel="Y",
                    output_file="scatter_classes.svg"):
    if not (len(x_vals) == len(y_vals) == len(class_ids)):
        raise TrezRuntimeError("Plotdoz.scatter_classes(): listas de distinto tamaño.")
    output_file = _ensure_svg(output_file)

    W, H, pad = 600, 360, 50
    pw, ph = W - 2 * pad, H - 2 * pad
    xf = [float(v) for v in x_vals]
    yf = [float(v) for v in y_vals]
    xs_px = _scale_vals(xf, pad, pad + pw)
    ys_px = _scale_vals(yf, pad + ph, pad)

    elems = []
    for i in range(len(xf)):
        cid = int(class_ids[i])
        color = _SVG_COLORS[cid % len(_SVG_COLORS)]
        elems.append(_circle(xs_px[i], ys_px[i], 3, color))

    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] scatter_classes guardado en: {output_file}")
    return output_file


# ── line_chart ────────────────────────────────────────────────────────────────

def line_chart(x_vals, y_vals, title="Línea", xlabel="X", ylabel="Y",
               output_file="linea.svg"):
    output_file = _ensure_svg(output_file)

    W, H, pad = 600, 360, 50
    pw, ph = W - 2 * pad, H - 2 * pad
    xf = [float(v) for v in x_vals]
    yf = [float(v) for v in y_vals]

    xs_px = _scale_vals(xf, pad, pad + pw)
    ys_px = _scale_vals(yf, pad + ph, pad)

    elems = [_polyline(list(zip(xs_px, ys_px)), "#4e79a7", 2)]
    svg_lines = _canvas_frame(W, H, pad, title, xlabel, ylabel, elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] line_chart guardado en: {output_file}")
    return output_file


# ── confusion_matrix (heatmap) ────────────────────────────────────────────────

def confusion_matrix(y_true, y_pred, class_names=None,
                     title="Confusion Matrix", output_file="confusion_matrix.svg"):
    if len(y_true) != len(y_pred):
        raise TrezRuntimeError("Plotdoz.confusion_matrix(): y_true y y_pred deben tener igual tamaño.")
    output_file = _ensure_svg(output_file)

    n_cls = max(max(int(v) for v in y_true), max(int(v) for v in y_pred)) + 1
    mat = [[0] * n_cls for _ in range(n_cls)]
    for t, p in zip(y_true, y_pred):
        mat[int(t)][int(p)] += 1

    if class_names is None:
        class_names = [str(i) for i in range(n_cls)]

    _render_heatmap(mat, class_names, title, output_file)
    print(f"[Plotdoz] confusion_matrix guardado en: {output_file}")
    return output_file


# ── heatmap ───────────────────────────────────────────────────────────────────

def heatmap(matrix, title="Heatmap", xlabel="", ylabel="",
            output_file="heatmap.svg"):
    if not matrix or not isinstance(matrix[0], list):
        raise TrezRuntimeError("Plotdoz.heatmap(): matrix debe ser lista de listas.")
    output_file = _ensure_svg(output_file)
    _render_heatmap(matrix, None, title, output_file)
    print(f"[Plotdoz] heatmap guardado en: {output_file}")
    return output_file


def _render_heatmap(matrix, labels, title, output_file):
    rows = len(matrix)
    cols = len(matrix[0])
    cell = 50
    pad_l, pad_t = 60, 50
    W = pad_l + cols * cell + 20
    H = pad_t + rows * cell + 40

    flat = [matrix[r][c] for r in range(rows) for c in range(cols)]
    vmin, vmax = min(flat), max(flat)
    if vmax == vmin:
        vmax = vmin + 1

    def _gray(v):
        t = (v - vmin) / (vmax - vmin)
        g = int(230 * (1 - t) + 40 * t)
        return f"rgb({g},{g},{255 - g // 2})"

    def _text_color(v):
        t = (v - vmin) / (vmax - vmin)
        return "#ffffff" if t > 0.5 else "#222222"

    lines = [_svg_open(W, H), _rect(0, 0, W, H, "#ffffff")]
    lines.append(_text(W // 2, 22, title, size=13, fill="#222222"))

    for r in range(rows):
        for c in range(cols):
            val = matrix[r][c]
            rx = pad_l + c * cell
            ry = pad_t + r * cell
            lines.append(_rect(rx, ry, cell, cell, _gray(val), "#ffffff", 0.5))
            lines.append(_text(rx + cell // 2, ry + cell // 2 + 4, str(val),
                               size=10, fill=_text_color(val)))

    if labels:
        for i, lbl in enumerate(labels):
            cx = pad_l + i * cell + cell // 2
            lines.append(_text(cx, pad_t - 6, str(lbl), size=9))
            lines.append(_text(pad_l - 6, pad_t + i * cell + cell // 2 + 4,
                               str(lbl), anchor="end", size=9))

    lines.append(_svg_close())
    _write_svg(lines, output_file)


# ── cluster_scatter (nuevo) ───────────────────────────────────────────────────

def cluster_scatter(X, asignaciones, centroides, title="Clusters",
                    output_file="cluster_scatter.svg"):
    """Scatter de puntos coloreados por cluster + estrella en cada centroide."""
    if len(X) != len(asignaciones):
        raise TrezRuntimeError("Plotdoz.cluster_scatter(): X y asignaciones deben tener igual tamaño.")
    output_file = _ensure_svg(output_file)

    if not isinstance(X[0], list):
        raise TrezRuntimeError("Plotdoz.cluster_scatter(): X debe ser matriz (lista de listas).")

    xf = [float(row[0]) for row in X]
    yf = [float(row[1]) for row in X]
    cx_list = [float(c[0]) for c in centroides]
    cy_list = [float(c[1]) for c in centroides]

    all_x = xf + cx_list
    all_y = yf + cy_list

    W, H, pad = 600, 380, 50
    pw, ph = W - 2 * pad, H - 2 * pad
    xs_px = _scale_vals(all_x, pad, pad + pw)
    ys_px = _scale_vals(all_y, pad + ph, pad)
    n = len(xf)

    elems = []
    for i in range(n):
        cid = int(asignaciones[i])
        color = _SVG_COLORS[cid % len(_SVG_COLORS)]
        elems.append(_circle(xs_px[i], ys_px[i], 3, color))

    for j in range(len(centroides)):
        cx = xs_px[n + j]
        cy = ys_px[n + j]
        elems.append(_text(cx, cy + 1, "✕", size=14, fill="#222222"))

    svg_lines = _canvas_frame(W, H, pad, title, "x", "y", elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] cluster_scatter guardado en: {output_file}")
    return output_file


# ── learning_curve_ascii (nuevo) ─────────────────────────────────────────────

def learning_curve_ascii(train_losses, title="Loss"):
    """Imprime curva de pérdida en terminal. 60 cols × 20 filas."""
    if not train_losses:
        return
    W, H = 60, 20
    vals = [float(v) for v in train_losses]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        vmax = vmin + 1
    n = len(vals)
    grid = [[" "] * W for _ in range(H)]
    for i in range(n):
        col = int(i * (W - 1) / max(n - 1, 1))
        row = int((1 - (vals[i] - vmin) / (vmax - vmin)) * (H - 1))
        row = max(0, min(H - 1, row))
        grid[row][col] = "█"
    print(f"  {title} (min={vmin:.4f} max={vmax:.4f})")
    print("  ┌" + "─" * W + "┐")
    for row in grid:
        print("  │" + "".join(row) + "│")
    print("  └" + "─" * W + "┘")


# ── decision_boundary ────────────────────────────────────────────────────────

def decision_boundary(X, y_true, predict_fn, title="Decision Boundary",
                      resolution=60, output_file="decision_boundary.svg"):
    """
    Visualiza la frontera de decisión de cualquier clasificador.
    predict_fn: función Python que recibe lista de listas y retorna lista de clases.
    X: (n, 2) — solo usa las dos primeras features.
    y_true: etiquetas reales (para pintar los puntos de entrenamiento encima).
    resolution: grilla resolución × resolución (más alto = más lento).
    """
    if not X or not isinstance(X[0], list):
        raise TrezRuntimeError("Plotdoz.decision_boundary(): X debe ser matriz (lista de listas).")
    output_file = _ensure_svg(output_file)

    xf = [float(row[0]) for row in X]
    yf = [float(row[1]) for row in X]
    xmin, xmax = min(xf), max(xf)
    ymin, ymax = min(yf), max(yf)
    xpad = (xmax - xmin) * 0.1 or 1.0
    ypad = (ymax - ymin) * 0.1 or 1.0
    xmin -= xpad; xmax += xpad
    ymin -= ypad; ymax += ypad

    res = int(resolution)
    grid_pts = [
        [xmin + (xmax - xmin) * c / (res - 1), ymin + (ymax - ymin) * r / (res - 1)]
        for r in range(res) for c in range(res)
    ]
    preds = predict_fn(grid_pts)

    W, H, pad = 620, 420, 50
    pw, ph = W - 2 * pad, H - 2 * pad

    def _to_px(gx, gy):
        px = pad + (gx - xmin) / (xmax - xmin) * pw
        py = pad + ph - (gy - ymin) / (ymax - ymin) * ph
        return px, py

    cell_w = pw / (res - 1)
    cell_h = ph / (res - 1)

    elems = []
    unique_cls = sorted(set(int(p) for p in preds))
    for idx, (pt, cl) in enumerate(zip(grid_pts, preds)):
        cid = unique_cls.index(int(cl))
        color = _SVG_COLORS[cid % len(_SVG_COLORS)]
        px, py = _to_px(pt[0], pt[1])
        elems.append(_rect(px - cell_w / 2, py - cell_h / 2, cell_w + 1, cell_h + 1, color, "none"))
        # overlay opacity
        elems[-1] = elems[-1].replace('fill="' + color + '"', f'fill="{color}" opacity="0.25"')

    unique_true = sorted(set(int(v) for v in y_true))
    for xi, yi, cl in zip(xf, yf, y_true):
        cid = unique_true.index(int(cl)) if int(cl) in unique_true else 0
        color = _SVG_COLORS[cid % len(_SVG_COLORS)]
        px, py = _to_px(xi, yi)
        elems.append(_circle(px, py, 4, color, 0.9))

    svg_lines = _canvas_frame(W, H, pad, title, "feature 1", "feature 2", elems)
    _write_svg(svg_lines, output_file)
    print(f"[Plotdoz] decision_boundary guardado en: {output_file}")
    return output_file


# ── Utilidad ──────────────────────────────────────────────────────────────────

def _ensure_svg(path):
    """Cambia extensión a .svg si el caller pasó .png u otra."""
    base, ext = os.path.splitext(str(path))
    if ext.lower() != ".svg":
        return base + ".svg"
    return str(path)
