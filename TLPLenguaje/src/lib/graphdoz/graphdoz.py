"""
graphdoz.py — Algoritmos de búsqueda en grafos para TREZ
Implementa BFS, DFS y A* según los pseudocódigos de
"Grokking Artificial Intelligence Algorithms" (Hurbans, 2020)
Caps 2 (BFS/DFS) y 3 (A*)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def _build_path(parent, node):
    path = [node]
    current = node
    while current in parent:
        current = parent[current]
        path.insert(0, current)
    return path


# ── BFS ───────────────────────────────────────────────────────────────────────
def bfs(graph, start, goal):
    """
    Breadth-First Search.
    graph: dict {node: [neighbor, ...]}
    Retorna lista de nodos que forma el camino, o None si no existe.
    """
    if not isinstance(graph, dict):
        raise TrezRuntimeError("Graphdoz.bfs: graph debe ser un diccionario.")
    queue = [start]
    visited = {start}
    parent = {}
    while queue:
        current = queue.pop(0)
        if current == goal:
            return _build_path(parent, current)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    return None


# ── DFS ───────────────────────────────────────────────────────────────────────
def dfs(graph, start, goal):
    """
    Depth-First Search (iterativo con Stack).
    graph: dict {node: [neighbor, ...]}
    Retorna lista de nodos que forma el camino, o None si no existe.
    """
    if not isinstance(graph, dict):
        raise TrezRuntimeError("Graphdoz.dfs: graph debe ser un diccionario.")
    stack = [start]
    visited = set()
    parent = {}
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return _build_path(parent, current)
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                parent[neighbor] = current
                stack.append(neighbor)
    return None


# ── A* ────────────────────────────────────────────────────────────────────────
def astar(graph, start, goal, heuristic=None):
    """
    A* Search.
    graph: dict {node: [(neighbor, edge_cost), ...]}
           o     {node: [neighbor, ...]} (costo uniforme 1)
    heuristic: dict {node: h_value} o None (h=0 => Dijkstra)
    f(n) = g(n) + h(n)
    Retorna lista de nodos del camino, o None si no existe.
    """
    if not isinstance(graph, dict):
        raise TrezRuntimeError("Graphdoz.astar: graph debe ser un diccionario.")
    h = heuristic or {}

    def get_h(node):
        return h.get(node, 0)

    def get_neighbors(node):
        raw = graph.get(node, [])
        if raw and isinstance(raw[0], (list, tuple)):
            return [(n, c) for n, c in raw]
        return [(n, 1) for n in raw]

    # stack ordenado por costo: lista de (cost, node)
    stack = [(get_h(start), start)]
    visited = set()
    parent = {}
    g_cost = {start: 0}

    while stack:
        # pop menor costo (stack ordenado manualmente)
        stack.sort(key=lambda x: x[0])
        f, current = stack.pop(0)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return _build_path(parent, current)

        for neighbor, edge_cost in get_neighbors(current):
            if neighbor in visited:
                continue
            new_g = g_cost[current] + edge_cost
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                parent[neighbor] = current
                f_cost = new_g + get_h(neighbor)
                stack.append((f_cost, neighbor))

    return None


# ── Utilidades de grafo ───────────────────────────────────────────────────────
def make_graph(edges):
    """
    Construye dict de adyacencia desde lista de aristas.
    edges: [[u, v], ...] para grafo no dirigido
           [[u, v, w], ...] para grafo ponderado
    """
    graph = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append((v, w))
        graph[v].append((u, w))
    return graph


def make_directed_graph(edges):
    """
    Construye dict de adyacencia dirigido.
    edges: [[u, v], ...] o [[u, v, w], ...]
    """
    graph = {}
    for edge in edges:
        u, v = edge[0], edge[1]
        w = edge[2] if len(edge) > 2 else 1
        if u not in graph:
            graph[u] = []
        graph[u].append((v, w))
    return graph


def adjacency_matrix(nodes, edges):
    """
    Construye matriz de adyacencia.
    nodes: lista de nodos
    edges: [[u, v], ...]
    Retorna lista de listas (0/1).
    """
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    matrix = [[0] * n for _ in range(n)]
    for edge in edges:
        u, v = edge[0], edge[1]
        if u in idx and v in idx:
            matrix[idx[u]][idx[v]] = 1
            matrix[idx[v]][idx[u]] = 1
    return matrix


def path_cost(graph_weighted, path):
    """
    Calcula costo total de un camino en grafo ponderado.
    graph_weighted: {node: [(neighbor, cost), ...]}
    path: [n0, n1, n2, ...]
    """
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        found = False
        for neighbor, cost in graph_weighted.get(u, []):
            if neighbor == v:
                total += cost
                found = True
                break
        if not found:
            raise TrezRuntimeError(f"Graphdoz.path_cost: no hay arista de {u} a {v}.")
    return total
