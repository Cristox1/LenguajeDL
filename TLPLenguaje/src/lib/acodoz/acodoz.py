"""
acodoz.py — Ant Colony Optimization para TREZ
Según "Grokking AI Algorithms" (Hurbans, 2020) Cap 6.
Resuelve problemas de optimización de rutas (TSP-style).
"""
import sys, os, random as _random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def _distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _tour_cost(tour, dist_matrix):
    total = 0
    n = len(tour)
    for i in range(n):
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]
    return total


def run(dist_matrix, num_ants=20, generations=100, alpha=1.0, beta=2.0,
        evaporation=0.5, q=100.0, seed=None):
    """
    ACO para TSP (Travelling Salesman Problem).
    dist_matrix: lista de listas [[d00,d01,...],[d10,d11,...],...]
    alpha: importancia de feromonas
    beta: importancia de visibilidad (1/distancia)
    evaporation: tasa de evaporación [0,1]
    q: constante de depósito de feromona
    Retorna {'best_tour': [i,j,...], 'best_cost': float, 'history': [min_cost/gen]}
    """
    if seed is not None:
        _random.seed(seed)

    n = len(dist_matrix)
    if n < 2:
        raise TrezRuntimeError("Acodoz.run: se necesitan al menos 2 nodos.")

    # Feromonas iniciales
    pheromone = [[1.0] * n for _ in range(n)]

    best_tour = None
    best_cost = float('inf')
    history = []

    for gen in range(generations):
        all_tours = []
        all_costs = []

        for ant in range(num_ants):
            # Construir tour
            start = _random.randint(0, n - 1)
            tour = [start]
            visited = {start}

            while len(tour) < n:
                current = tour[-1]
                probs = []
                candidates = []
                for j in range(n):
                    if j not in visited:
                        d = dist_matrix[current][j]
                        visibility = 1.0 / d if d > 0 else 1e9
                        prob = (pheromone[current][j] ** alpha) * (visibility ** beta)
                        probs.append(prob)
                        candidates.append(j)

                total = sum(probs)
                if total == 0:
                    next_node = _random.choice(candidates)
                else:
                    r = _random.uniform(0, total)
                    acc = 0
                    next_node = candidates[-1]
                    for idx, prob in enumerate(probs):
                        acc += prob
                        if acc >= r:
                            next_node = candidates[idx]
                            break

                tour.append(next_node)
                visited.add(next_node)

            cost = _tour_cost(tour, dist_matrix)
            all_tours.append(tour)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_tour = list(tour)

        # Evaporación
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - evaporation)
                if pheromone[i][j] < 1e-10:
                    pheromone[i][j] = 1e-10

        # Depósito de feromonas
        for tour, cost in zip(all_tours, all_costs):
            deposit = q / cost if cost > 0 else q
            for k in range(len(tour)):
                i = tour[k]
                j = tour[(k + 1) % len(tour)]
                pheromone[i][j] += deposit
                pheromone[j][i] += deposit

        history.append(best_cost)

    return {"best_tour": best_tour, "best_cost": best_cost, "history": history}


def make_dist_matrix(coords):
    """
    Genera matriz de distancias euclidianas desde lista de coordenadas.
    coords: [[x0,y0],[x1,y1],...]
    """
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = _distance(coords[i], coords[j])
    return matrix
