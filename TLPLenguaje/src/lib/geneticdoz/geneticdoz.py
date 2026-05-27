"""
geneticdoz.py — Algoritmo Genético para TREZ
Implementa el ciclo completo según "Grokking AI Algorithms" (Hurbans, 2020) Caps 4-5:
  generate_population -> evaluate_fitness -> select_parents
  -> crossover -> mutate -> next_generation -> run
"""
import sys, os, random as _random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


# ── Paso 1: Generar población inicial ────────────────────────────────────────
def generate_population(population_size, individual_size):
    """
    Crea población de individuos con genes binarios (0/1) aleatorios.
    Equivale al pseudocódigo generate_initial_population del libro.
    """
    population = []
    for _ in range(population_size):
        individual = [_random.randint(0, 1) for _ in range(individual_size)]
        population.append(individual)
    return population


# ── Paso 2: Evaluar fitness ──────────────────────────────────────────────────
def evaluate_fitness(population, fitness_fn):
    """
    Aplica fitness_fn a cada individuo.
    Retorna lista de [individuo, score] ordenada de mayor a menor score.
    """
    scored = [[ind, fitness_fn(ind)] for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ── Paso 3: Selección de padres (torneo o top-k) ─────────────────────────────
def select_parents(scored_population, num_parents=2, method="tournament", tournament_size=3):
    """
    Selecciona padres para reproducción.
    method: "tournament" | "top" | "roulette"
    """
    n = len(scored_population)
    if n < 2:
        raise TrezRuntimeError("Geneticdoz.select_parents: población muy pequeña.")

    if method == "top":
        return [scored_population[i][0] for i in range(min(num_parents, n))]

    elif method == "roulette":
        scores = [max(0.0, s[1]) for s in scored_population]
        total = sum(scores)
        parents = []
        for _ in range(num_parents):
            r = _random.uniform(0, total) if total > 0 else 0
            acc = 0
            chosen = scored_population[0][0]
            for ind, score in scored_population:
                acc += score
                if acc >= r:
                    chosen = ind
                    break
            parents.append(chosen)
        return parents

    else:  # tournament (default del libro)
        parents = []
        for _ in range(num_parents):
            candidates = _random.sample(scored_population, min(tournament_size, n))
            winner = max(candidates, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents


# ── Paso 4: Crossover (single-point) ─────────────────────────────────────────
def crossover(parent1, parent2, point=None):
    """
    Crossover de un punto. Retorna [hijo1, hijo2].
    Si point=None elige punto aleatorio.
    """
    size = len(parent1)
    if size != len(parent2):
        raise TrezRuntimeError("Geneticdoz.crossover: padres de distinto tamaño.")
    if point is None:
        point = _random.randint(1, size - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return [child1, child2]


def crossover_uniform(parent1, parent2):
    """
    Crossover uniforme: cada gen elegido al azar de uno u otro padre.
    """
    size = len(parent1)
    child1 = [parent1[i] if _random.random() < 0.5 else parent2[i] for i in range(size)]
    child2 = [parent2[i] if _random.random() < 0.5 else parent1[i] for i in range(size)]
    return [child1, child2]


# ── Paso 5: Mutación ─────────────────────────────────────────────────────────
def mutate(individual, mutation_rate=0.01):
    """
    Flip bit con probabilidad mutation_rate por gen.
    """
    return [1 - gene if _random.random() < mutation_rate else gene
            for gene in individual]


def mutate_real(individual, mutation_rate=0.1, sigma=0.1):
    """
    Mutación gaussiana para cromosomas de valores reales.
    """
    import math
    result = []
    for gene in individual:
        if _random.random() < mutation_rate:
            noise = _random.gauss(0, sigma)
            result.append(gene + noise)
        else:
            result.append(gene)
    return result


# ── Paso 6: Poblar siguiente generación ──────────────────────────────────────
def next_generation(scored_population, fitness_fn, mutation_rate=0.01,
                    elitism=2, method="tournament"):
    """
    Genera la siguiente generación completa.
    elitism: cuántos mejores individuos pasan sin cambios.
    """
    pop_size = len(scored_population)
    new_pop = []

    # Elitismo: conservar los mejores
    for i in range(min(elitism, pop_size)):
        new_pop.append(scored_population[i][0])

    # Rellenar con hijos
    while len(new_pop) < pop_size:
        parents = select_parents(scored_population, num_parents=2, method=method)
        children = crossover(parents[0], parents[1])
        for child in children:
            if len(new_pop) < pop_size:
                new_pop.append(mutate(child, mutation_rate))

    scored = evaluate_fitness(new_pop, fitness_fn)
    return scored


# ── Ciclo completo ────────────────────────────────────────────────────────────
def run(fitness_fn, individual_size, population_size=50, generations=100,
        mutation_rate=0.01, elitism=2, method="tournament", seed=None):
    """
    Ejecuta el algoritmo genético completo.
    Retorna {'best': individuo, 'score': float, 'history': [mejor_score_por_gen]}
    """
    if seed is not None:
        _random.seed(seed)

    population = generate_population(population_size, individual_size)
    scored = evaluate_fitness(population, fitness_fn)
    history = [scored[0][1]]

    for _ in range(generations):
        scored = next_generation(scored, fitness_fn, mutation_rate, elitism, method)
        history.append(scored[0][1])

    return {"best": scored[0][0], "score": scored[0][1], "history": history}


# ── Codificación de orden / permutación (Cap 5: TSP, scheduling) ─────────────
def generate_population_order(population_size, items):
    """
    Genera población con codificación de orden (permutaciones).
    items: lista de elementos a permutar (p.ej., ciudades para TSP).
    """
    population = []
    for _ in range(population_size):
        individual = list(items)
        _random.shuffle(individual)
        population.append(individual)
    return population


def crossover_order(parent1, parent2):
    """
    Order Crossover (OX1) para permutaciones — preserva orden relativo.
    Retorna dos hijos como permutaciones válidas.
    """
    n = len(parent1)
    a, b = sorted(_random.sample(range(n), 2))
    child1 = [None] * n
    child2 = [None] * n

    child1[a:b] = parent1[a:b]
    child2[a:b] = parent2[a:b]

    def _fill(child, donor):
        pos = b
        for gene in donor[b:] + donor[:b]:
            if gene not in child:
                child[pos % n] = gene
                pos += 1

    _fill(child1, parent2)
    _fill(child2, parent1)
    return child1, child2


def mutate_order(individual, mutation_rate=0.01):
    """
    Mutación por intercambio (swap mutation) para codificación de orden.
    """
    individual = list(individual)
    for i in range(len(individual)):
        if _random.random() < mutation_rate:
            j = _random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual


def run_order(fitness_fn, items, population_size=50, generations=100,
              mutation_rate=0.05, elitism=2, seed=None):
    """
    GA con codificación de orden para problemas de permutación (TSP, scheduling).
    items: lista de elementos a permutar.
    fitness_fn: fn(permutation) -> float (minimizar o maximizar según convención)
    """
    if seed is not None:
        _random.seed(seed)

    population = generate_population_order(population_size, items)
    scored = sorted([[ind, fitness_fn(ind)] for ind in population],
                    key=lambda x: x[1], reverse=True)
    history = [scored[0][1]]

    for _ in range(generations):
        elite = [s[0] for s in scored[:elitism]]
        new_pop = list(elite)
        while len(new_pop) < population_size:
            # Selección por torneo
            def tournament():
                cands = _random.sample(scored, min(3, len(scored)))
                return max(cands, key=lambda x: x[1])[0]
            p1, p2 = tournament(), tournament()
            c1, c2 = crossover_order(p1, p2)
            new_pop.append(mutate_order(c1, mutation_rate))
            if len(new_pop) < population_size:
                new_pop.append(mutate_order(c2, mutation_rate))
        scored = sorted([[ind, fitness_fn(ind)] for ind in new_pop],
                        key=lambda x: x[1], reverse=True)
        history.append(scored[0][1])

    return {'best': scored[0][0], 'score': scored[0][1], 'history': history}


# ── Codificación real / continua (Cap 5: funciones continuas) ─────────────────
def generate_population_real(population_size, dimensions, bounds):
    """
    Genera población con codificación real (valores continuos).
    bounds: [[min, max], ...] por dimensión.
    """
    population = []
    for _ in range(population_size):
        individual = [_random.uniform(bounds[d][0], bounds[d][1]) for d in range(dimensions)]
        population.append(individual)
    return population


def run_real(fitness_fn, dimensions, bounds, population_size=50, generations=100,
             mutation_rate=0.1, mutation_strength=0.1, elitism=2, seed=None):
    """
    GA con codificación real para optimización de funciones continuas.
    bounds: [[min_d, max_d], ...] por dimensión.
    mutation_strength: fracción del rango aplicada como ruido gaussiano.
    """
    if seed is not None:
        _random.seed(seed)

    import math

    population = generate_population_real(population_size, dimensions, bounds)
    scored = sorted([[ind, fitness_fn(ind)] for ind in population],
                    key=lambda x: x[1], reverse=True)
    history = [scored[0][1]]

    def _crossover_real(p1, p2):
        # Arithmetic crossover
        alpha = _random.random()
        c1 = [alpha * p1[i] + (1 - alpha) * p2[i] for i in range(dimensions)]
        c2 = [(1 - alpha) * p1[i] + alpha * p2[i] for i in range(dimensions)]
        return c1, c2

    def _mutate_real_gauss(ind):
        new = []
        for d in range(dimensions):
            r = bounds[d][1] - bounds[d][0]
            val = ind[d] + _random.gauss(0, mutation_strength * r) if _random.random() < mutation_rate else ind[d]
            new.append(max(bounds[d][0], min(bounds[d][1], val)))
        return new

    for _ in range(generations):
        elite = [s[0] for s in scored[:elitism]]
        new_pop = list(elite)
        while len(new_pop) < population_size:
            cands = _random.sample(scored, min(3, len(scored)))
            p1 = max(cands, key=lambda x: x[1])[0]
            cands = _random.sample(scored, min(3, len(scored)))
            p2 = max(cands, key=lambda x: x[1])[0]
            c1, c2 = _crossover_real(p1, p2)
            new_pop.append(_mutate_real_gauss(c1))
            if len(new_pop) < population_size:
                new_pop.append(_mutate_real_gauss(c2))
        scored = sorted([[ind, fitness_fn(ind)] for ind in new_pop],
                        key=lambda x: x[1], reverse=True)
        history.append(scored[0][1])

    return {'best': scored[0][0], 'score': scored[0][1], 'history': history}


# ── Fitness para Knapsack (ejemplo del libro) ─────────────────────────────────
def knapsack_fitness(weights, values, capacity):
    """
    Retorna función fitness para el Problema de la Mochila.
    weights: [w1, w2, ...], values: [v1, v2, ...], capacity: límite de peso
    """
    def fitness(individual):
        total_w = sum(individual[i] * weights[i] for i in range(len(individual)))
        total_v = sum(individual[i] * values[i] for i in range(len(individual)))
        if total_w > capacity:
            return 0  # solución inválida
        return float(total_v)
    return fitness
