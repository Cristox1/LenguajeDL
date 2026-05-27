"""
psodoz.py — Particle Swarm Optimization para TREZ
Según "Grokking AI Algorithms" (Hurbans, 2020) Cap 7.
Optimización continua: minimiza o maximiza una función objetivo.
"""
import sys, os, random as _random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def run(fitness_fn, dimensions, num_particles=30, iterations=100,
        bounds=None, w=0.7, c1=1.5, c2=1.5,
        minimize=True, seed=None):
    """
    PSO estándar.
    fitness_fn: función callable que recibe lista de floats (posición) y retorna float
    dimensions: número de dimensiones del espacio
    bounds: [[min_d0, max_d0], [min_d1, max_d1], ...] o None (usa [-5,5])
    w: inercia
    c1: coeficiente cognitivo (atracción al mejor personal)
    c2: coeficiente social (atracción al mejor global)
    minimize: True para minimizar, False para maximizar
    Retorna {'best_position': [...], 'best_score': float, 'history': [best_score/iter]}
    """
    if seed is not None:
        _random.seed(seed)

    if bounds is None:
        bounds = [[-5.0, 5.0]] * dimensions

    if len(bounds) != dimensions:
        raise TrezRuntimeError("Psodoz.run: bounds debe tener una entrada por dimensión.")

    sign = 1 if minimize else -1

    def score(pos):
        raw = fitness_fn(pos)
        return sign * raw

    # Inicializar partículas
    positions = []
    velocities = []
    personal_best_pos = []
    personal_best_score = []

    for _ in range(num_particles):
        pos = [_random.uniform(bounds[d][0], bounds[d][1]) for d in range(dimensions)]
        vel = [_random.uniform(-1, 1) for _ in range(dimensions)]
        s = score(pos)
        positions.append(pos)
        velocities.append(vel)
        personal_best_pos.append(list(pos))
        personal_best_score.append(s)

    # Mejor global
    best_idx = personal_best_score.index(min(personal_best_score))
    global_best_pos = list(personal_best_pos[best_idx])
    global_best_score = personal_best_score[best_idx]

    history = [sign * global_best_score]

    for _ in range(iterations):
        for i in range(num_particles):
            # Actualizar velocidad
            r1 = [_random.random() for _ in range(dimensions)]
            r2 = [_random.random() for _ in range(dimensions)]
            new_vel = []
            for d in range(dimensions):
                v = (w * velocities[i][d]
                     + c1 * r1[d] * (personal_best_pos[i][d] - positions[i][d])
                     + c2 * r2[d] * (global_best_pos[d] - positions[i][d]))
                new_vel.append(v)
            velocities[i] = new_vel

            # Actualizar posición y aplicar bounds
            new_pos = []
            for d in range(dimensions):
                p = positions[i][d] + velocities[i][d]
                p = max(bounds[d][0], min(bounds[d][1], p))
                new_pos.append(p)
            positions[i] = new_pos

            # Actualizar mejor personal
            s = score(positions[i])
            if s < personal_best_score[i]:
                personal_best_score[i] = s
                personal_best_pos[i] = list(positions[i])

                # Actualizar mejor global
                if s < global_best_score:
                    global_best_score = s
                    global_best_pos = list(positions[i])

        history.append(sign * global_best_score)

    return {
        "best_position": global_best_pos,
        "best_score": sign * global_best_score,
        "history": history,
    }
