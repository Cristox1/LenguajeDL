"""
qlearndoz.py — Q-Learning (Reinforcement Learning) para TREZ
Según "Grokking AI Algorithms" (Hurbans, 2020) Cap 10.
Implementa tabla Q, entrenamiento por episodios y política epsilon-greedy.
"""
import sys, os, random as _random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError


def make_qtable(states, actions, init_val=0.0):
    """
    Crea tabla Q inicializada en init_val.
    states: lista de estados posibles
    actions: lista de acciones posibles
    Retorna dict {estado: {accion: valor}}
    """
    return {s: {a: float(init_val) for a in actions} for s in states}


def choose_action(qtable, state, actions, epsilon=0.1):
    """
    Política epsilon-greedy.
    epsilon: probabilidad de exploración aleatoria
    Retorna acción elegida.
    """
    if _random.random() < epsilon:
        return _random.choice(actions)
    q_vals = qtable.get(state, {})
    if not q_vals:
        return _random.choice(actions)
    return max(q_vals, key=lambda a: q_vals[a])


def update(qtable, state, action, reward, next_state, alpha=0.1, gamma=0.9):
    """
    Actualiza tabla Q con la ecuación de Bellman:
    Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a'(Q(s',a')) - Q(s,a)]
    Retorna nueva qtable (inmutable).
    """
    new_table = {s: dict(qtable[s]) for s in qtable}

    current_q = new_table.get(state, {}).get(action, 0.0)
    next_qs = new_table.get(next_state, {})
    max_next_q = max(next_qs.values()) if next_qs else 0.0

    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)

    if state not in new_table:
        new_table[state] = {}
    new_table[state][action] = new_q
    return new_table


def train(env_step_fn, states, actions, episodes=1000,
          alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995,
          epsilon_min=0.01, max_steps=200, seed=None):
    """
    Entrena agente Q-Learning.
    env_step_fn: fn(state, action) -> (next_state, reward, done)
                 done=True cuando el episodio termina
    Retorna {'qtable': ..., 'rewards': [total_reward/episodio]}
    """
    if seed is not None:
        _random.seed(seed)

    qtable = make_qtable(states, actions)
    rewards_history = []
    eps = epsilon

    for ep in range(episodes):
        state = _random.choice(states)
        total_reward = 0.0

        for step in range(max_steps):
            action = choose_action(qtable, state, actions, eps)
            result = env_step_fn(state, action)
            next_state, reward, done = result[0], result[1], result[2]
            qtable = update(qtable, state, action, reward, next_state, alpha, gamma)
            state = next_state
            total_reward += reward
            if done:
                break

        eps = max(epsilon_min, eps * epsilon_decay)
        rewards_history.append(total_reward)

    return {"qtable": qtable, "rewards": rewards_history}


def get_policy(qtable):
    """
    Extrae política óptima de tabla Q.
    Retorna dict {estado: mejor_accion}
    """
    policy = {}
    for state, actions in qtable.items():
        if actions:
            policy[state] = max(actions, key=lambda a: actions[a])
    return policy


def get_q_value(qtable, state, action):
    """Retorna Q(state, action)."""
    return qtable.get(state, {}).get(action, 0.0)
