"""
minimaxdoz.py — Minimax y Alpha-Beta Pruning para TREZ
Según "Grokking Artificial Intelligence Algorithms" (Hurbans, 2020) Cap 3.
Python puro, sin dependencias externas.

Diseño genérico: el estado es cualquier objeto con:
  - state_score(state)       -> float  (0 si no terminado)
  - state_is_terminal(state) -> bool
  - state_moves(state)       -> list   (movimientos legales)
  - state_apply(state, move) -> nuevo estado (no muta el original)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from errors import TrezRuntimeError

INF = float('inf')
MAX = 1
MIN = -1


# ── Minimax puro ──────────────────────────────────────────────────────────────
def minimax(state_score_fn, state_terminal_fn, state_moves_fn, state_apply_fn,
            state, depth, is_maximizing):
    """
    Minimax clásico.

    Args:
        state_score_fn:    fn(state) -> float
        state_terminal_fn: fn(state) -> bool
        state_moves_fn:    fn(state) -> [move, ...]
        state_apply_fn:    fn(state, move) -> new_state
        state:             estado inicial
        depth:             profundidad máxima (int >= 0)
        is_maximizing:     True = turno MAX, False = turno MIN

    Returns:
        {'score': float, 'move': move | None}
    """
    if depth == 0 or state_terminal_fn(state):
        return {'score': state_score_fn(state), 'move': None}

    moves = state_moves_fn(state)
    if not moves:
        return {'score': state_score_fn(state), 'move': None}

    best_move = None
    if is_maximizing:
        best_score = -INF
        for move in moves:
            child = state_apply_fn(state, move)
            result = minimax(state_score_fn, state_terminal_fn, state_moves_fn,
                             state_apply_fn, child, depth - 1, False)
            if result['score'] > best_score:
                best_score = result['score']
                best_move = move
        return {'score': best_score, 'move': best_move}
    else:
        best_score = INF
        for move in moves:
            child = state_apply_fn(state, move)
            result = minimax(state_score_fn, state_terminal_fn, state_moves_fn,
                             state_apply_fn, child, depth - 1, True)
            if result['score'] < best_score:
                best_score = result['score']
                best_move = move
        return {'score': best_score, 'move': best_move}


# ── Alpha-Beta Pruning ─────────────────────────────────────────────────────────
def alphabeta(state_score_fn, state_terminal_fn, state_moves_fn, state_apply_fn,
              state, depth, is_maximizing, alpha=-INF, beta=INF):
    """
    Minimax con Alpha-Beta pruning.
    Misma interfaz que minimax() pero drasticamente más rápido en árboles profundos.

    Returns:
        {'score': float, 'move': move | None}
    """
    if depth == 0 or state_terminal_fn(state):
        return {'score': state_score_fn(state), 'move': None}

    moves = state_moves_fn(state)
    if not moves:
        return {'score': state_score_fn(state), 'move': None}

    best_move = None
    if is_maximizing:
        best_score = -INF
        for move in moves:
            child = state_apply_fn(state, move)
            result = alphabeta(state_score_fn, state_terminal_fn, state_moves_fn,
                               state_apply_fn, child, depth - 1, False, alpha, beta)
            if result['score'] > best_score:
                best_score = result['score']
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break  # poda beta
        return {'score': best_score, 'move': best_move}
    else:
        best_score = INF
        for move in moves:
            child = state_apply_fn(state, move)
            result = alphabeta(state_score_fn, state_terminal_fn, state_moves_fn,
                               state_apply_fn, child, depth - 1, True, alpha, beta)
            if result['score'] < best_score:
                best_score = result['score']
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break  # poda alfa
        return {'score': best_score, 'move': best_move}


# ── Tic-Tac-Toe helpers (demo built-in) ──────────────────────────────────────
def _ttt_score(board):
    """Heurística básica para Tic-Tac-Toe. +10 X gana, -10 O gana, 0 resto."""
    wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,b,c in wins:
        if board[a] == board[b] == board[c]:
            if board[a] == 'X':
                return 10
            if board[a] == 'O':
                return -10
    return 0


def _ttt_terminal(board):
    if _ttt_score(board) != 0:
        return True
    return all(cell != '.' for cell in board)


def _ttt_moves(board):
    return [i for i, cell in enumerate(board) if cell == '.']


def _ttt_apply(board, move, is_maximizing=None):
    # is_maximizing detectado por número de X vs O
    new_board = list(board)
    x_count = board.count('X')
    o_count = board.count('O')
    new_board[move] = 'X' if x_count == o_count else 'O'
    return new_board


def ttt_best_move(board, depth=9):
    """
    Encuentra el mejor movimiento para Tic-Tac-Toe usando Alpha-Beta.
    board: lista de 9 elementos ('X', 'O', '.')
    Retorna índice del mejor movimiento.
    """
    x_count = board.count('X')
    o_count = board.count('O')
    is_max = (x_count == o_count)  # X = maximizador

    def apply_fn(b, m):
        return _ttt_apply(b, m)

    result = alphabeta(_ttt_score, _ttt_terminal, _ttt_moves, apply_fn,
                       board, depth, is_max)
    return result


# ── API pública genérica ──────────────────────────────────────────────────────
def minimax_search(score_fn, terminal_fn, moves_fn, apply_fn, state, depth, maximizing=True):
    """Wrapper TREZ-friendly para minimax."""
    return minimax(score_fn, terminal_fn, moves_fn, apply_fn, state, depth, maximizing)


def alphabeta_search(score_fn, terminal_fn, moves_fn, apply_fn, state, depth, maximizing=True):
    """Wrapper TREZ-friendly para alpha-beta."""
    return alphabeta(score_fn, terminal_fn, moves_fn, apply_fn, state, depth, maximizing)


def make_game(score_fn, terminal_fn, moves_fn, apply_fn):
    """
    Crea un objeto juego reutilizable.
    Retorna dict con métodos minimax y alphabeta pre-configurados.
    """
    return {
        'score': score_fn,
        'terminal': terminal_fn,
        'moves': moves_fn,
        'apply': apply_fn,
    }


def game_minimax(game, state, depth, maximizing=True):
    return minimax(game['score'], game['terminal'], game['moves'], game['apply'],
                   state, depth, maximizing)


def game_alphabeta(game, state, depth, maximizing=True):
    return alphabeta(game['score'], game['terminal'], game['moves'], game['apply'],
                     state, depth, maximizing)
