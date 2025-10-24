"""
board_utils.py
==============
Helpers para crear e intercambiar tableros con GoBoard.
"""

from typing import Iterable
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .go_game_engine import GoBoard


def empty_board(size: int) -> 'GoBoard':
    """Crea un GoBoard vacÃ­o del tamaÃ±o dado."""
    from .go_game_engine import GoBoard
    return GoBoard(size=size)


def board_from_matrix(matrix: Iterable[Iterable[str]]) -> 'GoBoard':
    """Construye un GoBoard inicializado desde una matriz de 'B'/'W'/'.'.

    La matriz debe ser cuadrada.
    """
    arr = np.array(matrix, dtype=str)
    assert arr.ndim == 2 and arr.shape[0] == arr.shape[1], "La matriz debe ser cuadrada"
    size = int(arr.shape[0])
    # Validar smbolos
    valid = {'.', 'B', 'W'}
    if not np.isin(arr, list(valid)).all():
        raise ValueError("La matriz debe contener solo '.', 'B', 'W'")

    from .go_game_engine import GoBoard
    gb = GoBoard(size=size)
    gb.board[:, :] = arr
    # Resetea contadores/historiales bsicos coherentes
    gb.captures = {'B': 0, 'W': 0}
    gb.move_history.clear()
    gb.board_history.clear()
    gb._group_cache.clear()
    gb._liberty_cache.clear()
    return gb


def board_to_numpy(board: 'GoBoard') -> np.ndarray:
    """Devuelve una copia numpy del tablero interno."""
    return np.array(board.board, dtype=str)
