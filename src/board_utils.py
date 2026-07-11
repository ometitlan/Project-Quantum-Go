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


# ---------------------------------------------------------------------------
# Representaciones matriciales para GBS / grafos
# ---------------------------------------------------------------------------

def board_to_adjacency(board_np: np.ndarray, scheme: str = 'neutral') -> np.ndarray:
    """Construye la matriz de adyacencia del tablero de Go.

    Cada intersección (i, j) es el nodo i*m + j en la matriz N×N
    (N = filas × columnas).  Solo se conectan intersecciones adyacentes
    en el tablero (4-conectividad).

    Esquemas de peso:
        'neutral'       : w = 1 para toda arista (solo topología)
        'ferromagnetic' : mismo color → +1, colores distintos → −0.5,
                          al menos un vacío → +0.1
        'ising'         : w = s_u × s_v  con s∈{−1,0,+1}

    Returns
    -------
    A : np.ndarray, forma (N, N), simétrica, diagonal cero.
    """
    SPIN = {'B': -1.0, 'W': 1.0, '.': 0.0}
    n, m = board_np.shape
    N = n * m
    A = np.zeros((N, N), dtype=float)

    for i in range(n):
        for j in range(m):
            for di, dj in [(0, 1), (1, 0)]:
                ni, nj = i + di, j + dj
                if not (0 <= ni < n and 0 <= nj < m):
                    continue
                u = i * m + j
                v = ni * m + nj
                su = SPIN[board_np[i, j]]
                sv = SPIN[board_np[ni, nj]]

                if scheme == 'neutral':
                    w = 1.0
                elif scheme == 'ferromagnetic':
                    if su == 0 or sv == 0:
                        w = 0.1
                    elif su == sv:
                        w = 1.0
                    else:
                        w = -0.5
                elif scheme == 'ising':
                    w = su * sv
                else:
                    raise ValueError(f"scheme desconocido: {scheme!r}")

                A[u, v] = w
                A[v, u] = w

    return A


def board_to_cfg(board_np: np.ndarray):
    """Construye el Common Fate Graph (CFG) del tablero.

    Nodos:
        - Un nodo por cada grupo de piedras (BFS sobre misma color).
        - Un nodo por cada intersección vacía.

    Aristas:
        - (grupo, celda_vacía) si la celda vacía es libertad del grupo.
        - (grupo_B, grupo_W) si comparten una celda vacía como libertad.

    Returns
    -------
    A        : np.ndarray, matriz de adyacencia del CFG (n_nodes × n_nodes)
    labels   : list[str], etiqueta de cada nodo ('B_0', 'W_1', 'E_(i,j)')
    node_pos : dict, {nodo_idx: (col, fila)} para visualización
    """
    n, m = board_np.shape

    # 1. Encontrar grupos mediante BFS
    visited = np.zeros((n, m), dtype=bool)
    groups = []   # list of (color, frozenset of positions)

    for i in range(n):
        for j in range(m):
            if board_np[i, j] == '.' or visited[i, j]:
                continue
            color = board_np[i, j]
            # BFS
            queue = [(i, j)]
            cells = []
            while queue:
                ci, cj = queue.pop()
                if visited[ci, cj]:
                    continue
                visited[ci, cj] = True
                cells.append((ci, cj))
                for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < n and 0 <= nj < m:
                        if not visited[ni, nj] and board_np[ni, nj] == color:
                            queue.append((ni, nj))
            groups.append((color, frozenset(cells)))

    # 2. Celdas vacías
    empty_cells = [(i, j) for i in range(n) for j in range(m)
                   if board_np[i, j] == '.']

    # 3. Construir índices de nodos
    n_groups = len(groups)
    n_empty = len(empty_cells)
    n_nodes = n_groups + n_empty

    labels = []
    node_pos = {}
    color_count = {'B': 0, 'W': 0}
    for idx, (color, cells) in enumerate(groups):
        labels.append(f"{color}_{color_count[color]}")
        color_count[color] += 1
        # posición = centroide del grupo
        avg_row = sum(r for r, c in cells) / len(cells)
        avg_col = sum(c for r, c in cells) / len(cells)
        node_pos[idx] = (avg_col, n - 1 - avg_row)

    empty_idx = {cell: n_groups + k for k, cell in enumerate(empty_cells)}
    for k, (ei, ej) in enumerate(empty_cells):
        labels.append(f"E_({ei},{ej})")
        node_pos[n_groups + k] = (ej, n - 1 - ei)

    # 4. Aristas grupo ↔ libertad vacía
    A = np.zeros((n_nodes, n_nodes), dtype=float)

    for g_idx, (color, cells) in enumerate(groups):
        liberties = set()
        for ci, cj in cells:
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < n and 0 <= nj < m and board_np[ni, nj] == '.':
                    liberties.add((ni, nj))
        for lib in liberties:
            e_idx = empty_idx[lib]
            A[g_idx, e_idx] = 1.0
            A[e_idx, g_idx] = 1.0

    return A, labels, node_pos
