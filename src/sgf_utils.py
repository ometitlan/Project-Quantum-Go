"""
sgf_utils.py
=============
Utilidades para cargar partidas SGF, extraer movimientos y reproducirlos
en un GoBoard del motor existente.

Nota: Maneja 'pass' (coords vacÃ­as) y metadatos comunes (PB, PW, BR, WR, DT, KM, SZ, RE).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import sgf  # type: ignore
except Exception:  # pragma: no cover
    sgf = None

from typing import TYPE_CHECKING
from .board_utils import board_to_numpy as _board_to_numpy

if TYPE_CHECKING:  # evitar import en tiempo de carga (circular)
    from .go_game_engine import GoBoard


def _sgf_to_coords(sgf_coords: str) -> Optional[Tuple[int, int]]:
    """Convierte coordenadas SGF ('aa', 'pd', ...) a (fila, columna) 0-index.

    Devuelve None si es un 'pass' (cadena vacÃ­a).
    """
    if sgf_coords is None:
        return None
    if len(sgf_coords) == 0:
        return None
    if len(sgf_coords) != 2:
        raise ValueError(f"Coordenadas SGF invÃ¡lidas: {sgf_coords}")
    a = ord('a')
    col = ord(sgf_coords[0]) - a
    row = ord(sgf_coords[1]) - a
    if col < 0 or row < 0:
        raise ValueError(f"Coordenadas SGF invÃ¡lidas: {sgf_coords}")
    return (row, col)


def extract_moves_and_info(file_path: str) -> Tuple[List[Dict], Optional[str], Dict]:
    """Extrae movimientos, ganador y metadatos desde un archivo SGF.

    Returns:
        (moves, winner, info)
        - moves: lista de dicts {'color': 'B'|'W', 'coords': str, 'pass': bool, 'comment': str?}
        - winner: 'B' | 'W' | None
        - info: metadatos del juego
    """
    if sgf is None:
        raise ImportError("El mÃ³dulo 'sgf' no estÃ¡ disponible. Instala sgfmill u otro parser SGF compatible.")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    collection = sgf.parse(content)
    game = collection[0]

    info: Dict = {}
    props = game.root.properties
    if 'PB' in props:
        info['black_player'] = props['PB'][0]
    if 'PW' in props:
        info['white_player'] = props['PW'][0]
    if 'BR' in props:
        info['black_rank'] = props['BR'][0]
    if 'WR' in props:
        info['white_rank'] = props['WR'][0]
    if 'DT' in props:
        info['date'] = props['DT'][0]
    if 'KM' in props:
        info['komi'] = float(props['KM'][0])
    if 'SZ' in props:
        try:
            info['board_size'] = int(props['SZ'][0])
        except Exception:
            pass

    winner: Optional[str] = None
    if 'RE' in props:
        result = props['RE'][0]
        if isinstance(result, str):
            # Guardar texto de resultado para consumo aguas arriba
            info['result'] = result
            if result.startswith('B'):
                winner = 'B'
            elif result.startswith('W'):
                winner = 'W'

    moves: List[Dict] = []
    for node in game.rest:
        entry: Dict = {}
        if 'B' in node.properties:
            entry['color'] = 'B'
            entry['coords'] = node.properties['B'][0]
        elif 'W' in node.properties:
            entry['color'] = 'W'
            entry['coords'] = node.properties['W'][0]
        else:
            continue
        entry['pass'] = (len(entry['coords']) == 0)
        if 'C' in node.properties:
            entry['comment'] = node.properties['C'][0]
        moves.append(entry)

    return moves, winner, info


def replay_game(moves: List[Dict], size: int = 19) -> 'GoBoard':
    """Reproduce movimientos en un GoBoard. Ignora 'pass'.

    Args:
        moves: lista de movimientos del extractor SGF
        size: tamaÃ±o del tablero (5, 9, 13, 19)
    """
    from .go_game_engine import GoBoard  # import diferido para evitar ciclos
    board = GoBoard(size=size)
    # Delegar en la lgica del motor para evitar duplicacin
    board.replay_moves(moves)
    return board

def board_to_numpy(board: 'GoBoard') -> np.ndarray:
    """Devuelve el tablero interno como np.ndarray[str] (delegado)."""
    return _board_to_numpy(board)

