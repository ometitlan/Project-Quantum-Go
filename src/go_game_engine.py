"""
go_game_engine.py
=================
Motor completo del juego de Go con validaciÃƒÂ³n de reglas,
detecciÃƒÂ³n de capturas, historial y cÃƒÂ¡lculo de territorio.

Uso:
    from go_game_engine import GoBoard, SGFParser
    
    # Crear tablero
    board = GoBoard(size=19)
    
    # Colocar piedra
    success, msg = board.place_stone('B', (3, 3))
    
    # Cargar partida desde SGF
    parser = SGFParser()
    moves, info = parser.parse_file("game.sgf")
    board.replay_moves(moves)
"""

import numpy as np
from typing import Tuple, List, Dict, Set, Optional, Union
from dataclasses import dataclass
from copy import deepcopy
from .sgf_utils import extract_moves_and_info, _sgf_to_coords


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class MoveInfo:
    """InformaciÃƒÂ³n sobre un movimiento"""
    color: str  # 'B' o 'W'
    position: Tuple[int, int]  # (fila, columna)
    captured_stones: int = 0
    is_legal: bool = True
    reason: str = ""  # Si es ilegal, por quÃƒÂ©
    comment: str = ""  # Comentario del SGF


@dataclass
class GameInfo:
    """Metadatos de la partida"""
    black_player: str = ""
    white_player: str = ""
    black_rank: str = ""
    white_rank: str = ""
    date: str = ""
    komi: float = 6.5
    board_size: int = 19
    result: str = ""
    winner: Optional[str] = None


# ============================================================================
# CLASE PRINCIPAL: GoBoard
# ============================================================================

class GoBoard:
    """
    Motor del juego de Go con reglas completas.
    
    CaracterÃƒÂ­sticas:
    - DetecciÃƒÂ³n de capturas automÃƒÂ¡tica
    - ValidaciÃƒÂ³n de movimientos (incluye Ko)
    - Historial de estados
    - CÃƒÂ¡lculo de libertades y grupos
    - ExportaciÃƒÂ³n a diferentes formatos
    
    Atributos:
        size (int): TamaÃƒÂ±o del tablero (19, 13, 9, etc.)
        board (np.ndarray): Matriz del tablero ('.' vacÃƒÂ­o, 'B' negro, 'W' blanco)
        captures (dict): Piedras capturadas por cada jugador
        move_history (list): Historial de movimientos
        board_history (list): Historial de estados del tablero (para Ko)
    """
    
    def __init__(self, size: int = 19, initial_matrix: Optional[Union[np.ndarray, List[List[str]]]] = None):
        """
        # Delegate to sgf_utils to avoid duplicated parsing logic
        moves, winner, info = extract_moves_and_info(file_path)
        gi = GameInfo(
            black_player=info.get('PB', info.get('black_player', '')) if isinstance(info.get('PB', ''), str) else info.get('black_player', ''),
            white_player=info.get('PW', info.get('white_player', '')) if isinstance(info.get('PW', ''), str) else info.get('white_player', ''),
            black_rank=info.get('BR', info.get('black_rank', '')) if isinstance(info.get('BR', ''), str) else info.get('black_rank', ''),
            white_rank=info.get('WR', info.get('white_rank', '')) if isinstance(info.get('WR', ''), str) else info.get('white_rank', ''),
            date=info.get('date', ''),
            komi=float(info.get('komi', 6.5)) if info.get('komi') is not None else 6.5,
            board_size=int(info.get('board_size', 19)),
            result=info.get('result', '') if 'result' in info else ''
        )
        gi.winner = winner
        return moves, gi
        Inicializa un tablero de Go vacÃƒÂ­o.
        
        Args:
            size: TamaÃƒÂ±o del tablero (tÃƒÂ­picamente 19, 13 o 9)
        """
        if size < 2:
            raise ValueError("El tamaÃƒÂ±o del tablero debe ser al menos 2")
            
        self.size = size
        self.board = np.full((size, size), '.', dtype=str)
        self.captures = {'B': 0, 'W': 0}
        self.move_history: List[MoveInfo] = []
        self.board_history: List[np.ndarray] = []
        
        # Cach para optimizacin
        self._group_cache: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
        self._liberty_cache: Dict[Tuple[int, int], int] = {}

        # Inicializacin opcional desde matriz
        if initial_matrix is not None:
            arr = np.array(initial_matrix, dtype=str)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ValueError("initial_matrix debe ser cuadrada")
            if arr.shape[0] != size:
                raise ValueError(f"initial_matrix de tamaÃƒÂ±o {arr.shape[0]} no coincide con size={size}")
            # Validar smbolos
            if not np.isin(arr, ['.', 'B', 'W']).all():
                raise ValueError("initial_matrix debe contener solo '.', 'B', 'W'")
            self.board[:, :] = arr
    
    # ------------------------------------------------------------------------
    # MTODOS PRINCIPALES
    # ------------------------------------------------------------------------
    
    def place_stone(self, color: str, position: Tuple[int, int], 
                   validate: bool = True) -> Tuple[bool, str]:
        """
        Coloca una piedra en el tablero con validaciÃƒÂ³n completa.
        
        Args:
            color: 'B' (negro) o 'W' (blanco)
            position: Tupla (fila, columna) en coordenadas 0-indexed
            validate: Si True, valida que el movimiento sea legal
        
        Returns:
            Tupla (ÃƒÂ©xito, mensaje)
            - ÃƒÂ©xito (bool): True si se colocÃƒÂ³ la piedra
            - mensaje (str): DescripciÃƒÂ³n del resultado
        
        Ejemplo:
            >>> board = GoBoard(19)
            >>> success, msg = board.place_stone('B', (3, 3))
            >>> print(success)  # True
            >>> print(msg)      # "Piedra colocada. 0 capturas."
        """
        if color not in ['B', 'W']:
            return False, "Color invÃƒÂ¡lido. Usa 'B' o 'W'"
        
        row, col = position
        
        # Validar lmites
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False, f"PosiciÃƒÂ³n fuera del tablero: {position}"
        
        # Validar que est vaca
        if self.board[row, col] != '.':
            return False, f"La posiciÃƒÂ³n {position} ya estÃƒÂ¡ ocupada"
        
        # Guardar estado anterior (para Ko y deshacer)
        previous_board = self.board.copy()
        
        # Colocar piedra temporalmente
        self.board[row, col] = color
        
        # Realizar capturas
        captured = self._perform_captures(row, col, color)
        
        # Validar movimiento si est activado
        if validate:
            is_legal, reason = self._validate_move(row, col, color, previous_board)
            if not is_legal:
                # Revertir movimiento
                self.board = previous_board
                return False, reason
        
        # Movimiento vlido: actualizar estado
        self.captures[color] += captured
        self.board_history.append(previous_board)
        
        move_info = MoveInfo(
            color=color,
            position=position,
            captured_stones=captured,
            is_legal=True
        )
        self.move_history.append(move_info)
        
        # Invalidar cach
        self._invalidate_cache()
        
        msg = f"Piedra colocada. {captured} captura(s)."
        return True, msg
    
    def _perform_captures(self, row: int, col: int, color: str) -> int:
        """
        Ejecuta las capturas resultantes de colocar una piedra.
        
        Args:
            row, col: PosiciÃƒÂ³n de la piedra reciÃƒÂ©n colocada
            color: Color de la piedra
        
        Returns:
            NÃƒÂºmero total de piedras capturadas
        """
        opponent = 'W' if color == 'B' else 'B'
        total_captured = 0
        
        # Revisar cada vecino
        for nx, ny in self._get_neighbors(row, col):
            if self.board[nx, ny] == opponent:
                # Si el grupo enemigo no tiene libertades, capturarlo
                if not self._has_liberties(nx, ny):
                    captured = self._remove_group(nx, ny)
                    total_captured += captured
        
        return total_captured
    
    def _validate_move(self, row: int, col: int, color: str, 
                      previous_board: np.ndarray) -> Tuple[bool, str]:
        """
        Valida que un movimiento sea legal segÃƒÂºn las reglas de Go.
        
        Reglas verificadas:
        1. No suicidio (colocar sin libertades)
        2. No Ko (repetir posiciÃƒÂ³n inmediata anterior)
        
        Args:
            row, col: PosiciÃƒÂ³n del movimiento
            color: Color de la piedra
            previous_board: Estado del tablero antes del movimiento
        
        Returns:
            Tupla (es_legal, razÃƒÂ³n)
        """
        # Regla 1: Suicidio
        # Solo es ilegal si no captura nada
        if not self._has_liberties(row, col):
            return False, "Movimiento suicida (sin libertades)"
        
        # Regla 2: Ko
        # No puede repetir la posicin inmediata anterior
        if len(self.board_history) > 0:
            last_board = self.board_history[-1]
            if np.array_equal(self.board, last_board):
                return False, "ViolaciÃƒÂ³n de Ko (repeticiÃƒÂ³n de posiciÃƒÂ³n)"
        
        return True, "Movimiento legal"
    
    # ------------------------------------------------------------------------
    # DETECCIN DE GRUPOS Y LIBERTADES
    # ------------------------------------------------------------------------
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Obtiene las 4 posiciones adyacentes vÃƒÂ¡lidas (arriba, abajo, izq, der).
        
        Args:
            row, col: PosiciÃƒÂ³n central
        
        Returns:
            Lista de tuplas (fila, columna) de vecinos vÃƒÂ¡lidos
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                neighbors.append((nr, nc))
        return neighbors
    
    def _get_group(self, row: int, col: int) -> Set[Tuple[int, int]]:
        """
        Encuentra todas las piedras conectadas del mismo color.
        
        Usa bÃƒÂºsqueda en profundidad (DFS) para encontrar el grupo completo.
        
        Args:
            row, col: PosiciÃƒÂ³n de una piedra en el grupo
        
        Returns:
            Set de tuplas (fila, columna) de todas las piedras del grupo
        
        Ejemplo:
            Tablero:  . B B .
                      . B . .
                      . . . .
            
            _get_group(0, 1) retorna: {(0,1), (0,2), (1,1)}
        """
        # Verificar cach
        if (row, col) in self._group_cache:
            return self._group_cache[(row, col)]
        
        color = self.board[row, col]
        if color == '.':
            return set()
        
        group = set()
        stack = [(row, col)]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            
            if self.board[r, c] == color:
                group.add((r, c))
                stack.extend(self._get_neighbors(r, c))
        
        # Guardar en cach
        for pos in group:
            self._group_cache[pos] = group
        
        return group
    
    def _has_liberties(self, row: int, col: int) -> bool:
        """
        Verifica si una piedra (o su grupo) tiene al menos una libertad.
        
        Una libertad es una posiciÃƒÂ³n vacÃƒÂ­a adyacente a cualquier piedra del grupo.
        
        Args:
            row, col: PosiciÃƒÂ³n de la piedra a verificar
        
        Returns:
            True si el grupo tiene al menos una libertad
        
        Complejidad: O(n) donde n es el tamaÃƒÂ±o del grupo
        """
        group = self._get_group(row, col)
        
        for gr, gc in group:
            for nr, nc in self._get_neighbors(gr, gc):
                if self.board[nr, nc] == '.':
                    return True
        
        return False
    
    def count_liberties(self, row: int, col: int) -> int:
        """
        Cuenta el nÃƒÂºmero exacto de libertades de un grupo.
        
        Args:
            row, col: PosiciÃƒÂ³n de una piedra en el grupo
        
        Returns:
            NÃƒÂºmero de libertades del grupo
        """
        group = self._get_group(row, col)
        liberties = set()
        
        for gr, gc in group:
            for nr, nc in self._get_neighbors(gr, gc):
                if self.board[nr, nc] == '.':
                    liberties.add((nr, nc))
        
        return len(liberties)
    
    def _remove_group(self, row: int, col: int) -> int:
        """
        Elimina un grupo completo del tablero.
        
        Args:
            row, col: PosiciÃƒÂ³n de cualquier piedra del grupo
        
        Returns:
            NÃƒÂºmero de piedras eliminadas
        """
        group = self._get_group(row, col)
        
        for gr, gc in group:
            self.board[gr, gc] = '.'
        
        return len(group)
    
    # ------------------------------------------------------------------------
    # HISTORIAL Y NAVEGACIN
    # ------------------------------------------------------------------------
    
    def undo_move(self) -> bool:
        """
        Deshace el ÃƒÂºltimo movimiento.
        
        Returns:
            True si se pudo deshacer
        """
        if len(self.board_history) == 0:
            return False
        
        self.board = self.board_history.pop()
        last_move = self.move_history.pop()
        
        # Revertir capturas
        self.captures[last_move.color] -= last_move.captured_stones
        
        self._invalidate_cache()
        return True
    
    def replay_moves(self, moves: List[Dict]) -> List[MoveInfo]:
        """
        Reproduce una lista de movimientos en el tablero.
        
        Args:
            moves: Lista de dicts con 'color' y 'coords' (formato SGF)
        
        Returns:
            Lista de MoveInfo con resultados de cada movimiento
        """
        results = []
        
        for move in moves:
            color = move['color']
            coords = move.get('coords', '')
            
            if not coords:  # Pass
                continue
            
            rc = _sgf_to_coords(coords)
            if rc is None:
                continue
            position = rc
            success, msg = self.place_stone(color, position)
            
            if success:
                self.move_history[-1].comment = move.get('comment', '')
            
            results.append(self.move_history[-1] if success else 
                          MoveInfo(color, position, is_legal=False, reason=msg))
        
        return results
    
    def get_captures(self) -> Dict[str, int]:
        """
        Obtiene el nÃƒÂºmero de capturas de cada jugador.
        
        Returns:
            Dict con capturas de 'B' y 'W'
        """
        return self.captures.copy()
    
    # ------------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------------
    
    def _sgf_to_position(self, sgf_coords: str) -> Tuple[int, int]:
        """
        Convierte coordenadas SGF (ej: 'pd') a posiciÃƒÂ³n (fila, columna).
        
        Args:
            sgf_coords: String de 2 letras (a-s para tablero 19x19)
        
        Returns:
            Tupla (fila, columna) en coordenadas 0-indexed
        """
        if len(sgf_coords) != 2:
            raise ValueError(f"Coordenadas SGF invÃƒÂ¡lidas: {sgf_coords}")
        
        col = ord(sgf_coords[0]) - ord('a')
        row = ord(sgf_coords[1]) - ord('a')
        
        return (row, col)
    
    def _position_to_sgf(self, position: Tuple[int, int]) -> str:
        """Convierte (fila, columna) a formato SGF."""
        row, col = position
        return chr(col + ord('a')) + chr(row + ord('a'))
    
    def _invalidate_cache(self):
        """Limpia los cachÃƒÂ©s de grupos y libertades."""
        self._group_cache.clear()
        self._liberty_cache.clear()
    
    def to_numpy(self) -> np.ndarray:
        """Retorna el tablero como array NumPy."""
        return self.board.copy()
    
    def __str__(self) -> str:
        """RepresentaciÃƒÂ³n en texto del tablero."""
        lines = []
        for row in self.board:
            lines.append(' '.join(row))
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return f"GoBoard(size={self.size}, moves={len(self.move_history)})"


# ============================================================================
# PARSER DE ARCHIVOS SGF
# ============================================================================

class SGFParser:
    """
    Parser para archivos SGF (Smart Game Format).
    
    Extrae movimientos, metadatos y comentarios de partidas de Go.
    Implementacin delegada a sgf_utils para evitar duplicacin.
    """
    
    @staticmethod
    def parse_file(file_path: str) -> Tuple[List[Dict], GameInfo]:
        """Parsea un archivo SGF delegando en sgf_utils (sin dependencias duplicadas)."""
        moves, winner, info = extract_moves_and_info(str(file_path))
        gi = GameInfo(
            black_player=info.get('black_player', ''),
            white_player=info.get('white_player', ''),
            black_rank=info.get('black_rank', ''),
            white_rank=info.get('white_rank', ''),
            date=info.get('date', ''),
            komi=float(info.get('komi', 6.5)) if info.get('komi') is not None else 6.5,
            board_size=int(info.get('board_size', 19)),
            result=info.get('result', '') if 'result' in info else ''
        )
        gi.winner = winner
        return moves, gi
