"""
go_isings_models.py
===================
Modelos de Ising (cuÃ¡nticos y clÃ¡sicos) para el juego de Go.

CaracterÃ­sticas:
- Hamiltoniano cuÃ¡ntico con PennyLane
- Hamiltoniano clÃ¡sico equivalente
- Kernels Manhattan 1 y 2
- IntegraciÃ³n con GoBoard
- GeneraciÃ³n de mapas de energÃ­a
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import pennylane as qml
from scipy.optimize import minimize
from scipy.signal import convolve2d

# ============================================================================
# CLASE 1: CONFIGURACIN FSICA
# ============================================================================

class IsingGoConfig:
    """
    ConfiguraciÃ³n fÃ­sica del modelo de Ising para Go.
    
    Define parÃ¡metros fundamentales:
    - Mapeo de piedras a spins/qubits
    - TopologÃ­a de kernels (Manhattan distance)
    - Coeficientes de interacciÃ³n
    """
    
    # Mapeo estndar: piedras  spins clsicos
    STONE_TO_SPIN = {'B': -1, 'W': +1, '.': 0}
    SPIN_TO_STONE = {-1: 'B', +1: 'W', 0: '.'}
    
    # Mapeo cuntico: piedras  estados de qubit
    # 'B'  |1 (eigenestado de Z con valor -1)
    # 'W'  |0 (eigenestado de Z con valor +1)
    # '.'  |+ (superposicin)
    
    # Coeficientes de interaccion por distancia Manhattan: w_R = 1/R.
    # El dict permite overrides puntuales; para cualquier otra distancia se
    # usa la formula general 1/R (ver interaction_coeff), sin tope de radio.
    INTERACTION_COEFFS = {
        1: 1.0,     # Vecinos inmediatos: peso completo
        2: 0.5,
        3: 1.0 / 3.0,
        4: 0.25,
    }

    @classmethod
    def interaction_coeff(cls, dist: int) -> float:
        """Peso w_R = 1/R para cualquier distancia Manhattan >= 1."""
        if dist < 1:
            return 0.0
        return cls.INTERACTION_COEFFS.get(dist, 1.0 / float(dist))
    
    @staticmethod
    def get_kernel_positions(manhattan_distance: int = 1) -> Dict[int, Tuple[int, int]]:
        """
        Genera posiciones del kernel en coordenadas relativas (dx, dy).
        
        Args:
            manhattan_distance: Radio del kernel (1, 2, 3, o 4)
            
        Returns:
            dict: {qubit_index: (dx, dy)}
                  qubit_index=0 siempre es el centro (0, 0)
        
        Ejemplo:
            manhattan_distance=1 â†’ 5 posiciones (cruz)
            manhattan_distance=2 â†’ 13 posiciones (cruz extendida + diagonales)
        """
        positions = {0: (0, 0)}  # Centro
        idx = 1
        
        # Iterar por capas de distancia
        for d in range(1, manhattan_distance + 1):
            for dx in range(-d, d + 1):
                for dy in range(-d, d + 1):
                    # Condicin de Manhattan: |dx| + |dy| = d
                    if abs(dx) + abs(dy) == d:
                        positions[idx] = (dx, dy)
                        idx += 1
        
        return positions
    
    @staticmethod
    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calcula distancia Manhattan entre dos posiciones."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @classmethod
    def get_kernel_info(cls, manhattan_distance: int) -> Dict:
        """
        Retorna informaciÃ³n completa del kernel.
        
        Returns:
            dict con: positions, n_qubits, coefficients
        """
        positions = cls.get_kernel_positions(manhattan_distance)
        n_qubits = len(positions)
        
        # Calcular coeficientes para cada qubit
        coefficients = {}
        for idx, pos in positions.items():
            if idx == 0:
                coefficients[idx] = 0.0  # Centro no tiene coeficiente (no interactÃºa consigo mismo)
            else:
                dist = cls.manhattan_distance((0, 0), pos)
                coefficients[idx] = cls.interaction_coeff(dist)
        
        return {
            'positions': positions,
            'n_qubits': n_qubits,
            'coefficients': coefficients,
            'manhattan_distance': manhattan_distance
        }


# ============================================================================
# MAPAS POSICIONALES SPIN-1
# ============================================================================

MAP_METHOD_ALIASES = {
    'cubic': 'cubic',
    'cubico': 'cubic',
    'cúbico': 'cubic',
    'spin-cubic': 'cubic',
    'spin_cubic': 'cubic',
    'classical': 'cubic',
    'quadratic': 'quadratic',
    'cuadratico': 'quadratic',
    'cuadrático': 'quadratic',
    'quad': 'quadratic',
    'atomic': 'quadratic',
    'atomic-go': 'quadratic',
    'atomic_go': 'quadratic',
    'quantum': 'quantum',
    'legacy-quantum': 'quantum',
    'legacy_quantum': 'quantum',
    'legacy-classical': 'legacy-classical',
    'legacy_classical': 'legacy-classical',
}

MAP_METHOD_LABELS = {
    'cubic': 'Cúbico',
    'quadratic': 'Cuadrático',
    'quantum': 'Cuántico heredado',
    'legacy-classical': 'Clásico heredado',
}


def normalize_map_method(method: str) -> str:
    """Return the canonical map method name used by the current API."""
    key = str(method).strip().lower()
    try:
        return MAP_METHOD_ALIASES[key]
    except KeyError as exc:
        valid = ', '.join(sorted(MAP_METHOD_ALIASES))
        raise ValueError(f"Metodo de mapa no soportado: {method!r}. Opciones: {valid}") from exc


# Sufijos para lectura de jugada hipotetica en claves compuestas,
# p. ej. 'cubic-hipB' o 'quadratic-hip-blanco'.
HYP_METHOD_SUFFIXES = {
    'hipb': 'B', 'hip-b': 'B', 'hipnegro': 'B', 'hip-negro': 'B',
    'hipw': 'W', 'hip-w': 'W', 'hipblanco': 'W', 'hip-blanco': 'W',
}


def parse_map_method(method: str):
    """Devuelve (metodo canonico, kwargs) a partir de una clave posiblemente compuesta.

    Sufijos disponibles:
    - '-hipB' / '-hipW': lectura de jugada hipotetica -> hypothetical_color
    - '-raw' (alias '-suma'): alias historico SIN efecto — la suma directa por
      capa es ahora el unico comportamiento de los mapas.

    Ejemplos: 'cubic' -> ('cubic', {});
              'quadratic-hipB' -> ('quadratic', {'hypothetical_color': 'B'}).
    """
    raw = str(method).strip().lower().replace('_', '-')
    kwargs = {}
    changed = True
    while changed:
        changed = False
        for flag in ('-raw', '-suma'):
            if raw.endswith(flag):
                # alias historico: se acepta y se ignora
                raw = raw[: -len(flag)]
                changed = True
        for suffix, color in HYP_METHOD_SUFFIXES.items():
            marker = f'-{suffix}'
            if raw.endswith(marker):
                kwargs['hypothetical_color'] = color
                raw = raw[: -len(marker)]
                changed = True
                break
    return normalize_map_method(raw), kwargs


def split_map_method(method: str):
    """Compatibilidad: (metodo canonico, color hipotetico o None)."""
    canonical, kwargs = parse_map_method(method)
    return canonical, kwargs.get('hypothetical_color')


def map_method_label(method: str) -> str:
    canonical, kwargs = parse_map_method(method)
    label = MAP_METHOD_LABELS.get(canonical, str(canonical).title())
    hyp_color = kwargs.get('hypothetical_color')
    if hyp_color is not None:
        label = f"{label} | hip. {'Negro' if hyp_color == 'B' else 'Blanco'}"
    return label


def spin_from_stone(stone) -> int:
    """Map board symbols to spin-1 values: B=-1, W=+1, empty=0."""
    value = str(stone)
    if value in IsingGoConfig.STONE_TO_SPIN:
        return IsingGoConfig.STONE_TO_SPIN[value]
    if value in {'0', '.', 'None'}:
        return 0
    raise ValueError(f"Valor de tablero no soportado: {stone!r}")


def board_to_spins(board: np.ndarray) -> np.ndarray:
    """Tablero de simbolos -> matriz de spins float (B=-1, W=+1, vacio=0)."""
    b = np.asarray(board, dtype=str)
    s = np.zeros(b.shape, dtype=float)
    s[b == 'W'] = 1.0
    s[b == 'B'] = -1.0
    return s


def manhattan_ring_kernel(dist: int) -> np.ndarray:
    """Anillo Manhattan de radio `dist` como kernel (2d+1)x(2d+1) de 0/1."""
    d = int(dist)
    size = 2 * d + 1
    dx = np.abs(np.arange(size) - d)
    K = (dx[:, None] + dx[None, :] == d).astype(float)
    return K


def spin_from_color(color) -> int:
    """Map a color token to a spin value for hypothetical moves."""
    if color is None:
        raise ValueError("color no puede ser None")
    if isinstance(color, (int, float)):
        spin = int(color)
        if spin in (-1, 1):
            return spin
    token = str(color).strip().lower()
    if token in {'b', 'black', 'negro', '-1'}:
        return -1
    if token in {'w', 'white', 'blanco', '+1', '1'}:
        return 1
    raise ValueError(f"Color no soportado: {color!r}")


def cubic_interaction(s0: float, s1: float) -> float:
    """Mapa cúbico: s0 + 2*s1 - s0*s1^2 - s0^2*s1."""
    return s0 + 2.0 * s1 - s0 * (s1 ** 2) - (s0 ** 2) * s1


def quadratic_interaction(s0: float, s1: float, J: float = 1.0) -> float:
    """Mapa cuadrático tipo Ising/Atomic-Go: -J*s0*s1."""
    return -float(J) * s0 * s1


class PositionalMapModel:
    """Modelo local spin-1 para mapas posicionales del tablero de Go.

    Implementa la linea actual del README:
    - capas Manhattan R=1..manhattan_distance;
    - peso por capa w_R = 1/R;
    - suma directa sobre los vecinos validos de cada capa (sin promediar):
      los puntos con menos vecinos reciben menos senal por construccion —
      una esquina vale menos que una orilla y esta menos que el centro.
    """

    def __init__(
        self,
        method: str = 'cubic',
        *,
        manhattan_distance: int = 1,
        J: float = 1.0,
        hypothetical_color=None,
    ):
        canonical = normalize_map_method(method)
        if canonical not in {'cubic', 'quadratic'}:
            raise ValueError(
                f"PositionalMapModel solo acepta 'cubic' o 'quadratic', no {method!r}"
            )
        self.method = canonical
        self.model_type = canonical
        self.manhattan_distance = int(manhattan_distance)
        self.J = float(J)
        self.hypothetical_spin = (
            spin_from_color(hypothetical_color) if hypothetical_color is not None else None
        )
        self.layers = self._build_layers(self.manhattan_distance)

    @staticmethod
    def _build_layers(manhattan_distance: int):
        layers = {}
        for dist in range(1, int(manhattan_distance) + 1):
            offsets = []
            for dx in range(-dist, dist + 1):
                for dy in range(-dist, dist + 1):
                    if abs(dx) + abs(dy) == dist:
                        offsets.append((dx, dy))
            layers[dist] = offsets
        return layers

    @staticmethod
    def distance_weight(dist: int) -> float:
        return 1.0 / float(dist)

    def local_interaction(self, center_spin: float, neighbor_spin: float) -> float:
        if self.method == 'cubic':
            return cubic_interaction(center_spin, neighbor_spin)
        return quadratic_interaction(center_spin, neighbor_spin, J=self.J)

    def _center_spin(self, board: np.ndarray, x: int, y: int) -> int:
        if self.hypothetical_spin is not None and str(board[x, y]) == '.':
            return self.hypothetical_spin
        return spin_from_stone(board[x, y])

    def compute_energy(self, board: np.ndarray, x: int, y: int) -> float:
        center_spin = self._center_spin(board, x, y)
        total = 0.0

        for dist, offsets in self.layers.items():
            layer_sum = 0.0
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                    neighbor_spin = spin_from_stone(board[nx, ny])
                    layer_sum += self.local_interaction(center_spin, neighbor_spin)
            total += self.distance_weight(dist) * layer_sum

        return float(total)

    def compute_map(self, board: np.ndarray) -> np.ndarray:
        """Mapa completo vectorizado con convoluciones de anillos Manhattan.

        Equivalente exacto a llamar compute_energy en cada punto, pero en
        O(capas x convolucion) para todo el tablero, lo que hace viables
        radios grandes (R = 9 o mas) y experimentos por lotes.

        Descomposicion usada (por capa R, con K_R el anillo y * convolucion):
        - cubico:    sum_j E = c_R*s~ + 2(K_R*s) - s~(K_R*s^2) - s~^2(K_R*s)
        - cuadratico: sum_j E = -J * s~ * (K_R*s)
        donde s~ es el spin del centro (con lectura hipotetica si aplica) y
        c_R el numero de vecinos validos (convolucion de un tablero de unos),
        necesario porque el termino s~ del cubico aporta una vez por vecino.
        """
        s = board_to_spins(board)
        s2 = s * s
        if self.hypothetical_spin is not None:
            center = np.where(np.asarray(board, dtype=str) == '.',
                              float(self.hypothetical_spin), s)
        else:
            center = s

        total = np.zeros_like(s)
        for dist in self.layers:
            K = manhattan_ring_kernel(dist)
            conv_s = convolve2d(s, K, mode='same', boundary='fill', fillvalue=0.0)
            if self.method == 'cubic':
                count = convolve2d(np.ones_like(s), K, mode='same',
                                   boundary='fill', fillvalue=0.0)
                conv_s2 = convolve2d(s2, K, mode='same', boundary='fill', fillvalue=0.0)
                layer = (count * center + 2.0 * conv_s
                         - center * conv_s2 - (center ** 2) * conv_s)
            else:  # quadratic
                layer = -self.J * center * conv_s
            total += self.distance_weight(dist) * layer

        return total


class CubicSpinMapModel(PositionalMapModel):
    """Mapa cúbico spin-1 para influencia/ventaja local."""

    def __init__(self, **kwargs):
        super().__init__('cubic', **kwargs)


class QuadraticSpinMapModel(PositionalMapModel):
    """Mapa cuadrático -J*s0*s1 para conexión/corte."""

    def __init__(self, **kwargs):
        super().__init__('quadratic', **kwargs)


def create_positional_map_model(method: str, *, manhattan_distance: int = 1, **kwargs):
    """Factory for the current named positional maps.

    Acepta claves compuestas, p. ej. 'cubic-hipB' (lectura hipotetica);
    ver parse_map_method.
    """
    canonical, method_kwargs = parse_map_method(method)
    for key, value in method_kwargs.items():
        kwargs.setdefault(key, value)
    if canonical == 'cubic':
        return CubicSpinMapModel(manhattan_distance=manhattan_distance, **kwargs)
    if canonical == 'quadratic':
        return QuadraticSpinMapModel(manhattan_distance=manhattan_distance, **kwargs)
    raise ValueError(f"El metodo {method!r} no es un mapa posicional directo")


# ============================================================================
# CLASE 2: MODELO CUNTICO
# ============================================================================

class QuantumIsingModel:
    """
    Modelo de Ising cuÃ¡ntico para Go usando PennyLane.
    
    Hamiltoniano:
        H = Î£_i c_i [IâŠ—Z_i + ZâŠ—X_i + XâŠ—Z_i]
    
    donde:
        - i itera sobre vecinos del centro
        - c_i es el coeficiente segÃºn distancia Manhattan
    """
    
    def __init__(self, manhattan_distance: int = 1, config: Optional[IsingGoConfig] = None, hamiltonian: Optional[qml.Hamiltonian] = None, allow_large_kernel: bool = False):
        """
        Inicializa el modelo cuÃ¡ntico.

        Args:
            manhattan_distance: Radio del kernel (1 o 2; el kernel usa
                1 + 2R(R+1) qubits, asi que R=3 ya son 25 qubits)
            config: ConfiguraciÃ³n personalizada (opcional)
            allow_large_kernel: permite kernels de mas de 13 qubits bajo tu
                propio riesgo (el statevector crece como 2^n)
        """
        self.config = config or IsingGoConfig()
        self.manhattan_distance = manhattan_distance

        # Obtener info del kernel
        kernel_info = self.config.get_kernel_info(manhattan_distance)
        self.positions = kernel_info['positions']
        self.n_qubits = kernel_info['n_qubits']
        self.coefficients = kernel_info['coefficients']

        # Limite fisico de la simulacion exacta: el kernel de radio R usa
        # 1 + 2R(R+1) qubits (R=1: 5, R=2: 13, R=3: 25, R=4: 41). Mas alla
        # de 13 el statevector (2^n amplitudes) deja de ser practico.
        if self.n_qubits > 13 and not allow_large_kernel:
            raise ValueError(
                f"Kernel Manhattan-{manhattan_distance} = {self.n_qubits} qubits: "
                f"la simulacion exacta requiere 2^{self.n_qubits} amplitudes. "
                "El backend cuantico esta acotado a R<=2 (13 qubits); para "
                "radios grandes usa los mapas clasicos (PositionalMapModel, "
                "que generaliza a cualquier R), o pasa allow_large_kernel=True "
                "bajo tu propio riesgo."
            )
        
        # Crear dispositivo cuntico
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        # Construir Hamiltoniano (o usar uno provisto)
        self.hamiltonian = hamiltonian if hamiltonian is not None else self._build_hamiltonian()
        
        # Crear circuito cuntico
        self.circuit = self._create_circuit()
    
    def _initialize_qubit(self, wire: int, stone_value: str):
        """
        Inicializa un qubit segÃºn el valor de la piedra.
        
        Args:
            wire: Ãndice del qubit
            stone_value: 'B', 'W', o '.'
        """
        if stone_value == 'B':
            qml.PauliX(wires=wire)  # |1âŸ©
        elif stone_value == 'W':
            pass  # |0âŸ© (estado inicial por defecto)
        else:  # VacÃ­o '.'
            qml.Hadamard(wires=wire)  # |+âŸ© = (|0âŸ© + |1âŸ©)/âˆš2
    
    def _initialize_kernel_qubits(self, board: np.ndarray, center_x: int, center_y: int):
        """
        Inicializa todos los qubits del kernel a partir del tablero.
        
        Args:
            board: Matriz del tablero (dtype=str, valores 'B'/'W'/'.')
            center_x, center_y: Coordenadas del centro
        """
        # Qubit 0: centro
        self._initialize_qubit(0, board[center_x, center_y])
        
        # Qubits vecinos
        for qubit_idx, (dx, dy) in self.positions.items():
            if qubit_idx == 0:
                continue
            
            nx, ny = center_x + dx, center_y + dy
            
            # Verificar lmites del tablero
            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                self._initialize_qubit(qubit_idx, board[nx, ny])
            else:
                # Fuera del tablero  tratamos como vaco
                qml.Hadamard(wires=qubit_idx)
    
    def _build_hamiltonian(self) -> qml.Hamiltonian:
        """
        Construye el Hamiltoniano cuÃ¡ntico.
        
        Returns:
            qml.Hamiltonian con operadores IâŠ—Z, ZâŠ—X, XâŠ—Z
        """
        coeffs = []
        observables = []
        
        for qubit_idx in range(1, self.n_qubits):
            coeff = self.coefficients[qubit_idx]
            
            if coeff == 0.0:
                continue
            
            # Operador 1: IZ (estado intrnseco del vecino)
            coeffs.append(coeff)
            observables.append(qml.Identity(0) @ qml.PauliZ(qubit_idx))
            
            # Operador 2: ZX (centro influye sobre vecino)
            coeffs.append(coeff)
            observables.append(qml.PauliZ(0) @ qml.PauliX(qubit_idx))
            
            # Operador 3: XZ (vecino influye sobre centro)
            coeffs.append(coeff)
            observables.append(qml.PauliX(0) @ qml.PauliZ(qubit_idx))
        
        return qml.Hamiltonian(coeffs, observables)
    
    def _create_circuit(self):
        """Crea el QNode (circuito cuÃ¡ntico compilado)."""
        @qml.qnode(self.dev)
        def circuit(board, center_x, center_y):
            self._initialize_kernel_qubits(board, center_x, center_y)
            return qml.expval(self.hamiltonian)
        
        return circuit
    
    def compute_energy(self, board: np.ndarray, x: int, y: int) -> float:
        """
        Calcula la energÃ­a cuÃ¡ntica en una posiciÃ³n.
        
        Args:
            board: Tablero de Go (array 2D con 'B'/'W'/'.')
            x, y: PosiciÃ³n a evaluar
            
        Returns:
            EnergÃ­a (float)
        """
        return float(self.circuit(board, x, y))
    
    def evolve_kernel(self, board: np.ndarray, x: int, y: int, t: float, *, steps: int = 2, return_state: bool = False) -> Dict:
        """
        Evoluciona el kernel local bajo e^{-iHt} y devuelve medidas base.

        Args:
            board: Tablero con 'B'/'W'/'.'
            x, y: Centro del kernel
            t: Tiempo de evolución
            steps: Pasos de Trotter para ApproxTimeEvolution
            return_state: Si True, incluye amplitudes complejas del estado

        Returns:
            dict con probs, expZ, expX, energy, t y position; opcionalmente state.
        """
        H = self.hamiltonian
        n_qubits = self.n_qubits
        dev = self.dev

        @qml.qnode(dev)
        def _evolution(board_in, cx, cy, time_param):
            self._initialize_kernel_qubits(board_in, cx, cy)
            qml.ApproxTimeEvolution(H, time=time_param, n=steps)
            probs = qml.probs(wires=range(n_qubits))
            expz = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            expx = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            energy_val = qml.expval(H)
            return probs, expz, expx, energy_val

        probs, expz, expx, energy_val = _evolution(board, x, y, t)
        result = {
            't': float(t),
            'position': (int(x), int(y)),
            'probs': np.array(probs),
            'expZ': [float(v) for v in expz],
            'expX': [float(v) for v in expx],
            'energy': float(energy_val),
        }

        if return_state:
            @qml.qnode(dev)
            def _state_qnode(board_in, cx, cy, time_param):
                self._initialize_kernel_qubits(board_in, cx, cy)
                qml.ApproxTimeEvolution(H, time=time_param, n=steps)
                return qml.state()
            result['state'] = np.array(_state_qnode(board, x, y, t))

        return result

    def evolve_over_times(self, board: np.ndarray, x: int, y: int, times, *, steps: int = 2, return_state: bool = False):
        """
        Ejecuta evolve_kernel para una lista/array de tiempos.

        Args:
            board: Tablero con 'B'/'W'/'.'
            x, y: Centro del kernel
            times: Iterable de tiempos
            steps: Pasos de Trotter
            return_state: Si True, incluye estado por tiempo

        Returns:
            Lista de dicts (uno por tiempo) listos para DataFrame o visualización.
        """
        records = []
        for t in times:
            rec = self.evolve_kernel(board, x, y, float(t), steps=steps, return_state=return_state)
            records.append(rec)
        return records

    # ------------------------------------------------------------------
    # Medidas dinamicas de entrelazamiento (parte genuinamente cuantica)
    # ------------------------------------------------------------------

    def _apply_evolution(self, time_param, steps: int):
        """Aplica e^{-iHt} con TrotterProduct; fallback a ApproxTimeEvolution."""
        try:
            qml.TrotterProduct(self.hamiltonian, time=time_param, n=int(steps), order=2)
        except Exception:
            qml.ApproxTimeEvolution(self.hamiltonian, time_param, int(steps))

    def _get_state_qnode(self):
        """QNode cacheado que devuelve el estado del kernel evolucionado."""
        if getattr(self, '_state_qnode_cache', None) is None:
            @qml.qnode(self.dev)
            def _state(board_in, cx, cy, time_param, steps):
                self._initialize_kernel_qubits(board_in, cx, cy)
                if float(time_param) != 0.0:
                    self._apply_evolution(time_param, steps)
                return qml.state()
            self._state_qnode_cache = _state
        return self._state_qnode_cache

    def evolved_state(self, board: np.ndarray, x: int, y: int, t: float, *, steps: int = 4) -> np.ndarray:
        """Estado del kernel local tras evolucionar bajo e^{-iHt}."""
        return np.asarray(self._get_state_qnode()(board, x, y, float(t), int(steps)))

    def entanglement_measures(self, board: np.ndarray, x: int, y: int, t: float, *, steps: int = 4) -> Dict:
        """Medidas del kernel evolucionado que el modelo clasico NO puede reproducir.

        Returns:
            dict con:
            - entropy_center: entropia de von Neumann (bits) del qubit central
            - entropies: entropia por qubit
            - expZ: <Z_i> por qubit
            - connected_zz: {j: <Z_0 Z_j> - <Z_0><Z_j>} (correlacion conectada)
        """
        state = self.evolved_state(board, x, y, t, steps=steps)
        n = self.n_qubits
        probs = np.abs(state) ** 2
        z_signs = [_z_eigenvalues(n, w) for w in range(n)]
        expZ = [float((probs * z_signs[w]).sum()) for w in range(n)]
        connected = {}
        for j in range(1, n):
            zz = float((probs * z_signs[0] * z_signs[j]).sum())
            connected[j] = zz - expZ[0] * expZ[j]
        entropies = [
            vn_entropy_bits(reduced_density_matrix(state, w, n)) for w in range(n)
        ]
        return {
            't': float(t),
            'position': (int(x), int(y)),
            'entropy_center': entropies[0],
            'entropies': entropies,
            'expZ': expZ,
            'connected_zz': connected,
        }

    def entanglement_over_times(self, board: np.ndarray, x: int, y: int, times, *, steps: int = 4):
        """entanglement_measures para una lista/array de tiempos."""
        return [
            self.entanglement_measures(board, x, y, float(t), steps=steps)
            for t in times
        ]

    def get_hamiltonian_info(self) -> Dict:
        """Retorna informaciÃ³n del Hamiltoniano."""
        return {
            'n_terms': len(self.hamiltonian.coeffs),
            'coefficients': self.hamiltonian.coeffs,
            'observables': [str(obs) for obs in self.hamiltonian.ops],
            'hamiltonian_str': str(self.hamiltonian)
        }


# ----------------------------------------------------------------------------
# Helpers de estados: eigenvalores Z, matriz densidad reducida, entropia
# ----------------------------------------------------------------------------

def _z_eigenvalues(n_qubits: int, wire: int) -> np.ndarray:
    """Eigenvalor de Z en `wire` (+1/-1) para cada indice de la base computacional.

    Convencion PennyLane: el wire 0 es el bit mas significativo del indice.
    """
    idx = np.arange(2 ** n_qubits)
    bits = (idx >> (n_qubits - 1 - wire)) & 1
    return 1.0 - 2.0 * bits


def reduced_density_matrix(state: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
    """Matriz densidad reducida (2x2) de un qubit a partir del estado puro."""
    psi = np.asarray(state).reshape([2] * n_qubits)
    psi = np.moveaxis(psi, wire, 0).reshape(2, -1)
    return psi @ psi.conj().T


def vn_entropy_bits(rho: np.ndarray, eps: float = 1e-12) -> float:
    """Entropia de von Neumann en bits: S = -sum(p log2 p)."""
    evals = np.linalg.eigvalsh(rho).real
    evals = evals[evals > eps]
    return float(-(evals * np.log2(evals)).sum()) if evals.size else 0.0


# ============================================================================
# CLASE 3: MODELO CLSICO
# ============================================================================

class ClassicalIsingModel:
    """
    Modelo de Ising clÃ¡sico equivalente al cuÃ¡ntico.
    
    Hamiltoniano:
        H = sâ‚€ + 2sâ‚ - sâ‚€sâ‚Â² - sâ‚€Â²sâ‚
    
    donde s_i âˆˆ {-1, 0, +1}
    """
    
    def __init__(self, config: Optional[IsingGoConfig] = None, manhattan_distance: int = 1):
        """
        Args:
            config: ConfiguraciÃ³n (opcional)
        """
        self.config = config or IsingGoConfig()
        # Distancia de Manhattan por defecto del modelo (p. ej., 1 o 2)
        self.manhattan_distance = int(manhattan_distance)
        
        # Parmetros del Hamiltoniano (ya optimizados)
        # Estos fueron encontrados mediante optimizacin numrica
        self.params = {
            'h0': 1.0,   # Coeficiente de sâ‚€
            'h1': 2.0,   # Coeficiente de sâ‚
            'K': -1.0,   # Coeficiente de sâ‚€sâ‚Â²
            'L': -1.0    # Coeficiente de sâ‚€Â²sâ‚
        }
    
    @staticmethod
    def _two_qubit_hamiltonian(s0: float, s1: float, params: Dict) -> float:
        """
        Hamiltoniano clÃ¡sico de 2 spins.
        
        Args:
            s0, s1: Valores de spin (cada uno en {-1, 0, +1})
            params: Dict con 'h0', 'h1', 'K', 'L'
            
        Returns:
            EnergÃ­a
        """
        h0, h1, K, L = params['h0'], params['h1'], params['K'], params['L']
        return h0*s0 + h1*s1 + K*s0*(s1**2) + L*(s0**2)*s1
    
    def compute_energy_single_interaction(self, center_spin: float, neighbor_spin: float) -> float:
        """
        Calcula energÃ­a de interacciÃ³n centro-vecino.
        
        Args:
            center_spin: Spin del centro (âˆ’1, 0, +1)
            neighbor_spin: Spin del vecino (âˆ’1, 0, +1)
            
        Returns:
            EnergÃ­a de la interacciÃ³n
        """
        return self._two_qubit_hamiltonian(center_spin, neighbor_spin, self.params)
    
    def compute_energy(self, board: np.ndarray, x: int, y: int,
                      manhattan_distance: Optional[int] = None) -> float:
        """
        Calcula energÃ­a clÃ¡sica total en una posiciÃ³n.
        
        Args:
            board: Tablero de Go
            x, y: PosiciÃ³n central
            manhattan_distance: Radio del kernel (si None, usa el del modelo)
            
        Returns:
            EnergÃ­a total
        """
        effective_distance = int(manhattan_distance) if manhattan_distance is not None else self.manhattan_distance
        positions = self.config.get_kernel_positions(effective_distance)
        center_spin = self.config.STONE_TO_SPIN[board[x, y]]
        
        total_energy = 0.0
        
        for qubit_idx, (dx, dy) in positions.items():
            if qubit_idx == 0:
                continue
            
            nx, ny = x + dx, y + dy
            
            # Verificar lmites
            if 0 <= nx < board.shape[0] and 0 <= ny < board.shape[1]:
                neighbor_spin = self.config.STONE_TO_SPIN[board[nx, ny]]
            else:
                neighbor_spin = 0.0  # Fuera del tablero = vacÃ­o
            
            # Coeficiente segn distancia
            dist = self.config.manhattan_distance((0, 0), (dx, dy))
            coeff = self.config.interaction_coeff(dist)
            
            # Energa de esta interaccin
            energy = self.compute_energy_single_interaction(center_spin, neighbor_spin)
            total_energy += coeff * energy
        
        return total_energy


# ============================================================================
# MAPA DINAMICO DE ENTRELAZAMIENTO
# ============================================================================

class QuantumDynamicMapModel:
    """Mapa dinamico: estadistico del kernel local evolucionado bajo e^{-iHt}.

    A diferencia del mapa estatico (que factoriza y es exactamente el mapa
    cubico clasico), la evolucion temporal genera entrelazamiento porque los
    terminos Z(x)X y X(x)Z del Hamiltoniano no conmutan. Este modelo resume esa
    dinamica en un escalar por punto del tablero, integrable con
    EnergyMapGenerator.

    Estadisticos disponibles (`statistic`):
    - 'entropy_mean': promedio temporal de la entropia de von Neumann del qubit
      central (bits). Mide cuanta "tension cuantica" genera el entorno; >= 0.
    - 'entropy_max': maximo temporal de esa entropia.
    - 'z_mean': promedio temporal de <Z_0> (influencia dinamica; escala
      blanco-positivo como el mapa cubico).
    - 'conn_mean': promedio temporal de la correlacion conectada media
      |<Z_0 Z_j> - <Z_0><Z_j>| con los vecinos del kernel.
    """

    STATISTICS = ('entropy_mean', 'entropy_max', 'z_mean', 'conn_mean')

    def __init__(
        self,
        manhattan_distance: int = 1,
        *,
        times=None,
        steps: int = 4,
        statistic: str = 'entropy_mean',
    ):
        if statistic not in self.STATISTICS:
            raise ValueError(
                f"statistic debe ser uno de {self.STATISTICS}, no {statistic!r}"
            )
        self.quantum = QuantumIsingModel(manhattan_distance=manhattan_distance)
        self.manhattan_distance = int(manhattan_distance)
        self.times = (
            np.linspace(0.0, 2.0 * np.pi, 16) if times is None
            else np.asarray(times, dtype=float)
        )
        self.steps = int(steps)
        self.statistic = str(statistic)
        self.model_type = f'quantum-dynamic-{self.statistic}'

    def compute_energy(self, board: np.ndarray, x: int, y: int) -> float:
        records = self.quantum.entanglement_over_times(
            board, x, y, self.times, steps=self.steps
        )
        if self.statistic == 'entropy_mean':
            return float(np.mean([r['entropy_center'] for r in records]))
        if self.statistic == 'entropy_max':
            return float(np.max([r['entropy_center'] for r in records]))
        if self.statistic == 'z_mean':
            return float(np.mean([r['expZ'][0] for r in records]))
        # 'conn_mean'
        return float(np.mean([
            np.mean(np.abs(list(r['connected_zz'].values()))) for r in records
        ]))


# ============================================================================
# CLASE 4: GENERADOR DE MAPAS DE ENERGA
# ============================================================================

class EnergyMapGenerator:
    """
    Genera mapas de energÃ­a para tableros de Go completos.
    
    Integra modelos cuÃ¡nticos y clÃ¡sicos con GoBoard.
    """
    
    def __init__(self, model):
        """
        Args:
            model: Instancia con metodo compute_energy(board, i, j)
        """
        self.model = model
        if isinstance(model, QuantumIsingModel):
            self.model_type = 'quantum'
        elif isinstance(model, ClassicalIsingModel):
            self.model_type = 'legacy-classical'
        else:
            self.model_type = getattr(model, 'model_type', 'positional')
    
    def generate_energy_map(self, board: np.ndarray) -> np.ndarray:
        """
        Genera mapa de energÃ­a completo para un tablero.

        Si el modelo expone compute_map (vectorizado), se usa esa via;
        de lo contrario se itera compute_energy punto por punto.

        Args:
            board: Tablero de Go (array 2D)
            
        Returns:
            energy_map: Array 2D con energÃ­as
        """
        if hasattr(self.model, 'compute_map'):
            return np.asarray(self.model.compute_map(board), dtype=float)

        energy_map = np.zeros_like(board, dtype=float)

        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                energy_map[i, j] = self.model.compute_energy(board, i, j)

        return energy_map
    
    def compute_statistics(self, board: np.ndarray, energy_map: np.ndarray) -> Dict:
        """
        Calcula estadÃ­sticas del mapa de energÃ­a.
        
        Returns:
            dict con energÃ­as por color y espacios vacÃ­os
        """
        stats = {
            'black_energy': 0.0,
            'white_energy': 0.0,
            'empty_positive': 0.0,
            'empty_negative': 0.0,
            'total_energy': 0.0
        }
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                energy = energy_map[i, j]
                stone = board[i, j]
                
                if stone == 'B':
                    stats['black_energy'] += energy
                elif stone == 'W':
                    stats['white_energy'] += energy
                else:  # VacÃ­o
                    if energy > 0:
                        stats['empty_positive'] += energy
                    else:
                        stats['empty_negative'] += energy
                
                stats['total_energy'] += energy
        
        return stats


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def optimize_classical_hamiltonian(target_states: Dict[Tuple[float, float], float]) -> Dict:
    """
    Optimiza los parÃ¡metros del Hamiltoniano clÃ¡sico para reproducir
    valores esperados cuÃ¡nticos.
    
    Args:
        target_states: Dict {(s0, s1): energÃ­a_objetivo}
        
    Returns:
        ParÃ¡metros optimizados {'h0', 'h1', 'K', 'L'}
    
    Ejemplo de uso:
        >>> target = {
        ...     (1, 1): 1.0,
        ...     (1, -1): -1.0,
        ...     (-1, 1): 1.0,
        ...     (-1, -1): -1.0,
        ...     (0, 1): 2.0,
        ...     (0, -1): -2.0,
        ...     (0, 0): 0.0
        ... }
        >>> params = optimize_classical_hamiltonian(target)
    """
    def hamiltonian(params_array, s0, s1):
        h0, h1, K, L = params_array
        return h0*s0 + h1*s1 + K*s0*(s1**2) + L*(s0**2)*s1
    
    def error(params_array):
        err = 0.0
        for (s0, s1), target_energy in target_states.items():
            predicted = hamiltonian(params_array, s0, s1)
            err += (predicted - target_energy) ** 2
        return err
    
    # Optimizar
    initial_params = [1.0, 1.0, 1.0, 1.0]
    result = minimize(error, initial_params, method='Nelder-Mead')
    
    optimized = result.x
    return {
        'h0': round(optimized[0], 2),
        'h1': round(optimized[1], 2),
        'K': round(optimized[2], 2),
        'L': round(optimized[3], 2)
    }


def quantum_classical_parity(
    board: np.ndarray,
    manhattan_distance: int = 1,
    atol: float = 1e-8,
):
    """Compara mapas de energÃ­a cuÃ¡ntico vs clÃ¡sico y devuelve mÃ©tricas de paridad.

    Returns:
        dict con 'max_abs_diff', 'mean_abs_diff', y opcionalmente los mapas.
    """
    q_model = QuantumIsingModel(manhattan_distance=manhattan_distance)
    c_model = ClassicalIsingModel(manhattan_distance=manhattan_distance)
    q_map = EnergyMapGenerator(q_model).generate_energy_map(board)
    c_map = EnergyMapGenerator(c_model).generate_energy_map(board)
    diff = np.abs(q_map - c_map)
    return {
        'max_abs_diff': float(np.max(diff)),
        'mean_abs_diff': float(np.mean(diff)),
        'allclose': bool(np.all(diff <= atol)),
        'q_map': q_map,
        'c_map': c_map,
    }
