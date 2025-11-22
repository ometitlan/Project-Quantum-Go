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

import pennylane as qml
import numpy as np
from typing import Dict, Tuple, Optional, Union
from scipy.optimize import minimize

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
    
    # Coeficientes de interaccin (ley 1/d para distancias > 1)
    INTERACTION_COEFFS = {
        1: 1.0,     # Vecinos inmediatos: peso completo
        2: 0.25,    # Manhattan=2: 1/4 del peso
        3: 0.111,   # Manhattan=3: 1/9 del peso
        4: 0.0625   # Manhattan=4: 1/16 del peso
    }
    
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
                coefficients[idx] = cls.INTERACTION_COEFFS.get(dist, 0.0)
        
        return {
            'positions': positions,
            'n_qubits': n_qubits,
            'coefficients': coefficients,
            'manhattan_distance': manhattan_distance
        }


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
    
    def __init__(self, manhattan_distance: int = 1, config: Optional[IsingGoConfig] = None, hamiltonian: Optional[qml.Hamiltonian] = None):
        """
        Inicializa el modelo cuÃ¡ntico.
        
        Args:
            manhattan_distance: Radio del kernel (1 o 2)
            config: ConfiguraciÃ³n personalizada (opcional)
        """
        self.config = config or IsingGoConfig()
        self.manhattan_distance = manhattan_distance
        
        # Obtener info del kernel
        kernel_info = self.config.get_kernel_info(manhattan_distance)
        self.positions = kernel_info['positions']
        self.n_qubits = kernel_info['n_qubits']
        self.coefficients = kernel_info['coefficients']
        
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

    def get_hamiltonian_info(self) -> Dict:
        """Retorna informaciÃ³n del Hamiltoniano."""
        return {
            'n_terms': len(self.hamiltonian.coeffs),
            'coefficients': self.hamiltonian.coeffs,
            'observables': [str(obs) for obs in self.hamiltonian.ops],
            'hamiltonian_str': str(self.hamiltonian)
        }


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
            coeff = self.config.INTERACTION_COEFFS.get(dist, 0.0)
            
            # Energa de esta interaccin
            energy = self.compute_energy_single_interaction(center_spin, neighbor_spin)
            total_energy += coeff * energy
        
        return total_energy


# ============================================================================
# CLASE 4: GENERADOR DE MAPAS DE ENERGA
# ============================================================================

class EnergyMapGenerator:
    """
    Genera mapas de energÃ­a para tableros de Go completos.
    
    Integra modelos cuÃ¡nticos y clÃ¡sicos con GoBoard.
    """
    
    def __init__(self, model: Union[QuantumIsingModel, ClassicalIsingModel]):
        """
        Args:
            model: Instancia de QuantumIsingModel o ClassicalIsingModel
        """
        self.model = model
        self.model_type = 'quantum' if isinstance(model, QuantumIsingModel) else 'classical'
    
    def generate_energy_map(self, board: np.ndarray) -> np.ndarray:
        """
        Genera mapa de energÃ­a completo para un tablero.
        
        Args:
            board: Tablero de Go (array 2D)
            
        Returns:
            energy_map: Array 2D con energÃ­as
        """
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
