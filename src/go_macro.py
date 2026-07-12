# -*- coding: utf-8 -*-
"""
go_macro.py
===========
Observables macroscopicos del tablero de Go sobre la codificacion spin-1.

Tres niveles del proyecto:
- micro: cada interseccion s_i en {-1 (negro), 0 (vacio), +1 (blanco)}
- meso:  mapas posicionales sobre vecindades Manhattan (go_isings_models)
- macro: numeros que resumen todo el tablero (este modulo)

Observables implementados (todos vectorizados con convoluciones de anillos):
- magnetization / occupation: balance de color (~capturas) y fraccion ocupada
- correlation_profile: C_occ(R) (alineacion entre piedras) y O(R) (densidad
  de pares), separando orden de vacio
- correlation_length: xi a partir del decaimiento de |C_occ(R)|
- mutual_information_profile: MI(R) y entropia condicional H(s_j|s_i) en bits,
  sobre el alfabeto ternario completo (incluye vacios)
- site_entropy: entropia de Shannon de la distribucion {B, vacio, W}
- cubic_intensity: intensidad media |mapa cubico| (magnitud de influencia)
- fit_beta_structural: (betaJ, betaD) efectivos por pseudoverosimilitud de
  Besag sobre el Hamiltoniano estructural PAR (Blume-Capel de largo alcance):
      H = -J sum_{d(i,j)<=R} w_d s_i s_j + D sum_i s_i^2,  w_d = 1/d
  Solo los productos betaJ y betaD son identificables (se fija J=1). NO es
  temperatura termodinamica (el tablero no es una muestra de Gibbs) ni la
  temperatura CGT de Berlekamp: es un descriptor de organizacion.
- macro_state / macro_trajectory: el vector macro X_t por jugada, listo para
  graficarse o integrarse al motor de visualizacion.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve2d
from scipy.special import logsumexp

from .go_isings_models import (
    PositionalMapModel,
    board_to_spins,
    manhattan_ring_kernel,
)

_SPIN_VALUES = (-1.0, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Nivel 0: balance y ocupacion
# ----------------------------------------------------------------------------

def magnetization(board: np.ndarray) -> float:
    """M = promedio de s_i. Por la alternancia de turnos, en la practica lee
    el balance de capturas (M>0: negro ha perdido mas piedras)."""
    return float(np.mean(board_to_spins(board)))


def occupation(board: np.ndarray) -> float:
    """Q = promedio de s_i^2 (fraccion del tablero con piedra)."""
    s = board_to_spins(board)
    return float(np.mean(s * s))


def site_entropy(board: np.ndarray) -> float:
    """Entropia de Shannon (bits) de la distribucion de sitios {B, vacio, W}."""
    s = board_to_spins(board)
    n = s.size
    probs = np.array([(s == v).sum() / n for v in _SPIN_VALUES])
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


# ----------------------------------------------------------------------------
# Pares a distancia Manhattan R: conteos, correlacion e informacion mutua
# ----------------------------------------------------------------------------

def pair_joint_counts(board: np.ndarray, dist: int) -> np.ndarray:
    """Conteo N[a,b] de pares ORDENADOS (s_i=a, s_j=b) a distancia Manhattan
    `dist`. Indices 0,1,2 = valores -1, 0, +1. Simetrica por construccion."""
    s = board_to_spins(board)
    K = manhattan_ring_kernel(dist)
    planes = [(s == v).astype(float) for v in _SPIN_VALUES]
    convs = [convolve2d(p, K, mode='same', boundary='fill', fillvalue=0.0)
             for p in planes]
    N = np.zeros((3, 3))
    for ia in range(3):
        for ib in range(3):
            N[ia, ib] = float(np.sum(planes[ia] * convs[ib]))
    return N


def correlation_profile(board: np.ndarray, r_max: int):
    """Perfiles C_occ(R) y O(R) para R = 1..r_max.

    C_occ(R) = (pares mismo color - pares color opuesto) / pares ocupados
               en [-1, +1]; alineacion dado que ambos puntos tienen piedra.
    O(R)     = pares ocupados / todos los pares; densidad a esa escala.
    """
    C, O = [], []
    for R in range(1, int(r_max) + 1):
        N = pair_joint_counts(board, R)
        total = N.sum()
        same = N[0, 0] + N[2, 2]
        opp = N[0, 2] + N[2, 0]
        occ = same + opp
        C.append((same - opp) / occ if occ > 0 else np.nan)
        O.append(occ / total if total > 0 else np.nan)
    return np.array(C), np.array(O)


def mutual_information_profile(board: np.ndarray, r_max: int):
    """MI(R) y entropia condicional H(s_j | s_i), en bits, para R = 1..r_max.

    MI mide TODA la dependencia estadistica entre el estado de dos puntos a
    distancia R (colores y vacios incluidos): 0 = independencia. Complementa a
    C_occ, que solo ve la parte de alineacion de color entre piedras.
    """
    mi_list, hcond_list = [], []
    for R in range(1, int(r_max) + 1):
        N = pair_joint_counts(board, R)
        total = N.sum()
        if total == 0:
            mi_list.append(np.nan)
            hcond_list.append(np.nan)
            continue
        p = N / total
        pa = p.sum(axis=1)
        pb = p.sum(axis=0)
        mask = p > 0
        mi = float((p[mask] * np.log2(p[mask] / np.outer(pa, pb)[mask])).sum())
        h_joint = float(-(p[mask] * np.log2(p[mask])).sum())
        h_a = float(-(pa[pa > 0] * np.log2(pa[pa > 0])).sum())
        mi_list.append(mi)
        hcond_list.append(h_joint - h_a)   # H(s_j | s_i)
    return np.array(mi_list), np.array(hcond_list)


def decay_length(profile: np.ndarray, min_points: int = 3,
                 eps: float = 1e-3) -> float:
    """Longitud de decaimiento del ajuste ln|f(R)| = a - R/xi.

    Usa los R con |f| > eps; devuelve nan si hay menos de `min_points` o si
    el perfil no decae. En Go conviene aplicarlo al perfil MI(R) (positivo y
    decreciente); C_occ(R) oscila de signo y da ajustes fragiles. Nota: para
    correlaciones debiles MI ~ C^2, asi que la longitud de MI es ~ xi_C / 2.
    """
    c = np.asarray(profile, dtype=float)
    R = np.arange(1, c.size + 1, dtype=float)
    mask = np.isfinite(c) & (np.abs(c) > eps)
    if mask.sum() < min_points:
        return float('nan')
    slope, _ = np.polyfit(R[mask], np.log(np.abs(c[mask])), 1)
    return float(-1.0 / slope) if slope < 0 else float('nan')


# Alias retrocompatible
correlation_length = decay_length


# ----------------------------------------------------------------------------
# Intensidad del campo cubico (magnitud de influencia, sin signo)
# ----------------------------------------------------------------------------

def cubic_intensity(board: np.ndarray, r_max: int = 2) -> float:
    """I_R = promedio de |mapa cubico| con radio r_max."""
    m = PositionalMapModel('cubic', manhattan_distance=int(r_max)).compute_map(board)
    return float(np.mean(np.abs(m)))


# ----------------------------------------------------------------------------
# Beta estructural por pseudoverosimilitud de Besag
# ----------------------------------------------------------------------------

def structural_field(board: np.ndarray, r_max: int) -> np.ndarray:
    """Campo local del Hamiltoniano estructural: F_i = sum_R (1/R) (K_R * s)."""
    s = board_to_spins(board)
    F = np.zeros_like(s)
    for R in range(1, int(r_max) + 1):
        K = manhattan_ring_kernel(R)
        F += (1.0 / R) * convolve2d(s, K, mode='same', boundary='fill', fillvalue=0.0)
    return F


def fit_beta_structural(board: np.ndarray, r_max: int = 2,
                        bounds: float = 10.0) -> Dict:
    """Ajusta (betaJ, betaD) maximizando la pseudoverosimilitud de Besag.

    Modelo condicional por sitio (vecinos congelados):
        P(s_i = s | vecinos) ~ exp( betaJ * s * F_i - betaD * s^2 )
    con s en {-1, 0, +1} y F_i el campo estructural.

    Returns:
        dict con betaJ, betaD, pll (pseudo-log-verosimilitud media por sitio,
        en nats) y success.
    """
    s_obs = board_to_spins(board).ravel()
    F = structural_field(board, r_max).ravel()
    states = np.array(_SPIN_VALUES)

    def neg_pll(params):
        bJ, bD = params
        # logits[s, i] = bJ*s*F_i - bD*s^2
        logits = bJ * states[:, None] * F[None, :] - bD * (states ** 2)[:, None]
        logZ = logsumexp(logits, axis=0)
        logits_obs = bJ * s_obs * F - bD * s_obs ** 2
        return -float(np.mean(logits_obs - logZ))

    res = minimize(neg_pll, x0=np.array([0.5, 0.0]), method='L-BFGS-B',
                   bounds=[(-bounds, bounds), (-bounds, bounds)])
    return {
        'betaJ': float(res.x[0]),
        'betaD': float(res.x[1]),
        'pll': -float(res.fun),
        'success': bool(res.success),
    }


# ----------------------------------------------------------------------------
# El vector macro X_t
# ----------------------------------------------------------------------------

def macro_state(board: np.ndarray, r_max: int = 6, beta_r: int = 2) -> Dict:
    """Vector macro de una posicion, como dict plano (una fila de DataFrame).

    Args:
        board: tablero 'B'/'W'/'.'
        r_max: alcance de los perfiles C_occ / O / MI
        beta_r: alcance del Hamiltoniano estructural para betaJ/betaD
    """
    C, O = correlation_profile(board, r_max)
    MI, Hc = mutual_information_profile(board, r_max)
    beta = fit_beta_structural(board, r_max=beta_r)
    out = {
        'M': magnetization(board),
        'Q': occupation(board),
        'H_sitios': site_entropy(board),
        'xi_info': decay_length(MI),
        'I_cubico': cubic_intensity(board, r_max=2),
        'betaJ': beta['betaJ'],
        'betaD': beta['betaD'],
        'pll': beta['pll'],
    }
    for k, (c, o, mi, hc) in enumerate(zip(C, O, MI, Hc), start=1):
        out[f'C_occ_{k}'] = float(c)
        out[f'O_{k}'] = float(o)
        out[f'MI_{k}'] = float(mi)
        out[f'Hcond_{k}'] = float(hc)
    return out


def macro_trajectory(moves: List[Dict], board_size: int = 19, *,
                     r_max: int = 6, beta_r: int = 2, step: int = 1,
                     board_cls=None):
    """Trayectoria macro X_t de una partida completa.

    Reproduce la partida jugada a jugada (incremental) y calcula macro_state
    cada `step` jugadas. Devuelve un pandas.DataFrame indexado por numero de
    jugada, con columnas extra 'color' y 'piedras_jugadas' — el formato
    pensado para graficarse por jugada en el motor de visualizacion.
    """
    import pandas as pd
    if board_cls is None:
        from .go_game_engine import GoBoard as board_cls

    gb = board_cls(size=board_size)
    rows = []
    stones_played = 0
    for t, move in enumerate(moves, start=1):
        results = gb.replay_moves([move])
        if results and results[-1].is_legal:
            stones_played += 1
        if t % step != 0 and t != len(moves):
            continue
        board = np.array(gb.board, dtype=str)
        row = macro_state(board, r_max=r_max, beta_r=beta_r)
        row['jugada'] = t
        row['color'] = move.get('color', '?')
        row['piedras_jugadas'] = stones_played
        # fraccion capturada acumulada: piedras jugadas que ya no estan
        n = board.size
        row['frac_capturada'] = (stones_played / n) - row['Q']
        rows.append(row)

    df = pd.DataFrame(rows).set_index('jugada')
    return df
