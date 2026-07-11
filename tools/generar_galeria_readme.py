# -*- coding: utf-8 -*-
"""Genera la galería de mapas del README desde una partida profesional real.

Produce en data/assets/:
- readme_galeria_mapas.png : 6 mapas posicionales sobre la posición final 19x19
- readme_mapa_dinamico.png : mapa dinámico de entrelazamiento en una ventana 9x9

Ejecutar con el entorno del proyecto (conda goq):
    python tools/generar_galeria_readme.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.go_game_engine import GoBoard, SGFParser              # noqa: E402
from src.go_isings_models import (                             # noqa: E402
    PositionalMapModel, QuantumDynamicMapModel, EnergyMapGenerator,
)

SGF = ROOT / 'data' / 'sgf partidas' / '3a3d-gokifu-20210609-Peng_Liyao-Xia_Chenkun.sgf'
ASSETS = ROOT / 'data' / 'assets'

WOOD = '#e8dcc0'
# Escalas consistentes con las láminas comparativas:
# cúbico: oscuro = hacia negro, ámbar = hacia blanco
CMAP_CUB = LinearSegmentedColormap.from_list('cubico', ['#0f172a', '#f5f4ef', '#b45309'])
# cuadrático: azul = conexión favorecida, rojo = contacto penalizado
CMAP_QUAD = LinearSegmentedColormap.from_list('quad', ['#2563eb', '#f5f4ef', '#dc2626'])


def draw_board_panel(ax, board, m=None, title='', cmap=None, signed=True, stone_size=55):
    n, w = board.shape
    ax.set_facecolor(WOOD)
    im = None
    if m is not None:
        if signed:
            bound = float(np.max(np.abs(m))) or 1.0
            im = ax.imshow(m, cmap=cmap, vmin=-bound, vmax=bound, alpha=0.92)
        else:
            im = ax.imshow(m, cmap=cmap, vmin=0.0, alpha=0.92)
    else:
        ax.imshow(np.zeros(board.shape), cmap='Greys', vmin=0, vmax=1, alpha=0.0)
    for k in range(n):
        ax.axhline(k, color='#00000022', lw=0.5)
        ax.axvline(k, color='#00000022', lw=0.5)
    for i in range(n):
        for j in range(w):
            if board[i, j] == 'B':
                ax.scatter(j, i, s=stone_size, c='#111111', edgecolors='black',
                           linewidths=0.5, zorder=5)
            elif board[i, j] == 'W':
                ax.scatter(j, i, s=stone_size, c='white', edgecolors='black',
                           linewidths=0.7, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, w - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.set_title(title, fontsize=10.5)
    return im


def fig_galeria(board):
    specs = [
        (PositionalMapModel('cubic', manhattan_distance=1), 'Cúbico M1 — influencia local', CMAP_CUB, True),
        (PositionalMapModel('cubic', manhattan_distance=2), 'Cúbico M2 — influencia media', CMAP_CUB, True),
        (PositionalMapModel('cubic', manhattan_distance=9), 'Cúbico R=9 — influencia global', CMAP_CUB, True),
        (PositionalMapModel('quadratic', manhattan_distance=2), 'Cuadrático M2 — conexión / contacto', CMAP_QUAD, True),
        (PositionalMapModel('cubic', manhattan_distance=1, hypothetical_color='B'),
         'Hipotético Negro M1 — ¿y si juega negro?', CMAP_CUB, True),
        (PositionalMapModel('cubic', manhattan_distance=1, hypothetical_color='W'),
         'Hipotético Blanco M1 — ¿y si juega blanco?', CMAP_CUB, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, (model, title, cmap, signed) in zip(axes.flat, specs):
        m = model.compute_map(board)
        im = draw_board_panel(ax, board, m, title, cmap, signed)
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle('Mapas posicionales spin-1 — posición final, Peng Liyao vs Xia Chenkun (2021)\n'
                 'Escala cúbica: oscuro = hacia negro, ámbar = hacia blanco. '
                 'Escala cuadrática: azul = conexión, rojo = contacto.',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS / 'readme_galeria_mapas.png'
    fig.savefig(out, dpi=140, facecolor='#fbf9f3', bbox_inches='tight')
    plt.close(fig)
    print(out)


def fig_dinamico(board, r0, c0, size=9):
    window = board[r0:r0 + size, c0:c0 + size].copy()
    t0 = time.time()
    dyn = QuantumDynamicMapModel(manhattan_distance=1, statistic='entropy_mean',
                                 times=np.linspace(0, 2 * np.pi, 12))
    dmap = EnergyMapGenerator(dyn).generate_energy_map(window)
    print(f'mapa dinámico {size}x{size}: {time.time() - t0:.0f}s')

    mcub = PositionalMapModel('cubic', manhattan_distance=1).compute_map(window)
    r = float(np.corrcoef(dmap.flatten(), np.abs(mcub).flatten())[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    draw_board_panel(axes[0], window, None, f'Ventana {size}×{size} de la partida', stone_size=160)
    im1 = draw_board_panel(axes[1], window, dmap,
                           'Mapa dinámico: entropía media S̄(ρ₀) (bits)',
                           'viridis', signed=False, stone_size=160)
    plt.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.02)
    im2 = draw_board_panel(axes[2], window, np.abs(mcub),
                           '|Cúbico M1| — magnitud de influencia',
                           'viridis', signed=False, stone_size=160)
    plt.colorbar(im2, ax=axes[2], fraction=0.045, pad=0.02)
    fig.suptitle('Mapa dinámico de entrelazamiento bajo $e^{-iHt}$ — '
                 f'correlación con el mapa clásico: r = {r:+.2f} (información distinta)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = ASSETS / 'readme_mapa_dinamico.png'
    fig.savefig(out, dpi=140, facecolor='#fbf9f3', bbox_inches='tight')
    plt.close(fig)
    print(out)


if __name__ == '__main__':
    moves, info = SGFParser().parse_file(str(SGF))
    gb = GoBoard(size=info.board_size)
    gb.replay_moves(moves)
    board = np.array(gb.board, dtype=str)
    print(f'Partida: {info.black_player} vs {info.white_player}, {len(moves)} jugadas')
    fig_galeria(board)
    fig_dinamico(board, r0=3, c0=10)
