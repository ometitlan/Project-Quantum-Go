"""
go_export_utils.py
===================
Utilidades de exportacion para mapas posicionales (PNG/HTML) y GIF con overlay.

Uso tÃƒÂ­pico (en notebook):
    from src.go_export_utils import (
        export_positional_map_image,
        export_energy_tabs_html,
        create_move_animation_map,
    )
"""

from typing import Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def export_positional_map_image(
    board,
    filename: str,
    *,
    manhattan_distance: int = 1,
    method: str = 'cubic',
    dpi: int = 300,
    output_dir: str = "../results",
    figsize=(12, 12),
    cmap: str = 'coolwarm',
    alpha: float = 0.45,
    symmetric: bool = True,
):
    """Exporta un PNG con tablero + mapa posicional superpuesto (Matplotlib)."""
    from src.go_visualization import GoBoardVisualizer
    from src.go_energy_viz import _compute_energy_map as _cem
    from src.go_isings_models import map_method_label

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    board_np = np.array(board.board, dtype=str)
    emap = _cem(board_np, manhattan_distance, method)

    if emap.size:
        if symmetric:
            v = float(max(abs(np.min(emap)), abs(np.max(emap)))) or 1.0
            vmin, vmax = -v, v
        else:
            vmin, vmax = float(np.min(emap)), float(np.max(emap))
            if vmin == vmax:
                vmin, vmax = vmin - 1.0, vmax + 1.0
    else:
        vmin, vmax = -1.0, 1.0

    vis = GoBoardVisualizer(board)
    fig, ax = vis.plot_matplotlib(
        title=f"Mapa posicional (M{manhattan_distance} | {map_method_label(method)})",
        show_liberties=False,
        show_move_numbers=False,
        figsize=figsize,
    )
    ax.imshow(emap, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', alpha=alpha, zorder=1)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Imagen de mapa posicional guardada: {output_path}")


def export_energy_map_image(*args, **kwargs):
    """Compatibilidad: usa export_positional_map_image."""
    return export_positional_map_image(*args, **kwargs)


def export_positional_map_tabs_html(
    board,
    filename: str = "positional_map_tabs.html",
    *,
    output_dir: str = "../results",
):
    """Exporta un HTML interactivo (Bokeh) con tabs de mapas posicionales."""
    from bokeh.embed import file_html as _bokeh_file_html
    from bokeh.resources import INLINE as _BK_INLINE
    from src.go_energy_viz import build_energy_tabs

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    board_np = np.array(board.board, dtype=str)
    tabs = build_energy_tabs(board_np)
    html = _bokeh_file_html(tabs, _BK_INLINE, title="Mapas posicionales")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML interactivo de mapas guardado: {output_path}")


def export_energy_tabs_html(*args, **kwargs):
    """Compatibilidad: usa export_positional_map_tabs_html."""
    return export_positional_map_tabs_html(*args, **kwargs)


def create_move_animation_map(
    navigator,
    output_file: str = "game_animation_map.gif",
    interval: int = 500,
    output_dir: str = "../results",
    max_frames: int | None = None,
    *,
    figsize=(10, 10),
    map_manhattan: int = 1,
    map_method: str = 'cubic',
    map_cmap: str = 'coolwarm',
    map_alpha: float = 0.45,
    symmetric_map: bool = True,
    **legacy_kwargs,
):
    """Crea un GIF con tablero y overlay de mapa posicional por frame.

    Nota: calcular mapas por frame puede ser costoso en partidas largas.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib
    from src.go_energy_viz import _compute_energy_map as _cem
    from src.go_isings_models import map_method_label

    if 'energy_manhattan' in legacy_kwargs:
        map_manhattan = legacy_kwargs.pop('energy_manhattan')
    if 'energy_method' in legacy_kwargs:
        map_method = legacy_kwargs.pop('energy_method')
    if 'energy_cmap' in legacy_kwargs:
        map_cmap = legacy_kwargs.pop('energy_cmap')
    if 'energy_alpha' in legacy_kwargs:
        map_alpha = legacy_kwargs.pop('energy_alpha')
    if 'symmetric_energy' in legacy_kwargs:
        symmetric_map = legacy_kwargs.pop('symmetric_energy')
    if legacy_kwargs:
        unknown = ', '.join(sorted(legacy_kwargs))
        raise TypeError(f"Argumentos no soportados: {unknown}")

    matplotlib.use('Agg')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    num_frames = len(navigator.moves) + 1
    if max_frames and num_frames > max_frames:
        num_frames = max_frames
        print(f"[Aviso] Limitando a {max_frames} frames (partida larga)")

    print(f"Creando GIF con {num_frames} frames...")
    print(f"   Guardando en: {output_path}")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    def update(frame):
        if frame % 20 == 0:
            print(f"   Frame {frame}/{num_frames}...")
        ax.clear()
        navigator.go_to_move(frame)

        # Dibujo base reutilizando el visualizador central para evitar duplicacion
        from src.go_visualization import GoBoardVisualizer as _Vis
        _Vis(navigator.board).draw_on_axes(
            ax,
            title=f"Frame {frame}/{num_frames} - Mapa M{map_manhattan} {map_method_label(map_method)}",
            show_liberties=False,
            show_captures=False,
            show_move_numbers=False,
            highlight_last_move=True,
        )

        # Overlay del mapa por frame.
        board_np = np.array(navigator.board.board, dtype=str)
        emap = _cem(board_np, map_manhattan, map_method)
        if emap.size:
            if symmetric_map:
                v = float(max(abs(np.min(emap)), abs(np.max(emap)))) or 1.0
                vmin, vmax = -v, v
            else:
                vmin, vmax = float(np.min(emap)), float(np.max(emap))
                if vmin == vmax:
                    vmin, vmax = vmin - 1.0, vmax + 1.0
        else:
            vmin, vmax = -1.0, 1.0
        ax.imshow(emap, cmap=map_cmap, vmin=vmin, vmax=vmax, origin='upper', alpha=map_alpha, zorder=1)

        if frame > 0 and navigator.board.move_history:
            lr, lc = navigator.board.move_history[-1].position
            ax.add_patch(Circle((lc, lr), 0.15, facecolor='red', zorder=4))
        ax.set_xlim(-0.5, navigator.board.size - 0.5)
        ax.set_ylim(-0.5, navigator.board.size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(navigator.board.size))
        ax.set_yticks(range(navigator.board.size))
        letters = [chr(i) for i in range(ord('A'), ord('T') + 1) if chr(i) != 'I']
        ax.set_xticklabels(letters[: navigator.board.size])
        ax.set_yticklabels(range(1, navigator.board.size + 1))
        ax.set_title(f"Frame {frame}/{num_frames} - Mapa M{map_manhattan} {map_method_label(map_method)}")

        return []

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    writer = PillowWriter(fps=max(1, int(1000 / max(1, interval))))
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"GIF guardado: {output_path}")


def create_move_animation_energy(*args, **kwargs):
    """Compatibilidad: usa create_move_animation_map."""
    return create_move_animation_map(*args, **kwargs)

