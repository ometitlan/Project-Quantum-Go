"""
go_energy_viz.py
================
Visualizacion Bokeh de mapas posicionales spin-1 (Manhattan-1/2).

Uso bÃ¡sico (en notebook):
    from src.sgf_utils import extract_moves_and_info, replay_game, board_to_numpy
    from src.go_energy_viz import build_energy_tabs

    moves, winner, info = extract_moves_and_info(path)
    gb = replay_game(moves, size=info.get('board_size', 19))
    board_np = board_to_numpy(gb)
    tabs = build_energy_tabs(board_np)
    show(tabs)
"""

from typing import Iterable, Tuple

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
try:
    # Prefer Bokeh >= 3.0 names
    from bokeh.models import Tabs, TabPanel
except Exception:
    # Fallback for Bokeh < 3.0
    from bokeh.models.widgets import Tabs  # type: ignore
    from bokeh.models.widgets import Panel as TabPanel  # type: ignore
from bokeh.palettes import Turbo256

# --- Simple helpers to build a two-color diverging palette ---
def _hex_to_rgb(h: str):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"

def _gradient(c1: str, c2: str, n: int):
    if n <= 1:
        return [c2]
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    cols = []
    for i in range(n):
        t = i / (n - 1)
        r = int(round(r1 * (1 - t) + r2 * t))
        g = int(round(g1 * (1 - t) + g2 * t))
        b = int(round(b1 * (1 - t) + b2 * t))
        cols.append(_rgb_to_hex((r, g, b)))
    return cols

def _two_color_diverging_palette(n: int = 256,
                                 neg_color: str = '#2563EB',  # blue-600
                                 pos_color: str = '#DC2626',  # red-600
                                 neutral_color: str = '#F5F5F5'):
    if n < 3:
        return [neg_color, pos_color]
    left = n // 2
    right = n - left
    neg_to_neutral = _gradient(neg_color, neutral_color, left)
    neutral_to_pos = _gradient(neutral_color, pos_color, right)
    return neg_to_neutral + neutral_to_pos

from .go_isings_models import (
    QuantumIsingModel,
    ClassicalIsingModel,
    EnergyMapGenerator,
    create_positional_map_model,
    map_method_label,
    normalize_map_method,
    parse_map_method,
    split_map_method,
)


DEFAULT_MAP_TABS = ('m1-cubic', 'm2-cubic', 'm1-quadratic', 'm2-quadratic')


def parse_map_tab_key(key: str):
    """Parse tab keys like m1-cubic, m2-quadratic, m1-cubic-hipB, m2-quadratic-raw.

    Sufijos combinables tras el método:
    - '-hipB'/'-hipW': lectura de jugada hipotética en los vacíos;
    - '-raw': sin promedio por capa (suma acumulada).
    Los sufijos se conservan dentro del campo `method` como clave compuesta
    ('cubic-hipb', 'quadratic-raw') para no romper a los consumidores de 4-tuplas.
    """
    raw = str(key).strip().lower().replace('_', '-')
    if not raw.startswith('m') or '-' not in raw:
        raise ValueError(f"Pestaña de mapa inválida: {key!r}")
    dist_part, method_part = raw.split('-', 1)
    try:
        dist = int(dist_part[1:])
    except ValueError as exc:
        raise ValueError(f"Distancia Manhattan inválida en {key!r}") from exc
    method, method_kwargs = parse_map_method(method_part)
    hyp_color = method_kwargs.get('hypothetical_color')
    if hyp_color is not None:
        method = f"{method}-hip{hyp_color.lower()}"
    if method_kwargs.get('normalize_by_layer') is False:
        method = f"{method}-raw"
    canonical_key = f"m{dist}-{method}"
    title = f"M{dist} | {map_method_label(method)}"
    return canonical_key, dist, method, title


def selected_map_tab_specs(keys: Iterable[str] | None = None):
    """Return unique (key, distance, method, title) specs for requested tabs."""
    requested = DEFAULT_MAP_TABS if keys is None else tuple(keys)
    specs = []
    seen = set()
    for key in requested:
        spec = parse_map_tab_key(key)
        if spec[0] in seen:
            continue
        seen.add(spec[0])
        specs.append(spec)
    return specs


def _compute_energy_map(board: np.ndarray, manhattan_distance: int, method: str, **map_kwargs) -> np.ndarray:
    canonical, method_kwargs = parse_map_method(method)
    if canonical in {'quantum', 'legacy-classical'}:
        if method_kwargs or map_kwargs.get('hypothetical_color'):
            raise ValueError(
                f"El backend {canonical!r} no soporta sufijos de mapa (hipotético/raw)"
            )
        if canonical == 'quantum':
            model = QuantumIsingModel(manhattan_distance=manhattan_distance)
        else:
            model = ClassicalIsingModel(manhattan_distance=manhattan_distance)
    else:
        for key, value in method_kwargs.items():
            map_kwargs.setdefault(key, value)
        model = create_positional_map_model(
            canonical,
            manhattan_distance=manhattan_distance,
            **map_kwargs,
        )
    gen = EnergyMapGenerator(model)
    return gen.generate_energy_map(board)


def _figure_for_energy(
    board: np.ndarray,
    energy_map: np.ndarray,
    title: str,
    *,
    board_color: str = '#F3F0D7',
    line_color: str = 'gray',
    show_hoshi: bool = True,
    overlay: str = 'aura',
    zero_threshold: float = 1e-9,
    aura_size: int = 34,
    aura_alpha: float = 0.55,
    palette: str = 'two_color',  # 'two_color' or 'turbo'
    neg_color: str = '#2563EB',  # blue-600
    pos_color: str = '#DC2626',  # red-600
    neutral_color: str = '#F5F5F5',
    symmetric: bool = True,
):
    h, w = board.shape
    low = float(np.min(energy_map)) if energy_map.size else -1.0
    high = float(np.max(energy_map)) if energy_map.size else 1.0
    if symmetric and energy_map.size:
        bound = float(max(abs(low), abs(high)))
        # Evitar rango cero
        if np.isclose(bound, 0.0):
            bound = 1.0
        low, high = -bound, bound
    elif np.isclose(low, high):
        low, high = low - 1.0, high + 1.0

    if palette == 'two_color':
        pal = _two_color_diverging_palette(256, neg_color=neg_color, pos_color=pos_color, neutral_color=neutral_color)
    else:
        pal = Turbo256
    mapper = LinearColorMapper(palette=pal, low=low, high=high)
    # Invertimos el eje Y para alinear con la vista de Matplotlib (origen arriba)
    p = figure(
        title=title,
        x_range=(-0.5, w - 0.5), y_range=(h - 0.5, -0.5),
        width=600, height=600, tools="hover,pan,wheel_zoom,reset,save",
        tooltips=[("Pos", "(@x, @y)"), ("Valor", "@energy{0.00}")]
    )
    p.background_fill_color = board_color
    p.grid.grid_line_color = None
    p.axis.visible = False
    p.outline_line_color = 'black'

    # Grid lines
    for i in range(h):
        p.line(x=[0, w - 1], y=[i, i], line_width=1, color=line_color)
    for j in range(w):
        p.line(x=[j, j], y=[0, h - 1], line_width=1, color=line_color)

    # Border
    p.line(x=[-0.5, w - 0.5], y=[-0.5, -0.5], line_width=2, color='black')
    p.line(x=[-0.5, -0.5], y=[-0.5, h - 0.5], line_width=2, color='black')
    p.line(x=[-0.5, w - 0.5], y=[h - 0.5, h - 0.5], line_width=2, color='black')
    p.line(x=[w - 0.5, w - 0.5], y=[-0.5, h - 0.5], line_width=2, color='black')

    # Capa del mapa
    if overlay == 'heatmap':
        xs = np.tile(np.arange(w), h)
        ys = np.repeat(np.arange(h), w)
        source = ColumnDataSource(dict(x=xs, y=ys, energy=energy_map.flatten()))
        p.rect(x='x', y='y', width=1, height=1, source=source,
               fill_color={'field': 'energy', 'transform': mapper},
               line_color=None, fill_alpha=0.9)
    else:
        xs, ys, vals = [], [], []
        for i in range(h):
            for j in range(w):
                # Mostrar aura en cualquier interseccion con valor significativo,
                # incluyendo espacios vacios bajo influencia.
                val = float(energy_map[i, j])
                if abs(val) > zero_threshold:
                    xs.append(j); ys.append(i); vals.append(val)
        if xs:
            src_aura = ColumnDataSource(dict(x=xs, y=ys, energy=vals))
            # Bokeh 3.4+: usar scatter(size=...) en lugar de circle(size=...)
            p.scatter(x='x', y='y', size=aura_size, fill_alpha=aura_alpha,
                      fill_color={'field': 'energy', 'transform': mapper},
                      line_color=None, source=src_aura)

    # Hoshi points
    if show_hoshi:
        hoshi = []
        if h == 19 and w == 19:
            hoshi = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
        elif h == 13 and w == 13:
            hoshi = [(3,3), (3,9), (6,6), (9,3), (9,9)]
        elif h == 9 and w == 9:
            hoshi = [(2,2), (2,6), (4,4), (6,2), (6,6)]
        if hoshi:
            p.scatter(x=[c for r,c in hoshi], y=[r for r,c in hoshi], size=8, color='black')

    # Piedras superpuestas
    bxs, bys, wxs, wys = [], [], [], []
    for i in range(h):
        for j in range(w):
            if board[i, j] == 'B':
                bxs.append(j); bys.append(i)
            elif board[i, j] == 'W':
                wxs.append(j); wys.append(i)
    if bxs:
        p.scatter(x=bxs, y=bys, size=16, color='black', line_color='black')
    if wxs:
        p.scatter(x=wxs, y=wys, size=16, color='white', line_color='black')

    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=10),
                         formatter=PrintfTickFormatter(format="%.2f"), location=(0, 0),
                         title="Mapa")
    p.add_layout(color_bar, 'right')
    return p


def _panel(board: np.ndarray, manhattan_distance: int, method: str) -> TabPanel:
    emap = _compute_energy_map(board, manhattan_distance, method)
    title = f"Manhattan-{manhattan_distance} | {map_method_label(method)}"
    fig = _figure_for_energy(board, emap, title, overlay='aura')
    lay = column(fig)
    return TabPanel(child=lay, title=title)


def build_energy_tabs(board: np.ndarray, energy_tabs: Iterable[str] | None = None) -> Tabs:
    """Construye Tabs con mapas solicitados y un panel 'Board Only'."""
    panels = [
        _panel(board, dist, method)
        for _, dist, method, _ in selected_map_tab_specs(energy_tabs)
    ]
    # Tab extra solo con tablero.
    fig_title = "Board Only"
    dummy = np.zeros_like(board, dtype=float)
    fig_only = _figure_for_energy(board, dummy, fig_title)
    panels.append(TabPanel(child=column(fig_only), title=fig_title))
    return Tabs(tabs=panels)
