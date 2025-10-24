"""
go_visualization.py
===================
VisualizaciÃ³n limpia (UTF-8) para tableros de Go y un navegador interactivo.

Mantiene la API pÃºblica esperada:
- GoBoardVisualizer
- GameNavigator
- export_position_image, create_move_animation, compare_positions

El navegador ofrece pestaÃ±as superiores: Libertades, Jugadas y (opcional)
mapas de energÃ­a: M1-Quantum, M1-Classical, M2-Quantum, M2-Classical.
"""

from typing import List, Dict, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import clear_output
from ipywidgets import IntSlider, Button, ToggleButtons, HBox, VBox, Output, Layout, Tab, Checkbox


class GoBoardVisualizer:
    """Visualizador de tablero con Matplotlib (madera, cuadrÃ­cula, hoshi y piedras)."""

    def __init__(self, board):
        self.board = board

    def _get_move_number(self, row: int, col: int) -> int:
        for i, move in enumerate(self.board.move_history):
            if move.position == (row, col):
                return i + 1
        return 0

    def plot_matplotlib(
        self,
        title: str = "Tablero de Go",
        show_liberties: bool = False,
        show_captures: bool = True,
        show_move_numbers: bool = False,
        highlight_last_move: bool = True,
        figsize=(10, 10),
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_facecolor('#DEB887')  # madera

        # cuadrcula
        for i in range(self.board.size):
            ax.plot([0, self.board.size - 1], [i, i], 'k-', linewidth=1)
            ax.plot([i, i], [0, self.board.size - 1], 'k-', linewidth=1)

        # hoshi por tamaos comunes
        hoshi = []
        if self.board.size == 19:
            hoshi = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif self.board.size == 13:
            hoshi = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self.board.size == 9:
            hoshi = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        for x, y in hoshi:
            ax.plot(x, y, 'o', color='black', markersize=8 if self.board.size >= 13 else 6)

        # piedras y anotaciones
        liberty_done = set()
        for r in range(self.board.size):
            for c in range(self.board.size):
                stone = self.board.board[r, c]
                if stone == 'B':
                    ax.add_patch(Circle((c, r), 0.45, facecolor='black', zorder=2))
                    if show_move_numbers:
                        num = self._get_move_number(r, c)
                        if num:
                            ax.text(c, r, str(num), ha='center', va='center', color='white', fontsize=8, zorder=3)
                    if show_liberties:
                        libs = self.board.count_liberties(r, c)
                        group = tuple(sorted(self.board._get_group(r, c)))
                        if group not in liberty_done:
                            ax.text(c, r, str(libs), ha='center', va='center', color='white', fontsize=10, zorder=3)
                            liberty_done.add(group)
                elif stone == 'W':
                    ax.add_patch(Circle((c, r), 0.45, facecolor='white', edgecolor='black', linewidth=1.5, zorder=2))
                    if show_move_numbers:
                        num = self._get_move_number(r, c)
                        if num:
                            ax.text(c, r, str(num), ha='center', va='center', color='black', fontsize=8, zorder=3)
                    if show_liberties:
                        libs = self.board.count_liberties(r, c)
                        group = tuple(sorted(self.board._get_group(r, c)))
                        if group not in liberty_done:
                            ax.text(c, r, str(libs), ha='center', va='center', color='black', fontsize=10, zorder=3)
                            liberty_done.add(group)

        if highlight_last_move and self.board.move_history:
            lr, lc = self.board.move_history[-1].position
            ax.add_patch(Circle((lc, lr), 0.15, facecolor='red', zorder=4))

        ax.set_xlim(-0.5, self.board.size - 0.5)
        ax.set_ylim(-0.5, self.board.size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.board.size))
        ax.set_yticks(range(self.board.size))
        letters = [chr(i) for i in range(ord('A'), ord('T') + 1) if chr(i) != 'I']
        ax.set_xticklabels(letters[: self.board.size])
        ax.set_yticklabels(range(1, self.board.size + 1))

        if show_captures:
            title += f"\nMovimiento {len(self.board.move_history)} | Capturas - Negro: {self.board.captures['B']}, Blanco: {self.board.captures['W']}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig, ax

    def draw_on_axes(
        self,
        ax,
        *,
        title: str = "Tablero de Go",
        show_liberties: bool = False,
        show_captures: bool = True,
        show_move_numbers: bool = False,
        highlight_last_move: bool = True,
    ):
        # Clear and draw using the same style as plot_matplotlib
        ax.clear()
        ax.set_facecolor('#DEB887')  # madera
        for i in range(self.board.size):
            ax.plot([0, self.board.size - 1], [i, i], 'k-', linewidth=1)
            ax.plot([i, i], [0, self.board.size - 1], 'k-', linewidth=1)
        hoshi = []
        if self.board.size == 19:
            hoshi = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif self.board.size == 13:
            hoshi = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif self.board.size == 9:
            hoshi = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        for x, y in hoshi:
            ax.plot(x, y, 'o', color='black', markersize=8 if self.board.size >= 13 else 6)
        liberty_done = set()
        for r in range(self.board.size):
            for c in range(self.board.size):
                stone = self.board.board[r, c]
                if stone == 'B':
                    ax.add_patch(Circle((c, r), 0.45, facecolor='black', zorder=2))
                    if show_move_numbers:
                        num = self._get_move_number(r, c)
                        if num:
                            ax.text(c, r, str(num), ha='center', va='center', color='white', fontsize=8, zorder=3)
                    if show_liberties:
                        libs = self.board.count_liberties(r, c)
                        group = tuple(sorted(self.board._get_group(r, c)))
                        if group not in liberty_done:
                            ax.text(c, r, str(libs), ha='center', va='center', color='white', fontsize=10, zorder=3)
                            liberty_done.add(group)
                elif stone == 'W':
                    ax.add_patch(Circle((c, r), 0.45, facecolor='white', edgecolor='black', linewidth=1.5, zorder=2))
                    if show_move_numbers:
                        num = self._get_move_number(r, c)
                        if num:
                            ax.text(c, r, str(num), ha='center', va='center', color='black', fontsize=8, zorder=3)
                    if show_liberties:
                        libs = self.board.count_liberties(r, c)
                        group = tuple(sorted(self.board._get_group(r, c)))
                        if group not in liberty_done:
                            ax.text(c, r, str(libs), ha='center', va='center', color='black', fontsize=10, zorder=3)
                            liberty_done.add(group)
        if highlight_last_move and self.board.move_history:
            lr, lc = self.board.move_history[-1].position
            ax.add_patch(Circle((lc, lr), 0.15, facecolor='red', zorder=4))
        ax.set_xlim(-0.5, self.board.size - 0.5)
        ax.set_ylim(-0.5, self.board.size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(self.board.size))
        ax.set_yticks(range(self.board.size))
        letters = [chr(i) for i in range(ord('A'), ord('T') + 1) if chr(i) != 'I']
        ax.set_xticklabels(letters[: self.board.size])
        ax.set_yticklabels(range(1, self.board.size + 1))
        if show_captures:
            title += f"\nMovimiento {len(self.board.move_history)} | Capturas - Negro: {self.board.captures['B']}, Blanco: {self.board.captures['W']}"
        ax.set_title(title, fontsize=14, fontweight='bold')


class GameNavigator:
    """NavegaciÃ³n interactiva con tabs superiores (Libertades, Jugadas y EnergÃ­a)."""

    def __init__(self, moves: List[Dict], board_size: int = 19):
        from src.go_game_engine import GoBoard
        self.moves = moves
        self.board_size = int(board_size)
        self.GoBoard = GoBoard
        self.board = GoBoard(size=self.board_size)
        self.current_move = 0

    def _rebuild_to(self, move: int):
        self.board = self.GoBoard(size=self.board_size)
        if move > 0:
            self.board.replay_moves(self.moves[:move])
        self.current_move = move

    def go_to_move(self, move: int):
        """Public wrapper to jump to a given move index."""
        self._rebuild_to(int(move))

    def create_analysis_view(self, figsize=(9, 9), include_energy_tab: bool = False, debug: bool = False, energy_backend: str = 'bokeh', energy_tabs: Optional[List[str]] = None):
        """Conserva compatibilidad: si include_energy_tab=True, usa pestaÃ±as de energÃ­a arriba.

        Args:
            figsize: tamaÃ±o de figura
            include_energy_tab: si True, aÃ±ade pestaÃ±as de energÃ­a (M1/M2 Ã— Q/C)
            debug: si True, imprime logs (min/max energÃ­a, shapes) en cada pestaÃ±a
        """
        return self.create_view(figsize=figsize, include_energy_tabs=include_energy_tab, debug=debug, energy_backend=energy_backend, energy_tabs=energy_tabs)

    def create_view(self, figsize=(9, 9), include_energy_tabs: bool = True, debug: bool = False, energy_backend: str = 'bokeh', energy_tabs: Optional[List[str]] = None):
        # outputs
        out_lib = Output(); out_moves = Output()
        out_m1_q = Output(); out_m1_c = Output(); out_m2_q = Output(); out_m2_c = Output()

        # Asegurar Bokeh en el notebook una sola vez
        if include_energy_tabs:
            try:
                from bokeh.io import output_notebook as _bn_out
                _bn_out()
            except Exception:
                pass

        def update_all():
            vis = GoBoardVisualizer(self.board)
            with out_lib:
                clear_output(wait=True)
                fig, _ = vis.plot_matplotlib(
                    title=f"Movimiento {self.current_move}/{len(self.moves)}",
                    show_liberties=True,
                    figsize=figsize,
                ); plt.show()
            with out_moves:
                clear_output(wait=True)
                fig, _ = vis.plot_matplotlib(
                    title=f"Movimiento {self.current_move}/{len(self.moves)}",
                    show_move_numbers=True,
                    figsize=figsize,
                ); plt.show()

            if include_energy_tabs:
                try:
                    from src.go_energy_viz import _compute_energy_map as _cem, _figure_for_energy as _fig
                    board_np = np.array(self.board.board, dtype=str)
                    # Seleccin de pestaas a renderizar
                    opt = [
                        ("m1-quantum",   1, 'quantum',  'M1 | Quantum',   out_m1_q),
                        ("m1-classical", 1, 'classical','M1 | Classical', out_m1_c),
                        ("m2-quantum",   2, 'quantum',  'M2 | Quantum',   out_m2_q),
                        ("m2-classical", 2, 'classical','M2 | Classical', out_m2_c),
                    ]
                    allowed = set(t.lower() for t in (energy_tabs if energy_tabs is not None else ['m2-quantum']))
                    sel = [o for o in opt if o[0] in allowed]

                    def _render_bokeh(out, dist, method, title):
                        from bokeh.embed import file_html as _bokeh_file_html
                        from bokeh.resources import INLINE as _BK_INLINE
                        from IPython.display import HTML as _IPY_HTML, display as _ipy_display
                        import matplotlib.pyplot as _plt
                        with out:
                            clear_output(wait=True)
                            emap = _cem(board_np, dist, method)

                            # Generar figura Bokeh (puede internamente crear figuras MPL; las cerramos)
                            bokeh_fig = _fig(board_np, emap, title)
                            try:
                                # Cerrar cualquier figura matplotlib que _fig pudiera haber creado
                                _plt.close('all')
                            except Exception:
                                pass

                            # Mostrar Bokeh (mapa) en el output
                            html = _bokeh_file_html(bokeh_fig, _BK_INLINE, title)
                            _ipy_display(_IPY_HTML(html))

                            # Barras compactas (MPL) ABAJO
                            if emap.size:
                                e_w = float(np.sum(emap[emap > 0.0])) if np.any(emap > 0) else 0.0
                                e_b = -float(np.sum(emap[emap < 0.0])) if np.any(emap < 0) else 0.0
                            else:
                                e_w = e_b = 0.0
                            eps = 1e-9
                            total = max(e_w + e_b, eps)
                            sw = e_w/total if total else 0.0
                            sb = e_b/total if total else 0.0
                            vmax = max(e_w, e_b, 1.0)
                            fig_s, ax_s = _plt.subplots(1, 1, figsize=(3.6, 2.8))
                            ax_s.bar(['Blanco','Negro'], [e_w, e_b], color=['#e5e7eb', '#111827'])
                            ax_s.set_ylim(0, vmax*1.15)
                            ax_s.text(0, e_w + vmax*0.03, '{:.1f}\n({:.1f}%)'.format(e_w, sw*100),
                                      ha='center', va='bottom', fontsize=9, color='#374151')
                            ax_s.text(1, e_b + vmax*0.03, '{:.1f}\n({:.1f}%)'.format(e_b, sb*100),
                                      ha='center', va='bottom', fontsize=9, color='#374151')
                            ax_s.set_title('Energia BRUTA (M2)'); ax_s.set_ylabel('Suma de energia')
                            ax_s.ticklabel_format(style='plain', axis='y')
                            fig_s.tight_layout()
                            _plt.show()

                    def _render_mpl(out, dist, method, title):
                        import matplotlib.pyplot as _plt
                        with out:
                            clear_output(wait=True)
                            if debug:
                                print(f"[Energy/MPL] Computing {title}â€¦ board shape={board_np.shape}")
                            emap = _cem(board_np, dist, method)
                            v = float(max(abs(np.min(emap)), abs(np.max(emap)))) or 1.0
                            # Base del tablero con el mismo estilo que GoBoardVisualizer
                            vis = GoBoardVisualizer(self.board)
                            fig, ax = vis.plot_matplotlib(title=title, show_liberties=False, show_move_numbers=False, figsize=(5.0, 5.0))
                            ax.imshow(emap, cmap='coolwarm', vmin=-v, vmax=v, origin='upper', alpha=0.45, zorder=1)
                            _plt.tight_layout(); _plt.show()
                            # Barras compactas (MPL) bajo el mapa
                            if emap.size:
                                _eW = float(np.sum(emap[emap > 0.0])) if np.any(emap > 0) else 0.0
                                _eB = -float(np.sum(emap[emap < 0.0])) if np.any(emap < 0) else 0.0
                            else:
                                _eW = _eB = 0.0
                            _eps = 1e-9; _total = max(_eW + _eB, _eps)
                            _sw, _sb = _eW/_total, _eB/_total
                            _vmax = max(_eW, _eB, 1.0)
                            _figS, _axS = _plt.subplots(1, 1, figsize=(4.6, 3.8))
                            _axS.bar(['Blanco','Negro'], [_eW, _eB], color=['#e5e7eb', '#111827'])
                            _axS.set_ylim(0, _vmax*1.15)
                            _axS.text(0, _eW + _vmax*0.03, f"{_eW:.1f}\n({(_sw*100):.1f}%)", ha='center', va='bottom', fontsize=9, color='#374151')
                            _axS.text(1, _eB + _vmax*0.03, f"{_eB:.1f}\n({(_sb*100):.1f}%)", ha='center', va='bottom', fontsize=9, color='#374151')
                            _axS.set_title('Energia BRUTA (M2)'); _axS.set_ylabel('Suma de energia')
                            _axS.ticklabel_format(style='plain', axis='y')
                            _figS.tight_layout(); _plt.show()
                            if debug:
                                print(f"[Energy/MPL] {title} rendered OK (Matplotlib)")

                    if energy_backend == 'mpl':
                        for _, dist, method, title, out in sel:
                            _render_mpl(out, dist, method, title)
                    else:
                        for _, dist, method, title, out in sel:
                            _render_bokeh(out, dist, method, title)
                except Exception as e:
                    for out in (out_m1_q, out_m1_c, out_m2_q, out_m2_c):
                        with out:
                            clear_output(wait=True)
                            print('[Aviso] No se pudo renderizar Bokeh:', e)

        # controls
        slider = IntSlider(value=0, min=0, max=len(self.moves), step=1,
                           description='Movimiento:', continuous_update=False,
                           layout=Layout(width='600px'), style={'description_width': 'initial'})
        btn_first = Button(description='Inicio', button_style='info', layout=Layout(width='100px'))
        btn_prev10 = Button(description='-10', layout=Layout(width='80px'))
        btn_prev = Button(description='Anterior', button_style='warning', layout=Layout(width='100px'))
        btn_next = Button(description='Siguiente', button_style='success', layout=Layout(width='110px'))
        btn_next10 = Button(description='+10', layout=Layout(width='80px'))
        btn_last = Button(description='Final', button_style='info', layout=Layout(width='100px'))

        def _jump(val):
            self._rebuild_to(val); update_all()

        slider.observe(lambda ch: _jump(ch['new']), names='value')
        btn_first.on_click(lambda _: _jump(0))
        btn_prev10.on_click(lambda _: _jump(max(0, slider.value - 10)))
        btn_prev.on_click(lambda _: _jump(max(0, slider.value - 1)))
        btn_next.on_click(lambda _: _jump(min(len(self.moves), slider.value + 1)))
        btn_next10.on_click(lambda _: _jump(min(len(self.moves), slider.value + 10)))
        btn_last.on_click(lambda _: _jump(len(self.moves)))

        # tabs superiores
        if include_energy_tabs:
            # Construir children segn seleccin
            mapping = [
                ("m1-quantum",   out_m1_q, 'M1-Quantum'),
                ("m1-classical", out_m1_c, 'M1-Classical'),
                ("m2-quantum",   out_m2_q, 'M2-Quantum'),
                ("m2-classical", out_m2_c, 'M2-Classical'),
            ]
            allowed = set(t.lower() for t in (energy_tabs if energy_tabs is not None else ['m2-quantum']))
            order = [m for m in mapping if m[0] in allowed]
            children = [out_lib, out_moves] + [m[1] for m in order]
            tabs = Tab(children=children)
            try:
                tabs.set_title(0, 'Libertades'); tabs.set_title(1, 'Jugadas')
                for idx, m in enumerate(order, start=2):
                    tabs.set_title(idx, m[2])
            except Exception:
                pass
        else:
            tabs = Tab(children=[out_lib, out_moves])
            tabs.set_title(0, 'Libertades'); tabs.set_title(1, 'Jugadas')

        # render inicial
        self._rebuild_to(0)
        update_all()
        controls = VBox([slider, HBox([btn_first, btn_prev10, btn_prev, btn_next, btn_next10, btn_last])])
        return VBox([controls, tabs])


# ========================= UTILITARIAS ========================= #

def export_position_image(board, filename, show_liberties=False, show_move_numbers=False, dpi=300, output_dir="../results"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    vis = GoBoardVisualizer(board)
    title = "PosiciÃ³n - Libertades" if show_liberties else ("PosiciÃ³n - NÃºmeros de Jugada" if show_move_numbers else "PosiciÃ³n")
    fig, _ = vis.plot_matplotlib(title=title, show_liberties=show_liberties, show_move_numbers=show_move_numbers, figsize=(12, 12))
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"âœ… Imagen guardada: {output_path}")


def create_move_animation(navigator, output_file="game_animation.gif", interval=500, output_dir="../results", max_frames=None):
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib
    matplotlib.use('Agg')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    num_frames = len(navigator.moves) + 1
    if max_frames and num_frames > max_frames:
        num_frames = max_frames
        print(f"âš ï¸ Limitando a {max_frames} frames (partida muy larga)")

    print(f"ðŸŽ¬ Creando GIF con {num_frames} frames...")
    print(f"   Guardando en: {output_path}")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    def update(frame):
        if frame % 20 == 0:
            print(f"   Frame {frame}/{num_frames}...")
        ax.clear()
        navigator.go_to_move(frame)
        vis = GoBoardVisualizer(navigator.board)
        ax.set_facecolor('#DEB887')
        for i in range(navigator.board.size):
            ax.plot([0, navigator.board.size-1], [i, i], 'k-', linewidth=1)
            ax.plot([i, i], [0, navigator.board.size-1], 'k-', linewidth=1)
        if navigator.board.size == 19:
            hoshi = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
        elif navigator.board.size == 13:
            hoshi = [(3,3), (3,9), (6,6), (9,3), (9,9)]
        elif navigator.board.size == 9:
            hoshi = [(2,2), (2,6), (4,4), (6,2), (6,6)]
        else:
            hoshi = []
        for x, y in hoshi:
            ax.plot(x, y, 'o', color='black', markersize=8 if navigator.board.size >= 13 else 6)
        for r in range(navigator.board.size):
            for c in range(navigator.board.size):
                s = navigator.board.board[r, c]
                if s == 'B':
                    ax.add_patch(Circle((c, r), 0.45, facecolor='black', zorder=2))
                elif s == 'W':
                    ax.add_patch(Circle((c, r), 0.45, facecolor='white', edgecolor='black', linewidth=1.5, zorder=2))
        if frame > 0 and navigator.board.move_history:
            lr, lc = navigator.board.move_history[-1].position
            ax.add_patch(Circle((lc, lr), 0.15, facecolor='red', zorder=4))
        ax.set_xlim(-0.5, navigator.board.size - 0.5)
        ax.set_ylim(-0.5, navigator.board.size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xticks(range(navigator.board.size))
        ax.set_yticks(range(navigator.board.size))
        letters = [chr(i) for i in range(ord('A'), ord('T')+1) if chr(i) != 'I']
        ax.set_xticklabels(letters[:navigator.board.size])
        ax.set_yticklabels(range(1, navigator.board.size + 1))
        title = f"Movimiento {frame}/{len(navigator.moves)}\nCapturas - Negro: {navigator.board.captures['B']}, Blanco: {navigator.board.captures['W']}"
        ax.set_title(title, fontsize=12, fontweight='bold')

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
    writer = PillowWriter(fps=max(1, 1000//interval), bitrate=1800)
    anim.save(output_path, writer=writer, dpi=80)
    plt.close('all')
    matplotlib.use('module://matplotlib_inline.backend_inline')
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"âœ… GIF creado: {output_path}\n   TamaÃ±o: {size_mb:.2f} MB")
    else:
        print("âŒ Error al crear GIF")


def compare_positions(board1, board2, titles=["PosiciÃ³n 1", "PosiciÃ³n 2"], figsize=(16, 8)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    vis1 = GoBoardVisualizer(board1); vis2 = GoBoardVisualizer(board2)
    plt.sca(axes[0]); axes[0].set_facecolor('#DEB887')
    for i in range(board1.size):
        axes[0].plot([0, board1.size-1], [i, i], 'k-', linewidth=1)
        axes[0].plot([i, i], [0, board1.size-1], 'k-', linewidth=1)
    for r in range(board1.size):
        for c in range(board1.size):
            s = board1.board[r, c]
            if s == 'B': axes[0].add_patch(Circle((c, r), 0.45, facecolor='black', zorder=2))
            elif s == 'W': axes[0].add_patch(Circle((c, r), 0.45, facecolor='white', edgecolor='black', linewidth=1.5, zorder=2))
    axes[0].set_xlim(-0.5, board1.size - 0.5); axes[0].set_ylim(-0.5, board1.size - 0.5)
    axes[0].set_aspect('equal'); axes[0].invert_yaxis(); axes[0].set_title(titles[0], fontsize=14, fontweight='bold')

    plt.sca(axes[1]); axes[1].set_facecolor('#DEB887')
    for i in range(board2.size):
        axes[1].plot([0, board2.size-1], [i, i], 'k-', linewidth=1)
        axes[1].plot([i, i], [0, board2.size-1], 'k-', linewidth=1)
    for r in range(board2.size):
        for c in range(board2.size):
            s = board2.board[r, c]
            if s == 'B': axes[1].add_patch(Circle((c, r), 0.45, facecolor='black', zorder=2))
            elif s == 'W': axes[1].add_patch(Circle((c, r), 0.45, facecolor='white', edgecolor='black', linewidth=1.5, zorder=2))
    axes[1].set_xlim(-0.5, board2.size - 0.5); axes[1].set_ylim(-0.5, board2.size - 0.5)
    axes[1].set_aspect('equal'); axes[1].invert_yaxis(); axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, axes


class BoardEditor:
    """Editor interactivo de tablero vacÃ­o con clicks de ratÃ³n.

    - Permite colocar piedras B/W con validaciÃ³n de reglas (capturas, ko, suicidio).
    - Controles: Color actual, Pass, Undo, Clear, Vista de EnergÃ­a.
    - No requiere SGF ni matrices; usa GoBoard como fuente de verdad.

    Uso en notebook:
        from src.go_visualization import BoardEditor
        editor = BoardEditor(board_size=19)
        display(editor.view())
    """

    def __init__(self, board_size: int = 19, figsize=(8, 8), energy_backend: str = 'mpl', show_log: bool = False, energy_tabs=None):
        from src.go_game_engine import GoBoard
        self.board_size = int(board_size)
        self.GoBoard = GoBoard
        self.board = GoBoard(size=self.board_size)
        self._figsize = figsize
        self._energy_backend = energy_backend
        self._show_log = bool(show_log)
        # Seleccin opcional de pestaas para "Ver Energa"
        self._energy_tabs = tuple(t.lower() for t in energy_tabs) if energy_tabs else None

        # Asegurar pestaas por defecto (M2-Quantum) si no se especificaron
        if getattr(self, '_energy_tabs', None) is None:
            if energy_tabs is None:
                energy_tabs = ['m2-quantum']
            self._energy_tabs = tuple(t.lower() for t in energy_tabs)

        # Widgets y salidas
        self._out_board = Output(layout=Layout(min_width='420px', flex='1 1 auto'))
        self._out_log = Output()
        self._out_energy_plot = Output(layout=Layout(min_width='360px', flex='1 1 auto'))
        self._energy_hist = {'white': [], 'black': [], 'adv': [], 'adv_norm': []}
        self._energy_tabs_box = VBox()
        # Salidas para pestaas de energa
        self._out_m1_q = Output(); self._out_m1_c = Output(); self._out_m2_q = Output(); self._out_m2_c = Output()
        self._color_picker = ToggleButtons(
            options=[('Negro', 'B'), ('Blanco', 'W')],
            value='B',
            description='Color:',
            style={'description_width': 'initial'}
        )
        self._auto_toggle = Checkbox(value=True, description='Auto alternar')
        self._btn_pass = Button(description='Pass', button_style='')
        self._btn_undo = Button(description='Undo', button_style='warning')
        self._btn_clear = Button(description='Clear', button_style='danger')
        self._btn_energy = Button(description='Ver EnergÃ­a', button_style='info')

        # Estado de figura para manejar clicks
        self._fig = None
        self._ax = None
        self._cid = None  # connection id de mpl_connect

    # ---------- Render principal ---------- #
    def _render_board(self, *, show_liberties: bool = False, show_numbers: bool = False):
        with self._out_board:
            clear_output(wait=True)
            # Aviso si el backend no es interactivo (no habr clicks)
            try:
                import matplotlib as _mpl
                _backend = str(_mpl.get_backend()).lower()
                if 'inline' in _backend:
                    with self._out_log:
                        print("[Aviso] Backend Matplotlib='inline' no soporta clicks.")
                        print("        En el notebook, ejecuta: %pip install ipympl  (si hace falta)")
                        print("        y luego: %matplotlib widget")
            except Exception:
                pass
            vis = GoBoardVisualizer(self.board)
            fig, ax = vis.plot_matplotlib(
                title=f"Editor | Movimiento {len(self.board.move_history)}",
                show_liberties=show_liberties,
                show_captures=False,
                show_move_numbers=show_numbers,
                figsize=self._figsize,
            )
            # Conectar manejador de clicks
            if self._cid is not None and self._fig is not None:
                try:
                    self._fig.canvas.mpl_disconnect(self._cid)
                except Exception:
                    pass
            self._fig, self._ax = fig, ax
            self._cid = fig.canvas.mpl_connect('button_press_event', self._on_click)
            plt.show()

    def _log(self, msg: str):
        if not getattr(self, '_show_log', False):
            return
        with self._out_log:
            clear_output(wait=True)
            print(msg)

    def _on_click(self, event):
        # Ignorar clicks fuera del rea del tablero
        if event.inaxes is None or self._ax is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Snap a la interseccin ms cercana
        c = int(round(event.xdata))
        r = int(round(event.ydata))
        if not (0 <= r < self.board.size and 0 <= c < self.board.size):
            return
        color = self._color_picker.value
        ok, msg = self.board.place_stone(color, (r, c))
        if not ok:
            self._log(f"âŒ {msg}")
        else:
            self._log(f"âœ… {color} en {(r, c)} â€” {msg}")
        # Re-render
        self._render_board()
        # Alternar color para el siguiente click (si est activo)
        try:
            if ok and self._auto_toggle.value:
                self._color_picker.value = 'W' if color == 'B' else 'B'
        except Exception:
            pass
        # Resumen de energa global (M2, quantum) acorde a tu convencin
        try:
            self._energy_summary(dist=2, method='quantum')
        except Exception:
            pass
        # Actualizar panel grfico
        try:
            self._update_energy_plot()
        except Exception:
            pass
        
        
    def _do_pass(self, _):
        # En Go, pass no cambia tablero; aqu solo registramos en historial para consistencia si se desea.
        # Mantendremos simple: loguear el pass.
        self._log(f"â­ï¸ Pass de {self._color_picker.value}")

    def _do_undo(self, _):
        # Llama al mtodo correcto del motor (undo_move)
        if hasattr(self.board, 'undo_move') and self.board.undo_move():
            self._log("â†©ï¸ Deshacer: OK")
            self._render_board()
            try:
                self._energy_summary(dist=2, method='quantum')
                self._update_energy_plot()
            except Exception:
                pass
        else:
            self._log("âš ï¸ No hay movimientos que deshacer")

    def _do_clear(self, _):
        self.board = self.GoBoard(size=self.board_size)
        self._log("ðŸ§¹ Tablero reiniciado")
        self._render_board()
        try:
            self._energy_hist = {'white': [], 'black': [], 'adv': [], 'adv_norm': []}
            self._update_energy_plot()
        except Exception:
            pass

    def _do_energy(self, _):
        # Render a pestaas de energa (M1/M2  Q/C) usando backend seleccionado
        try:
            from src.go_energy_viz import _compute_energy_map as _cem, _figure_for_energy as _bokeh_fig
            board_np = np.array(self.board.board, dtype=str)
            # Preparar tabs de energa en el contenedor inferior
            tabs = Tab(children=[self._out_m1_q, self._out_m1_c, self._out_m2_q, self._out_m2_c])
            try:
                tabs.set_title(0, 'M1-Q'); tabs.set_title(1, 'M1-C'); tabs.set_title(2, 'M2-Q'); tabs.set_title(3, 'M2-C')
            except Exception:
                pass
            self._energy_tabs_box.children = [tabs]
            # Filtrar pestaas visibles si se especific self._energy_tabs
            try:
                allowed = set(self._energy_tabs) if self._energy_tabs is not None else {
                    'm1-quantum','m1-classical','m2-quantum','m2-classical'
                }
                child_map = [
                    ('m1-quantum',   self._out_m1_q, 'M1-Q'),
                    ('m1-classical', self._out_m1_c, 'M1-C'),
                    ('m2-quantum',   self._out_m2_q, 'M2-Q'),
                    ('m2-classical', self._out_m2_c, 'M2-C'),
                ]
                filtered_children = [w for key, w, _ in child_map if key in allowed]
                if filtered_children:
                    tabs.children = filtered_children
                    for i, (key, _, title) in enumerate([t for t in child_map if t[0] in allowed]):
                        try: tabs.set_title(i, title)
                        except Exception: pass
                else:
                    tabs.children = []
            except Exception:
                pass
            if self._energy_backend == 'bokeh':
                from bokeh.embed import file_html as _bokeh_file_html
                from bokeh.resources import INLINE as _BK_INLINE
                from IPython.display import HTML as _IPY_HTML, display as _ipy_display
                def _render(out, dist, method, title):
                    with out:
                        clear_output(wait=True)
                        fig = _bokeh_fig(board_np, _cem(board_np, dist, method), title)
                        html = _bokeh_file_html(fig, _BK_INLINE, title)
                        _ipy_display(_IPY_HTML(html))
                allowed = set(self._energy_tabs) if self._energy_tabs is not None else {
                    'm1-quantum','m1-classical','m2-quantum','m2-classical'
                }
                if 'm1-quantum' in allowed:
                    _render(self._out_m1_q, 1, 'quantum',  'M1 | Quantum')
                if 'm1-classical' in allowed:
                    _render(self._out_m1_c, 1, 'classical','M1 | Classical')
                if 'm2-quantum' in allowed:
                    _render(self._out_m2_q, 2, 'quantum',  'M2 | Quantum')
                if 'm2-classical' in allowed:
                    _render(self._out_m2_c, 2, 'classical','M2 | Classical')
            else:
                import matplotlib.pyplot as _plt
                def _render(out, dist, method, title):
                    with out:
                        clear_output(wait=True)
                        emap = _cem(board_np, dist, method)
                        v = float(max(abs(np.min(emap)), abs(np.max(emap)))) or 1.0
                        vis = GoBoardVisualizer(self.board)
                        fig, ax = vis.plot_matplotlib(title=title, show_liberties=False, show_move_numbers=False, figsize=(5, 5))
                        ax.imshow(emap, cmap='coolwarm', vmin=-v, vmax=v, origin='upper', alpha=0.45, zorder=1)
                        _plt.tight_layout(); _plt.show()
                allowed = set(self._energy_tabs) if self._energy_tabs is not None else {
                    'm1-quantum','m1-classical','m2-quantum','m2-classical'
                }
                if 'm1-quantum' in allowed:
                    _render(self._out_m1_q, 1, 'quantum',  'M1 | Quantum')
                if 'm1-classical' in allowed:
                    _render(self._out_m1_c, 1, 'classical','M1 | Classical')
                if 'm2-quantum' in allowed:
                    _render(self._out_m2_q, 2, 'quantum',  'M2 | Quantum')
                if 'm2-classical' in allowed:
                    _render(self._out_m2_c, 2, 'classical','M2 | Classical')
        except Exception as e:
            with self._out_m1_q:
                clear_output(wait=True)
                print('[Aviso] No se pudo renderizar energÃ­a:', e)

    def _energy_summary(self, dist: int = 2, method: str = 'quantum'):
        """Resumen de energÃ­a por color en todo el tablero.

        ConvenciÃ³n: energÃ­as positivas se suman para Blanco; energÃ­as negativas (mÃ³dulo) para Negro,
        independientemente de piezas presentes (influencia en vacÃ­os incluida).
        """
        try:
            from src.go_energy_viz import _compute_energy_map as _cem
            board_np = np.array(self.board.board, dtype=str)
            emap = _cem(board_np, dist, method)
            if emap.size == 0:
                return
            pos = float(np.sum(emap[emap > 0.0])) if np.any(emap > 0) else 0.0
            neg = float(np.sum(emap[emap < 0.0])) if np.any(emap < 0) else 0.0
            e_white = pos
            e_black = -neg
            advantage = e_white - e_black
            self._log(f"EnergÃ­a (M{dist}-{method}): Blanco={e_white:.3f} | Negro={e_black:.3f} | Ventaja(W-B)={advantage:.3f}")
        except Exception as e:
            if method != 'classical':
                try:
                    self._energy_summary(dist=dist, method='classical')
                    self._log(f"[Aviso] Resumen con 'classical' por fallo en '{method}': {e}")
                except Exception as e2:
                    self._log(f"[Aviso] No se pudo calcular resumen de energÃ­a: {e2}")
            else:
                self._log(f"[Aviso] No se pudo calcular resumen de energÃ­a: {e}")

    def _update_energy_plot(self):
        """Panel: barras BRUTAS (W/B) + ventaja NORMALIZADA (W-B)."""
        from src.go_energy_viz import _compute_energy_map as _cem
        board_np = np.array(self.board.board, dtype=str)
        try:
            emap = _cem(board_np, 2, 'quantum')
        except Exception:
            emap = _cem(board_np, 2, 'classical')
        if emap.size:
            pos = float(np.sum(emap[emap > 0.0])) if np.any(emap > 0) else 0.0
            neg = float(np.sum(emap[emap < 0.0])) if np.any(emap < 0) else 0.0
            e_white, e_black = pos, -neg
        else:
            e_white = e_black = 0.0
        eps = 1e-9
        total = max(e_white + e_black, eps)
        adv_raw  = e_white - e_black
        adv_norm = adv_raw / total
        # historial
        if not hasattr(self, '_energy_hist'):
            self._energy_hist = {'white': [], 'black': [], 'adv': [], 'adv_norm': []}
        H = self._energy_hist
        if 'adv_norm' not in H: H['adv_norm'] = []
        if 'white' not in H:    H['white']    = []
        if 'black' not in H:    H['black']    = []
        if 'adv' not in H:      H['adv']      = []
        H['white'].append(e_white); H['black'].append(e_black)
        H['adv'].append(adv_raw);  H['adv_norm'].append(adv_norm)

        with self._out_energy_plot:
            clear_output(wait=True)
            # Dos grficos en columna para ahorrar ancho
            fig, axes = plt.subplots(2, 1, figsize=(3.5, 5))

            # Barras BRUTAS + porcentaje (arriba)
            share_w = e_white/total; share_b = e_black/total
            vmax = max(e_white, e_black, 1.0)
            axes[0].bar(['Blanco','Negro'], [e_white, e_black], color=['#e5e7eb', '#111827'])
            axes[0].set_ylim(0, vmax*1.15)
            for i, v in enumerate([e_white, e_black]):
                pct = [share_w, share_b][i]*100
                axes[0].text(i, v + vmax*0.03, f"{v:.1f}\n({pct:.1f}%)",
                             ha='center', va='bottom', fontsize=9, color='#374151')
            axes[0].set_title('Energia BRUTA (M2)')
            axes[0].set_ylabel('Suma de energia')
            axes[0].ticklabel_format(style='plain', axis='y')

            # Ventaja NORMALIZADA [-1,1] (abajo)
            x = np.arange(1, len(H['adv_norm'])+1, dtype=int)
            y = np.array(H['adv_norm'], dtype=float)
            if len(x) == 1:
                axes[1].plot(x, y, color='#2563eb', lw=1.5, marker='o')
            else:
                axes[1].plot(x, y, color='#2563eb', lw=1.5)
            axes[1].fill_between(x, 0, y, where=(y>=0), color='#60a5fa', alpha=0.35)
            axes[1].fill_between(x, 0, y, where=(y<0), color='#f87171', alpha=0.35)
            axes[1].axhline(0, color='gray', lw=1)
            axes[1].set_ylim(-1.05, 1.05)
            axes[1].set_xlim(0.5, max(1.5, len(x)+0.5))
            axes[1].set_xticks(x)
            axes[1].set_title('Ventaja NORMALIZADA (W-B)')
            axes[1].set_xlabel('Movimiento')
            axes[1].set_ylabel('Delta energia / total')
            axes[1].ticklabel_format(style='plain', axis='y')
            fig.tight_layout()
            plt.show()

    def view(self):
        # Conectar botones
        self._btn_pass.on_click(self._do_pass)
        self._btn_undo.on_click(self._do_undo)
        self._btn_clear.on_click(self._do_clear)
        self._btn_energy.on_click(self._do_energy)

        controls = HBox([
            self._color_picker,
            self._auto_toggle,
            self._btn_pass,
            self._btn_undo,
            self._btn_clear,
            self._btn_energy,
        ])
        # Layout: tablero + panel de energa lado a lado
        main_row = HBox([self._out_board, self._out_energy_plot],
                        layout=Layout(justify_content='space-between', align_items='flex-start', gap='16px'))
        # Render inicial
        self._render_board()
        try:
            self._update_energy_plot()
        except Exception:
            pass
        children = [controls, main_row, self._energy_tabs_box]
        if getattr(self, '_show_log', False):
            children.append(self._out_log)
        return VBox(children)


