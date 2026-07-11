# -*- coding: utf-8 -*-
"""Genera las láminas SVG comparativas (mapa cúbico vs cuadrático/Atomic-Go)
para el README de Proyect-Quantum-Go. Convención del código (STONE_TO_SPIN): Negro=-1, Blanco=+1, Vacío=0.
"""
import math, os

OUT = r"c:\Users\ometi\Documents\IA\Github\Ising-go\Proyect-Quantum-Go\data\assets"
FONT = "Segoe UI, Arial, sans-serif"

def E_cub(a, b):  return a + 2*b - a*b*b - a*a*b
def E_quad(a, b): return -(a*b)   # J = 1

NAMES = {1: 'Blanco', 0: 'Vacío', -1: 'Negro'}
ORDER = [-1, 0, 1]
PAIRS = [(a, b) for a in ORDER for b in ORDER]

def fmt(v):
    if v == 0: return "0"
    return ("+" if v > 0 else "−") + str(abs(v))

# ---- escalas de color -------------------------------------------------------
def style_cub(v):
    """(fill, stroke, text) — inclinación de color (oscuro=negro, ámbar=blanco), NO energética."""
    if v == 0:  return ('#e7e2d5', '#b3ab97', '#6b6455')
    if v <= -2: return ('#0f172a', '#0f172a', '#ffffff')
    if v == -1: return ('#475569', '#475569', '#ffffff')
    if v == 1:  return ('#d97706', '#d97706', '#ffffff')
    return ('#92400e', '#92400e', '#ffffff')          # v >= +2

def style_quad(v):
    """(fill, stroke, text) — escala energética (atracción/repulsión)."""
    if v == 0: return ('#e7e2d5', '#b3ab97', '#6b6455')
    if v > 0:  return ('#fee2e2', '#dc2626', '#b91c1c')
    return ('#dbeafe', '#2563eb', '#1d4ed8')

def edge_color_cub(v):
    return {-2: '#0f172a', -1: '#475569', 0: '#c9c2b0', 1: '#d97706', 2: '#92400e'}[v]

# ---- primitivas -------------------------------------------------------------
def chip(cx, cy, v, style, w=44, h=26, fs=14):
    fill, stroke, txt = style(v)
    return (f'<rect x="{cx-w/2}" y="{cy-h/2}" width="{w}" height="{h}" rx="7" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'<text x="{cx}" y="{cy+fs*0.36}" text-anchor="middle" font-size="{fs}" '
            f'font-weight="700" fill="{txt}" font-family="{FONT}">{fmt(v)}</text>')

def stone(cx, cy, s, r=9):
    if s == -1:
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#111827"/>'
    if s == 1:
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#111827" stroke-width="1.6"/>'
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#efe9dc" stroke="#c9bfa5" stroke-width="1.2"/>'
            f'<circle cx="{cx}" cy="{cy}" r="1.8" fill="#8d846d"/>')

def text(x, y, s, size=12, weight=400, fill='#374151', anchor='middle', style=''):
    st = f' font-style="{style}"' if style else ''
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" font-family="{FONT}"{st}>{s}</text>')

def badge(cx, cy, label, color, fill, w=210):
    return (f'<rect x="{cx-w/2}" y="{cy-12}" width="{w}" height="24" rx="12" '
            f'fill="{fill}" stroke="{color}" stroke-width="1.4"/>'
            + text(cx, cy+4.5, label, 12, 700, color))

NEQ = lambda cx, cy: (f'<circle cx="{cx}" cy="{cy}" r="10" fill="#f5f3ff" stroke="#7c3aed" stroke-width="1.6"/>'
                      + text(cx, cy+4.5, '≠', 13, 700, '#7c3aed'))
EQ  = lambda cx, cy: text(cx, cy+4.5, '=', 13, 600, '#b3ab97')

# ============================================================================
# LÁMINA 1 — TABLA DE INTERACCIÓN COMPARADA
# ============================================================================
def table_panel(x0, y0, title, sub, subcolor, subfill, formula, symline, symcolor,
                Efn, style, legend_rows, note):
    W, H = 575, 624
    s = [f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" rx="14" fill="#ffffff" '
         f'stroke="#e2ddd0" stroke-width="1.5"/>']
    cx = x0 + W/2
    s.append(text(cx, y0+32, title, 17, 700, '#1f2937'))
    s.append(badge(cx, y0+58, sub, subcolor, subfill))
    s.append(text(cx, y0+88, formula, 13.5, 500, '#4b5563', style='italic'))
    s.append(text(cx, y0+110, symline, 11.5, 700, symcolor))
    # encabezados de columna
    c_s0, c_s1, c_par, c_e1, c_e2, c_sym = x0+36, x0+76, x0+106, x0+330, x0+425, x0+510
    hy = y0 + 138
    s.append(text(c_s0, hy, 's₀', 12, 700, '#6b7280'))
    s.append(text(c_s1, hy, 's₁', 12, 700, '#6b7280'))
    s.append(text(c_par, hy, 'Par', 12, 700, '#6b7280', 'start'))
    s.append(text(c_e1, hy, 'E(i→j)', 12, 700, '#6b7280'))
    s.append(text(c_e2, hy, 'E(j→i)', 12, 700, '#6b7280'))
    s.append(f'<line x1="{x0+18}" y1="{y0+148}" x2="{x0+W-18}" y2="{y0+148}" stroke="#e2ddd0" stroke-width="1.2"/>')
    # filas
    for i, (a, b) in enumerate(PAIRS):
        ry = y0 + 152 + i*40
        cy = ry + 20
        if i % 2 == 0:
            s.append(f'<rect x="{x0+12}" y="{ry}" width="{W-24}" height="40" fill="#f6f3ea"/>')
        s.append(stone(c_s0, cy, a)); s.append(stone(c_s1, cy, b))
        s.append(text(c_par, cy+4.5, f'{NAMES[a]} — {NAMES[b]}', 12.5, 500, '#374151', 'start'))
        e1, e2 = Efn(a, b), Efn(b, a)
        s.append(chip(c_e1, cy, e1, style)); s.append(chip(c_e2, cy, e2, style))
        s.append(NEQ(c_sym, cy) if e1 != e2 else EQ(c_sym, cy))
    # leyenda
    ly = y0 + 152 + 9*40 + 22
    for row in legend_rows:
        n = len(row); seg = (W-40) / n
        for k, (v, lbl) in enumerate(row):
            icx = x0 + 20 + seg*k + 26
            if v == 'neq':
                s.append(NEQ(icx, ly)); s.append(text(icx+18, ly+4.5, lbl, 11, 500, '#4b5563', 'start'))
            else:
                s.append(chip(icx, ly, v, style, w=38, h=22, fs=12))
                s.append(text(icx+26, ly+4.5, lbl, 11, 500, '#4b5563', 'start'))
        ly += 30
    s.append(text(cx, ly+6, note, 11.5, 700, symcolor))
    return ''.join(s)

def make_table_svg():
    Wt, Ht = 1200, 700
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wt} {Ht}" font-family="{FONT}">',
         f'<rect width="{Wt}" height="{Ht}" rx="16" fill="#fbf9f3"/>']
    s.append(text(Wt/2, 26, 'Convención del código:  Negro = −1 · Blanco = +1 · Vacío = 0',
                  13, 700, '#6b7280'))
    s.append(table_panel(
        16, 44,
        'Mapa cúbico — nuestro modelo (M1)',
        'modelo impar · grado 3', '#2563eb', '#eff6ff',
        'E(s₀, s₁) = s₀ + 2s₁ − s₀s₁² − s₀²s₁',
        'ASIMÉTRICO:  E(i→j) ≠ E(j→i)', '#7c3aed',
        E_cub, style_cub,
        [[(-2, 'fuerte hacia negro'), (-1, 'hacia negro'), (0, 'neutro')],
         [(1, 'hacia blanco'), (2, 'fuerte hacia blanco'), ('neq', 'par asimétrico')]],
        '⚠ El signo indica inclinación de color, no estabilidad.'))
    s.append(table_panel(
        609, 44,
        'Mapa cuadrático — Atomic-Go (Alvarado)',
        'modelo par · grado 2', '#ea580c', '#fff7ed',
        'E(s₀, s₁) = −J·s₀·s₁ ,   J = 1',
        'SIMÉTRICO:  E(i→j) = E(j→i)', '#16a34a',
        E_quad, style_quad,
        [[(-1, 'conexión favorecida'), (0, 'sin señal'), (1, 'contacto penalizado')]],
        'El signo sí es energético: negativo = configuración favorecida.'))
    s.append('</svg>')
    return ''.join(s)

# ============================================================================
# LÁMINA 2 — GRAFO DE INTERACCIÓN
# ============================================================================
def trim_pt(p, q, d):
    dx, dy = q[0]-p[0], q[1]-p[1]
    L = math.hypot(dx, dy)
    return (p[0] + dx/L*d, p[1] + dy/L*d)

def arrow(p0, p1, bow, v, style, r=30, marker_pref=''):
    col = edge_color_cub(v)
    mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    L = math.hypot(dx, dy)
    nx, ny = -dy/L, dx/L
    c = (mx + nx*bow, my + ny*bow)
    S = trim_pt(p0, c, r+6)
    E = trim_pt(p1, c, r+13)
    mid = (0.25*S[0] + 0.5*c[0] + 0.25*E[0], 0.25*S[1] + 0.5*c[1] + 0.25*E[1])
    w = 2 + abs(v)
    path = (f'<path d="M {S[0]:.1f} {S[1]:.1f} Q {c[0]:.1f} {c[1]:.1f} {E[0]:.1f} {E[1]:.1f}" '
            f'fill="none" stroke="{col}" stroke-width="{w}" marker-end="url(#{marker_pref}m{col.lstrip("#")})"/>')
    return path + chip(mid[0], mid[1], v, style, w=38, h=22, fs=12)

def loop(cx, cy, v, style, below=False, marker_pref=''):
    col = edge_color_cub(v) if style is style_cub else style(v)[1]
    sgn = 1 if below else -1
    y1, y2 = cy + sgn*27, cy + sgn*80
    path = (f'<path d="M {cx-13} {y1} C {cx-46} {y2} {cx+46} {y2} {cx+13} {y1}" '
            f'fill="none" stroke="{col}" stroke-width="{2+abs(v)}" '
            f'marker-end="url(#{marker_pref}m{col.lstrip("#")})"/>')
    return path + chip(cx, cy + sgn*72, v, style, w=38, h=22, fs=12)

def node(cx, cy, s, label, lx=None, ly=None, anchor='middle'):
    r = 30
    if s == -1:
        g = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#111827"/><circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="#ffffff" stroke-width="2"/>'
    elif s == 1:
        g = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#111827" stroke-width="2"/><circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="#111827" stroke-width="2"/>'
    else:
        g = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#efe9dc" stroke="#b3ab97" stroke-width="1.6"/><circle cx="{cx}" cy="{cy}" r="3" fill="#8d846d"/>'
    return g + text(lx or cx, ly or cy + r + 20, label, 12.5, 600, '#374151', anchor)

def make_graph_svg():
    Wt, Ht = 1200, 620
    colors = ['#0f172a', '#475569', '#c9c2b0', '#d97706', '#92400e', '#2563eb', '#dc2626', '#b3ab97']
    defs = ''.join(
        f'<marker id="m{c.lstrip("#")}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" '
        f'markerHeight="6.5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
        for c in colors)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wt} {Ht}" font-family="{FONT}">',
         f'<defs>{defs}</defs>',
         f'<rect width="{Wt}" height="{Ht}" rx="16" fill="#fbf9f3"/>']
    s.append(text(Wt/2, 26, 'Convención del código:  Negro = −1 · Blanco = +1 · Vacío = 0',
                  13, 700, '#6b7280'))

    # ---------- panel izquierdo: cúbico ----------
    x0, y0, W, H = 16, 44, 575, 540
    s.append(f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" rx="14" fill="#ffffff" stroke="#e2ddd0" stroke-width="1.5"/>')
    cx = x0 + W/2
    s.append(text(cx, y0+30, 'Mapa cúbico — nuestro modelo (M1)', 16, 700, '#1f2937'))
    s.append(text(cx, y0+52, 'E(s₀, s₁) = s₀ + 2s₁ − s₀s₁² − s₀²s₁', 12.5, 500, '#4b5563', style='italic'))
    s.append(text(cx, y0+72, 'Flecha i→j :  E(centro = i, vecino = j)', 11.5, 700, '#7c3aed'))
    N = (x0+150, y0+205); B = (x0+425, y0+205); V = (x0+287, y0+400)
    # aristas dirigidas (dos por par, arcos opuestos)
    s.append(arrow(N, B, 30, E_cub(-1,  1), style_cub))   # N->B = +1
    s.append(arrow(B, N, 30, E_cub( 1, -1), style_cub))   # B->N = -1
    s.append(arrow(N, V, 28, E_cub(-1,  0), style_cub))   # N->V = -1
    s.append(arrow(V, N, 28, E_cub( 0, -1), style_cub))   # V->N = -2
    s.append(arrow(B, V, 28, E_cub( 1,  0), style_cub))   # B->V = +1
    s.append(arrow(V, B, 28, E_cub( 0,  1), style_cub))   # V->B = +2
    # bucles (auto-interacción del par igual-igual)
    s.append(loop(N[0], N[1],  E_cub(-1, -1), style_cub))            # -1
    s.append(loop(B[0], B[1],  E_cub( 1,  1), style_cub))            # +1
    s.append(loop(V[0], V[1],  E_cub( 0,  0), style_cub, below=True))# 0
    s.append(node(N[0], N[1], -1, 'Negro (−1)', lx=N[0]-40, ly=N[1]+5, anchor='end'))
    s.append(node(B[0], B[1],  1, 'Blanco (+1)', lx=B[0]+40, ly=B[1]+5, anchor='start'))
    s.append(node(V[0], V[1],  0, 'Vacío (0)', lx=V[0]+44, ly=V[1]+5, anchor='start'))
    # mini-leyenda
    ly = y0 + H - 28
    items = [(-2, ''), (-1, ''), (0, ''), (1, ''), (2, '')]
    lx0 = x0 + 120
    s.append(text(lx0-10, ly+4.5, 'hacia negro', 11, 600, '#0f172a', 'end'))
    for k, (v, _) in enumerate(items):
        s.append(chip(lx0 + 30 + k*48, ly, v, style_cub, w=38, h=22, fs=12))
    s.append(text(lx0 + 30 + 5*48 - 14, ly+4.5, 'hacia blanco', 11, 600, '#92400e', 'start'))

    # ---------- panel derecho: cuadrático ----------
    x0 = 609
    s.append(f'<rect x="{x0}" y="{y0}" width="{W}" height="{H}" rx="14" fill="#ffffff" stroke="#e2ddd0" stroke-width="1.5"/>')
    cx = x0 + W/2
    s.append(text(cx, y0+30, 'Mapa cuadrático — Atomic-Go (Alvarado)', 16, 700, '#1f2937'))
    s.append(text(cx, y0+52, 'E(s₀, s₁) = −J·s₀·s₁ ,   J = 1', 12.5, 500, '#4b5563', style='italic'))
    s.append(text(cx, y0+72, 'Aristas sin flecha: la interacción es simétrica', 11.5, 700, '#16a34a'))
    N = (x0+150, y0+205); B = (x0+425, y0+205); V = (x0+287, y0+400)
    # N—B: +1 (repulsión)
    s.append(f'<line x1="{N[0]+32}" y1="{N[1]}" x2="{B[0]-32}" y2="{B[1]}" stroke="#dc2626" stroke-width="3.5"/>')
    s.append(chip((N[0]+B[0])/2, N[1], 1, style_quad, w=38, h=22, fs=12))
    # V—N y V—B: 0 (sin señal, tenue)
    for P in (N, B):
        S = trim_pt(V, P, 32); E2 = trim_pt(P, V, 32)
        s.append(f'<line x1="{S[0]:.1f}" y1="{S[1]:.1f}" x2="{E2[0]:.1f}" y2="{E2[1]:.1f}" '
                 f'stroke="#d9d3c4" stroke-width="1.6" stroke-dasharray="5 5"/>')
        mid = ((S[0]+E2[0])/2, (S[1]+E2[1])/2)
        s.append(chip(mid[0], mid[1], 0, style_quad, w=32, h=20, fs=11))
    # bucles: N,B = -1 (conexión propia favorecida); V = 0
    def qloop(cx_, cy_, v, below=False):
        col = style_quad(v)[1]
        sgn = 1 if below else -1
        y1, y2 = cy_ + sgn*27, cy_ + sgn*80
        return (f'<path d="M {cx_-13} {y1} C {cx_-46} {y2} {cx_+46} {y2} {cx_+13} {y1}" '
                f'fill="none" stroke="{col}" stroke-width="{2+abs(v)}"/>'
                + chip(cx_, cy_ + sgn*72, v, style_quad, w=38, h=22, fs=12))
    s.append(qloop(N[0], N[1], -1)); s.append(qloop(B[0], B[1], -1)); s.append(qloop(V[0], V[1], 0, below=True))
    s.append(node(N[0], N[1], -1, 'Negro (−1)', lx=N[0]-40, ly=N[1]+5, anchor='end'))
    s.append(node(B[0], B[1],  1, 'Blanco (+1)', lx=B[0]+40, ly=B[1]+5, anchor='start'))
    s.append(node(V[0], V[1],  0, 'Vacío (0)', lx=V[0]+44, ly=V[1]+5, anchor='start'))
    ly = y0 + H - 28
    for k, (v, lbl) in enumerate([(-1, 'conexión'), (0, 'sin señal'), (1, 'contacto')]):
        icx = x0 + 90 + k*160
        s.append(chip(icx, ly, v, style_quad, w=38, h=22, fs=12))
        s.append(text(icx+26, ly+4.5, lbl, 11, 500, '#4b5563', 'start'))

    s.append('</svg>')
    return ''.join(s)

# ============================================================================
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, svg in [('comparacion_tabla_m1.svg', make_table_svg()),
                      ('comparacion_grafo_m1.svg', make_graph_svg())]:
        p = os.path.join(OUT, name)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(svg)
        print(p, len(svg), 'bytes')
    # verificación rápida de valores
    print('\nVerificación E_cub (Negro=+1):')
    for a, b in PAIRS:
        print(f'  {NAMES[a]:7s}-{NAMES[b]:7s}: E(i,j)={E_cub(a,b):+d}  E(j,i)={E_cub(b,a):+d}  quad={E_quad(a,b):+d}')
