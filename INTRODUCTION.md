# 🎮⚛️ Quantum Go: Enfoques Adiabáticos y Fotónicos para la Estrategia

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![D-Wave](https://img.shields.io/badge/D--Wave-Ocean%20SDK-00ADD8)](https://ocean.dwavesys.com/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Xanadu-green)](https://pennylane.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> Explorando enfoques de computación cuántica para el milenario juego de Go utilizando recocido cuántico de D-Wave y computación fotónica de Xanadu.

<p align="center">
  <img src="data/assets/Interfaz_1.png" alt="Quantum Go Banner" width="800"/>
</p>

---

## 🌟 ¿Qué es esto?

Este proyecto investiga dos paradigmas de computación cuántica radicalmente diferentes para analizar el juego de Go:

- 🔷 D-Wave (Recocido Cuántico): mapea posiciones de Go a Hamiltonianos de Ising, encontrando jugadas óptimas mediante evolución adiabática y efecto túnel cuántico.
- 🔶 Xanadu (Computación Fotónica): representa el tablero como grafos (Grafos de Destino Común, CFG), extrayendo características estratégicas mediante Gaussian Boson Sampling (GBS).

Ambos enfoques se complementan: D-Wave optimiza posiciones; Xanadu extrae características cuánticas para aprendizaje automático.

---

## 🚀 Inicio rápido

### Instalación
```bash
pip install -e .
pip install -r requirements.txt
```

### Demo básica
```python
from src.go_game_engine import SGFParser
from src.go_visualization import GameNavigator

parser = SGFParser()
moves, info = parser.parse_file("data/sgf partidas/archivo.sgf")

navigator = GameNavigator(moves)
ui = navigator.create_view(figsize=(6, 6), include_energy_tabs=True, energy_backend='bokeh')
ui
```

### Modelos de energía (ejemplo)
```python
import numpy as np
from src.go_energy_viz import build_energy_tabs

# Tablero como matriz numpy (ejemplo 9x9 con vacíos '.')
board_np = np.full((9, 9), '.', dtype=str)

tabs = build_energy_tabs(board_np)
tabs  # mostrar en Jupyter (backend bokeh)
```

---

## 📚 Documentación

- 📘 Introducción y teoría: este archivo (`INTRODUCTION.md`)
- 🧭 Guía de análisis interactivo: `docs/analisis_interactivo_partidas_go.md`
- 📓 Notebooks de ejemplo: `notebooks/`

---

## 🎯 Características clave

- Motor clásico de Go: reglas, validación de jugadas (Ko, suicidio, capturas), grupos y libertades.
- Visualización interactiva: matplotlib/bokeh + widgets para navegación.
- Optimización cuántica (D-Wave): formulación Ising/QUBO y análisis energético.
- Extracción de características (Xanadu): grafos (CFG) y mapas cuánticos.

---

## 🎓 Fundamento teórico

1) Alvarado et al. (2019): "Modeling the Game of Go by Ising Hamiltonian, Deep Belief Networks and Common Fate Graphs". Modelo Atomic-Go: H = -∑_ij w_ij x_i x_j - μ ∑_i h_i x_i

2) Graepel et al. (2001): Grafos de Destino Común para representación en Go.

3) Recocido cuántico (D-Wave): uso de efecto túnel para escapar mínimos locales.

4) Gaussian Boson Sampling (Xanadu): ventaja cuántica en subgrafos densos.

---

## 📄 Licencia

Código bajo MIT (ver `LICENSE`).

---

## 👥 Autores y Contacto

- Autor: Dr. Mario Alberto Mercado Sánchez — ometitlan@gmail.com
- Colaborador (Matemático): Leonardo Jiménez — leonsinmiedo@gmail.com
- Repositorio: https://github.com/ometitlan/Project-Quantum-Go
