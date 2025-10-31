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

## 🎯 Introducción al Proyecto: Computación Cuántica para el Análisis Estratégico de Go

¿Qué significa usar una computadora cuántica para resolver un problema? La respuesta depende de qué fenómenos cuánticos están disponibles y cómo los explota la arquitectura del hardware. Este proyecto surgió de la oportunidad de investigar dos paradigmas radicalmente distintos de computación cuántica: D-Wave (quantum annealing) y Xanadu (photonic quantum computing), aplicados al modelado del juego de Go mediante Hamiltonianos de Ising y representaciones en grafos.

D-Wave representa el enfoque de optimización cuántica adiabática. Su arquitectura física —una red de flux qubits superconductores con conectividad limitada (topologías Chimera, Pegasus o Zephyr)— implementa naturalmente el modelo de Ising. El flujo de trabajo consiste en: (1) formular el problema como Hamiltoniano de Ising, (2) transformarlo a QUBO (Quadratic Unconstrained Binary Optimization), (3) realizar embedding en la topología del chip, y (4) ejecutar evolución adiabática donde el quantum tunneling permite escapar de mínimos locales para encontrar configuraciones de baja energía. Este enfoque es ideal para el modelo Atomic-Go propuesto por Alvarado et al., donde las interacciones entre piedras adyacentes se mapean directamente a términos w_ij x_i x_j del Hamiltoniano de Ising, permitiendo optimizar posiciones del tablero mediante minimización de energía.

Por otro lado, Xanadu explora la computación cuántica fotónica, utilizando estados squeezed de luz y redes de beam splitters para realizar Gaussian Boson Sampling (GBS). Aunque GBS ha mostrado ventaja cuántica en problemas de teoría de grafos —especialmente en la búsqueda de subgrafos densos y cliques máximos—, su aplicación a Go requiere una representación adecuada. El hallazgo clave de este proyecto es que el tablero de Go admite múltiples representaciones naturales como grafo: desde los Common Fate Graphs (CFG) de Graepel, que codifican grupos de piedras y sus libertades, hasta grafos de adyacencia ponderados que capturan territorio e influencia. Esta transformación abre la puerta a explotar las capacidades de Xanadu de dos formas complementarias:

1. GBS para extracción de características: usar muestreo de subgrafos densos para identificar configuraciones estratégicas fuertes (grupos con muchas libertades, formaciones de ojos, estructuras defensivas). Estos “features cuánticos” capturan propiedades topológicas del tablero que son costosas de computar clásicamente.
2. Aprendizaje automático cuántico: los features extraídos mediante GBS o construidos con quantum feature maps en PennyLane pueden alimentar modelos supervisados para tareas como clasificación de posiciones, predicción de ganador y evaluación de la “fuerza” de un movimiento.

Así, mientras D-Wave ataca directamente el problema de optimización del Hamiltoniano de Ising para encontrar jugadas de baja energía, Xanadu complementa con extracción de características cuánticas derivadas de la estructura de grafos, útiles como entradas para ML o descriptores de posiciones estratégicas.

Este documento explora ambas plataformas en profundidad, implementando:
- En D-Wave: embedding del Hamiltoniano Atomic-Go y Molecular-Go en arquitectura Pegasus, con análisis de calidad de soluciones vía quantum annealing.
- En Xanadu/PennyLane: construcción de representaciones en grafos (CFG y grafos de adyacencia), extracción de features mediante GBS y diseño de quantum kernels para clasificación de posiciones.

El objetivo final es determinar qué arquitectura cuántica —o combinación híbrida— resulta más adecuada para extraer información estratégica del juego de Go, estableciendo un marco metodológico extensible a otros problemas combinatorios con estructura de grafos subyacente.

---

## 🧭 Arquitecturas y Flujo

Consulta el diagrama comparativo de flujos en `docs/arquitecturas_cuanticas.md`. Resume:
- D-Wave: Ising → QUBO → Embedding (Pegasus/Zephyr) → Annealing → Soluciones.
- Xanadu: Tablero → Grafo (CFG/Adyacencia) → GBS → Features → Kernel/ML.

Implementaciones útiles:
- Código: `src/go_energy_viz.py`, `src/go_visualization.py`, `src/go_isings_models.py`
- Notebooks: `notebooks/go_sgf_analysis.ipynb`, `notebooks/Hamiltonian_and_Ising_models.ipynb`

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
