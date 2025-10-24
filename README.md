# Proyect Quantum Go
Ising models for optimization problems in the game of Go

## Agradecimientos

Este proyecto fue desarrollado con el apoyo del Consejo Nacional de Ciencia y Tecnología (CONACYT) a través de la beca de Estancias Posdoctorales por México 2022 (3), CVU 469604, en la modalidad de Estancia Posdoctoral Académica - Inicial.

**Institución:** Facultad de Ingeniería, UNAM  
**Director de Proyecto:** Dr. Boris Escalante Ramírez  
**Período:** Diciembre 2022 - Noviembre 2024

## Quickstart

- Install in editable mode so notebooks can import the package without path hacks:
  - `pip install -e .`

- Open the demo notebook:
  - `notebooks/energy_tabs_demo.ipynb`
  - It renders energy maps (Manhattan‑1/2) for quantum and classical models and checks parity.

## Layout

- `src/` — Python package (import as `src.*`)
- `notebooks/` — Jupyter notebooks
- `data/`, `results/` — optional datasets and outputs

## Notes

- Dependencies are intentionally minimal in packaging; install required libs in your environment (numpy, matplotlib, bokeh, ipywidgets, pennylane, scipy, an SGF parser).
