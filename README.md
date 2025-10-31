# Proyecto Quantum Go
Modelos de Ising para problemas de optimización en el juego de Go

## Agradecimientos

Este proyecto fue desarrollado con el apoyo del Consejo Nacional de Ciencia y Tecnología (CONACYT) a través de la beca de Estancias Posdoctorales por México 2022 (3), CVU 469604, en la modalidad de Estancia Posdoctoral Académica - Inicial.

**Institución:** Facultad de Ingeniería, UNAM  
**Director de Proyecto:** Dr. Boris Escalante Ramírez  
**Período:** Diciembre 2022 - Noviembre 2024

## Inicio rápido

- Instala en modo editable para que los notebooks importen el paquete sin hacks de ruta:
  - `pip install -e .`

- Abre el notebook de demostración:
  - `notebooks/go_sgf_analysis.ipynb`
  - Muestra navegación, visualización y pestañas de energía (cuando están habilitadas).

## Estructura

- `src/` — Paquete Python (importa como `src.*`)
- `notebooks/` — Jupyter notebooks
- `docs/` — Guías y documentación en Markdown
- `data/`, `results/` — Conjuntos de datos y salidas

## Documentación

- Introducción y teoría: `INTRODUCTION.md`
- Guía de análisis interactivo: `docs/analisis_interactivo_partidas_go.md`

## Notas

- Los requisitos de empaquetado son mínimos; instala en tu entorno las bibliotecas necesarias (numpy, matplotlib, bokeh, ipywidgets, pennylane, scipy y un parser SGF).

## Créditos

- Autor: Dr. Mario Alberto Mercado Sánchez — ometitlan@gmail.com
- Colaborador (Matemático): Leonardo Jiménez — leonsinmiedo@gmail.com
- Repositorio: https://github.com/ometitlan/Project-Quantum-Go
