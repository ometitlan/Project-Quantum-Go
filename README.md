<p align="center">
  <img src="data/assets/logo_ing.jpg" alt="Facultad de Ingeniería UNAM" width="180"/>
</p>

<h1 align="center">Proyecto Quantum Go</h1>
<p align="center">Modelos de Ising para optimización y análisis estratégico del juego de Go con recocido cuántico (D-Wave) y computación fotónica (Xanadu).</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
  <a href="https://ocean.dwavesys.com/"><img src="https://img.shields.io/badge/D--Wave-Ocean%20SDK-00ADD8" alt="D-Wave"/></a>
  <a href="https://pennylane.ai/"><img src="https://img.shields.io/badge/PennyLane-Xanadu-22c55e" alt="PennyLane"/></a>
</p>

---

## Tabla de contenidos

- [Visión general](#visión-general)
- [Demo visual](#demo-visual)
- [Inicio rápido](#inicio-rápido)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Componentes principales](#componentes-principales)
- [Datos y recursos](#datos-y-recursos)
- [Documentación relacionada](#documentación-relacionada)
- [Estado y siguientes pasos](#estado-y-siguientes-pasos)
- [Agradecimientos y créditos](#agradecimientos-y-créditos)
- [Licencia](#licencia)

---

## Visión general

Quantum Go explora dos paradigmas cuánticos complementarios para estudiar posiciones de Go:

- **D-Wave / Recocido cuántico:** formulamos Hamiltonianos de Ising (Atomic-Go y variantes) que respondan a la dinámica del juego de Go y los llevamos a formulación QUBO para poder ejecutar embeddings en arquitecturas Pegasus/Zephyr y obtener configuraciones de mínima energía de manera eficiente. Resolvemos problemas de optimización que sugieren jugadas prometedoras basadas en la minimización de energía del sistema Ising.
- **Xanadu / Computación fotónica:** convertimos el tablero en grafos donde los nodos son grupos de piedras conectadas y las aristas cuantifican relaciones estratégicas (libertades compartidas, presión táctica), transformando su matriz de adyacencia mediante descomposición de Takagi en parámetros de un interferómetro fotónico que, al propagar fotones a través de Xanadu Borealis, produce muestras que identifican subgrafos densos (formaciones sólidas, moyos, zonas de conflicto), De esta manera podemos extraer *features* cuánticos y alimentar kernels o modelos de aprendizaje automático en PennyLane.

El repositorio combina motor clásico de reglas, visualizaciones interactivas, pipelines de exportación y prototipos de modelos cuánticos, pensado tanto para notebooks exploratorios como para integraciones futuras con hardware real.

### Objetivos principales

- Describir y comparar los flujos D-Wave vs Xanadu para Go.
- Proveer un motor de juego autocontenido con validación de reglas, parsing SGF y herramientas de análisis.
- Visualizar posiciones, libertades y mapas de energía (clásicos y cuánticos) desde notebooks o scripts.

---

## Demo visual

<p align="center">
  <img src="data/assets/Interfaz_1.png" alt="Demo del navegador interactivo" width="600"/>
</p>

*La interfaz `GameNavigator` permite recorrer partidas SGF, mostrar libertades por grupo, números de jugada y pestañas de energía Manhattan-1/2 (cuántica y clásica).*

---

## Inicio rápido

1. **Instala el paquete en modo editable** (evita hacks de ruta en notebooks):
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```
2. **Abre el notebook de análisis interactivo**:
   - `notebooks/go_sgf_analysis.ipynb`
   - Carga partidas desde `data/sgf partidas/`, navega movimiento a movimiento y exporta PNG/GIF.
3. **Explora los modelos de Ising**:
   - `notebooks/Hamiltonian_and_Ising_models.ipynb`
   - Calcula mapas de energía (Manhattan-1/2) y compara kernels cuánticos vs clásicos.

---

## Estructura del repositorio

| Ruta | Contenido |
| --- | --- |
| `src/` | Paquete Python instalable (`pip install -e .`). Incluye motor de Go, parsers SGF, visualizaciones y modelos de Ising. |
| `notebooks/` | Cuadernos Jupyter de demostración (`go_sgf_analysis`, `Hamiltonian_and_Ising_models`). |
| `docs/` | Guías en Markdown (`INTRODUCTION.md`, `analisis_interactivo_partidas_go.md`, `arquitecturas_cuanticas.md`). |
| `data/assets/` | Imágenes para documentación e interfaces (interfaz, tableros, logo). |
| `data/sgf partidas/` | Biblioteca de partidas profesionales (.sgf) para pruebas. |
| `results/` | Salidas generadas (PNG, GIF, HTML) por los notebooks/utilidades. |

---

## Componentes principales

- **Motor de juego (`src/go_game_engine.py`):** clase `GoBoard` con reglas completas (Ko, suicidio, capturas, replay), estructuras `MoveInfo`/`GameInfo` y `SGFParser`.
- **Utilidades SGF/tableros (`src/sgf_utils.py`, `src/board_utils.py`):** conversión entre matrices y tableros, extracción de metadatos y movimientos.
- **Visualización (`src/go_visualization.py`):** `GoBoardVisualizer`, `GameNavigator` (tabs Libertades/Jugadas/Energía) y herramientas de exportación.
- **Modelos de Ising (`src/go_isings_models.py`):** `IsingGoConfig`, `QuantumIsingModel`, `ClassicalIsingModel` y `EnergyMapGenerator` con kernels Manhattan configurables.
- **Mapas energéticos (`src/go_energy_viz.py`, `src/go_export_utils.py`):** generación de pestañas Bokeh (M1/M2, cuántico/clásico) y exportación a PNG/HTML/GIF.

---

## Datos y recursos

- **Partidas SGF:** `data/sgf partidas/` contiene decenas de partidas profesionales (GoKifu, Waltheri) listas para análisis.
- **Activos visuales:** `data/assets/Interfaz_*.png`, `Tablero_*.png`, `partida.gif`, útiles en reportes o presentaciones.
- **Requisitos:** ver `requirements.txt` (numpy, scipy, matplotlib, bokeh, ipywidgets, pillow, pennylane, sgf/sgfmill, etc.). El paquete base (`pyproject.toml`) mantiene dependencias mínimas; instala extras según necesites.

---

## Documentación relacionada

- `INTRODUCTION.md`: visión completa, fundamentos teóricos y motivación de D-Wave + Xanadu.
- `docs/analisis_interactivo_partidas_go.md`: guía paso a paso del notebook de análisis.
- `docs/arquitecturas_cuanticas.md`: diagramas Mermaid con flujos D-Wave vs Xanadu.

---

## Estado y siguientes pasos

- [x] Motor de juego con validaciones y replay SGF.
- [x] Visualizaciones interactivas + exportación multimedia.
- [x] Prototipos de mapas de energía (Manhattan-1/2) cuánticos y clásicos.
- [ ] Integración completa de pestañas de energía en el navegador (WIP).
- [ ] Experimentos con hardware D-Wave/Xanadu reales y pipelines híbridos ML.

---

## Agradecimientos y créditos

Proyecto desarrollado con el apoyo del Consejo Nacional de Ciencia y Tecnología (CONACYT) mediante la beca de Estancias Posdoctorales por México 2022 (modalidad Académica - Inicial), CVU 469604.

- **Institución:** Facultad de Ingeniería, UNAM  
- **Director de Proyecto:** Dr. Boris Escalante Ramírez  
- **Período:** Diciembre 2022 - Noviembre 2024

**Autores**

- Dr. Mario Alberto Mercado Sánchez — ometitlan@gmail.com  
- Leonardo Jiménez (Matemático) — leonsinmiedo@gmail.com  
- Repositorio oficial: <https://github.com/ometitlan/Project-Quantum-Go>

---

## Licencia

Código disponible bajo licencia [MIT](LICENSE). Si empleas este trabajo en publicaciones o demostraciones, incluye los créditos correspondientes y referencia este repositorio.
