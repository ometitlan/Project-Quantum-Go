# Proyect Quantum Go
Adiabatic Quantum Strategies: Optimizing Go Game Configurations trought Ising models

## Agradecimientos

Este proyecto fue desarrollado con el apoyo del Consejo Nacional de Ciencia y Tecnología (CONACYT) a través de la beca de Estancias Posdoctorales por México 2022 (3), CVU 469604, en la modalidad de Estancia Posdoctoral Académica - Inicial.

**Institución:** Facultad de Ingeniería, UNAM  
**Director de Proyecto:** Dr. Boris Escalante Ramírez  
**Período:** Diciembre 2022 - Noviembre 2024

# Project-Quantum-Go

M. A. Mercado Sánchez, L. Jiménez — Universidad Nacional Autónoma de México (UNAM)

## Resumen

## Descripción

Este proyecto está pensado para ejecutarse en la computadora cuántica adiabática de D‑Wave, donde el problema central consiste en transformar un modelo de Ising en un modelo QUBO para resolver problemas de optimización clásica mediante efectos cuánticos (como trabaja D‑Wave). Es especialmente potente para problemas de redes como MaxCut, y usando el  mismo enfoque trabajamos la simulación en PennyLane de donde obtuvimos representaciones o mapas de energia y posteriormente replanteamos lienas de acción para trabajar en D‑Wave.

Este proyecto investiga la optimización del juego de Go mediante modelos de Ising y su resolución tanto en simulación clásica como en hardware cuántico (p. ej., D‑Wave, Xanadu). Adoptamos un modelo de Ising con restricciones adicionales para construir mapas de energía más realistas del tablero, integrando física estadística y cómputo cuántico adiabático. Esta aproximación dual permite evaluar estrategias de Go desde la perspectiva clásica y cuántica, mejorando la precisión y efectividad de las decisiones estratégicas.

En la práctica, mapeamos configuraciones del tablero de Go a variables tipo espín (Ising), añadimos términos de penalización para capturar reglas/estructuras del juego y convertimos la formulación Ising↔QUBO (p. ej., con el cambio de variable x = 2z − 1) para su ejecución en el annealer de D‑Wave. Validamos el pipeline con simulaciones en PennyLane (compatible con ecosistema Xanadu) y contrastamos resultados en hardware cuántico.

## Motivación y Contexto

- El cómputo adiabático y el recocido cuántico buscan el estado fundamental de un Hamiltoniano de costo, evitando mínimos locales mediante fluctuaciones cuánticas.
- El modelo de Ising mapea variables binarias a espines con interacciones locales; el mínimo de energía corresponde a la solución óptima del problema.
- El juego de Go se puede formular como optimización combinatoria: maximizar territorio y capturas bajo reglas simples en un espacio de estados inmenso.
- D‑Wave resuelve QUBO/Ising de gran escala (p. ej., MaxCut), lo que motiva la traducción del modelo de Go a esta clase de Hamiltonianos.
- PennyLane permite simular y prototipar el modelo (y facilita su transporte a hardware fotónico de Xanadu) antes de ejecutar en D‑Wave.


## Modelo de Ising Clásico (Atomic-Go)

Para una posición del tablero:
- Variable de sitio `x_i ∈ {−1, 0, +1}`: negra, vacío, blanca.
- Energía local por vecindad `N(i)`: `s_i = x_i * sum_{j∈N(i)} x_j`.
- Término de libertades (campo externo): `h_i` = número de libertades; `μ ≥ 0` su peso.
- Energía total: `H_Atomic(t) = − sum_i s_i^(t) − μ * sum_i h_i^(t) x_i^(t)`.

Interpretación: configuraciones de baja energía tienden a formar grupos robustos y posiciones favorables.

## Modelo de Ising Cuántico (Mapa de Energía Cuántico)

Extiende el clásico con superposición y correlaciones cuánticas:
- Un qubit por intersección: `|0⟩` piedra negra, `|1⟩` piedra blanca, vacío como superposición.
- Hamiltoniano tipo Ising con preparación/perturbación:
  `H = − J ∑_{⟨i,j⟩} σ_z^i σ_z^j − h ∑_{⟨i,j⟩} σ_x^i ⊗ (H σ_x^j)`
- La energía esperada `⟨H⟩ = ⟨ψ|H|ψ⟩` actúa como métrica del estado del juego y guía estratégica.

## Alcance y Contribuciones

- Construcción de mapas de energía clásicos y cuánticos del tablero de Go.
- Inclusión de restricciones/condiciones que acercan el mapa de energía a reglas reales del juego.
- Evaluación con partidas profesionales y comparación de estrategias derivadas de ambos mapas.
- Base para explorar resolución adiabática en hardware cuántico (p. ej., D‑Wave) del Hamiltoniano clásico.

## Estructura del Repositorio

- `src/`
  - `go_game_engine.py`: motor y reglas de Go.
  - `go_isings_models.py`: definiciones de modelos Ising (clásico/cuántico) y utilidades de energía.
  - `go_visualization.py`: funciones de visualización de tablero y jugadas.
  - `go_energy_viz.py`: visualización de mapas de energía.
  - Otros módulos de apoyo (mediciones, utilidades).
- `notebooks/`
  - Análisis de SGF, construcción de Hamiltonianos y experimentos (clásicos/cuánticos).
- `data/` (opcional en control de versiones)
  - Partidas y recursos de datos.
- `results/` (salidas generadas; típico no versionarlas).

## Instalación Rápida

Requisitos: Python 3.9+

Windows PowerShell:
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt



Linux/macOS:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt



## Uso Básico

- Ejecuta los notebooks en `notebooks/` para reproducir ejemplos y experimentos.
- Código fuente en `src/` para motor de Go, cálculo de energías (clásico/cuántico) y visualizaciones.
- Si compartes datasets/SGF, documenta su origen en `data/README.md`.

## Datos y Reproducibilidad

- Si tus SGF son pequeños y esenciales, versiona `data/`.
- Si `data/` crece mucho, considera un subconjunto curado o Git LFS.
- Mantén `results/` fuera de versión (artefactos reproducibles).

## Roadmap

- Analizar ≥6,000 partidas profesionales y correlacionar energía final con resultado real.
- Comparar arquitecturas que usan mapas clásicos vs cuánticos como feature maps.
- Explorar resolución adiabática del Hamiltoniano clásico en hardware cuántico especializado.

## Cita

Mercado Sánchez, M. A., Jiménez, L. “Adiabatic Quantum Strategies: Optimizing Go Game Configurations”. Universidad Nacional Autónoma de México (UNAM).

## Licencia

Ver `LICENSE`.
