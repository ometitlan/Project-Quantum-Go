# Arquitecturas y Flujos: D-Wave vs Xanadu

Este documento resume, de forma visual, los flujos de trabajo de D-Wave (recocido cuántico adiabático) y de Xanadu (computación cuántica fotónica) aplicados al análisis del juego de Go.

## Flujo D-Wave (Optimización Ising/QUBO)

```mermaid
flowchart TD
    A[Tablero de Go] --> B[Hamiltoniano de Ising<br/>(Atomic-Go / Molecular-Go)]
    B --> C[Transformación QUBO]
    C --> D[Minor-Embedding<br/>(Pegasus/Zephyr)]
    D --> E[Recocido Cuántico<br/>(Quantum Annealing)]
    E --> F[Soluciones de Baja Energía]
    F --> G[Post-procesamiento y selección de jugada]
```

- Implementación relacionada:
  - Código: `src/go_isings_models.py`, `src/go_energy_viz.py`, `src/go_visualization.py`
  - Notebook: `notebooks/Hamiltonian_and_Ising_models.ipynb`

## Flujo Xanadu (Grafos + GBS + ML)

```mermaid
flowchart TD
    A[Tablero de Go] --> B[Representación en Grafos<br/>(CFG, Adyacencia)]
    B --> C[Gaussian Boson Sampling<br/>(GBS)]
    C --> D[Extracción de Features Cuánticos]
    D --> E[Quantum Kernel / ML Supervisado]
    E --> F[Clasificación / Evaluación Estratégica]
```

- Implementación relacionada:
  - Código: `src/go_isings_models.py` (representaciones), `src/go_visualization.py`
  - Notebooks: `notebooks/go_sgf_analysis.ipynb`, `notebooks/Hamiltonian_and_Ising_models.ipynb`

Notas:
- Los flujos pueden combinarse en un enfoque híbrido: D-Wave genera candidatas de jugada y Xanadu evalúa con features cuánticos para apoyar decisiones.
- GitHub soporta Mermaid; si no se renderiza, consulta el archivo directamente en GitHub.
