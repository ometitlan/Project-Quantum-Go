# Análisis Interactivo de Partidas de Go 🎮

Notebook de Jupyter para cargar, analizar y visualizar partidas profesionales de Go con navegación interactiva y exportación de visualizaciones.

![Demo del navegador](../data/assets/Interfaz_1.png)
*Interfaz de navegación interactiva con múltiples vistas*

---

## 📋 Descripción

Este notebook proporciona una interfaz completa para:

✅ Cargar partidas profesionales en formato SGF  
✅ Reproducir y validar todos los movimientos  
✅ Navegar interactivamente por la partida  
✅ Visualizar posiciones con libertades o números de jugada  
✅ Exportar posiciones a imágenes PNG de alta calidad  
✅ Generar animaciones GIF de partidas completas  
  

---

## 🎯 Estructura del Notebook

### Celda 1: Configuración e Imports
Carga todas las dependencias necesarias.

### Celda 2: Parsear Archivo SGF y Reproducir Partida
- Define la ruta del archivo SGF a analizar
- Carga y valida el archivo
- Extrae metadata y movimientos
- Muestra información completa de la partida
- Reproduce todos los movimientos
- Valida reglas (Ko, capturas, suicidio)
- Visualiza posición final con dos vistas:
  - 📊 Con libertades
  - 🔢 Con números de movimiento

![Posición libertades por grupo](../data/assets/Tablero_Libertades.png)
*Ejemplo de visualización de posición final con libertades por grupo*

![Posición libertades por grupo](../data/assets/Tablero_movimientos.png)
*Ejemplo de visualización de posición final con números de movimiento*

### Celda 3: Navegador Interactivo Multi-Vista ⭐

Interfaz principal con controles de navegación y múltiples tabs:

**Controles disponibles:**
- ⏮️ Inicio / ⏭️ Final
- ⏪ -10 / ⏩ +10 movimientos
- ◀️ Anterior / ▶️ Siguiente

**Tabs disponibles:**

| Tab | Descripción | Utilidad |
|-----|-------------|----------|
| 📊 **Libertades** | Números = libertades de cada grupo | Análisis táctico, grupos en atari |
| 🔢 **Jugadas** | Números = orden de movimiento | Seguir secuencias, estudiar aperturas |
| ⚡ **Energía** | Preparado para modelos cuánticos | Análisis posicional avanzado (futuro) |

![Tabs múltiples](../data/assets/Interfaz_2.png)
*Interfaz*

### Celda 4: Exportar Posiciones a PNG 📸

Exporta imágenes de alta calidad configurables:

**Parámetros personalizables:**
- Movimiento específico a exportar
- Mostrar libertades o números de jugada
- Calidad DPI (150/300/600)
- Exportar posición final

**Archivos generados:** `../results/*.png`

### Celda 5: Crear Animación GIF 🎬

Genera animaciones completas de la partida:

**Parámetros configurables:**
- Velocidad de animación (ms por frame)
- Límite de movimientos (útil para partidas largas)
- Visualización en el notebook

**Archivo generado:** `../results/*.gif`

![GIF ejemplo](../data/assets/partida.gif)
*Animación de una partida completa*

### 🛠️ Módulos Utilizados

### `go_game_engine.py` - Motor del Juego

**Características principales:**

| Componente | Capacidades |
|------------|-------------|
| **GoBoard** | Motor completo con validación de reglas |
| ├─ Reglas | Ko, capturas, suicidio, superko |
| ├─ Análisis | Libertades, grupos, capturas |
| └─ Replay | Reproducción completa de partidas |
| **SGFParser** | Parser robusto de archivos SGF |
| ├─ Metadata | Jugadores, fecha, resultado, komi |
| ├─ Movimientos | Extracción de secuencias |
| └─ Comentarios | Preservación de anotaciones |

**Validaciones implementadas:**
- ✅ Ko simple y superko
- ✅ Detección de suicidio
- ✅ Captura de grupos sin libertades
- ✅ Validación de posiciones legales

---

### `go_visualization.py` - Visualización

**Características principales:**

| Componente | Capacidades |
|------------|-------------|
| **GoBoardVisualizer** | Renderizado de tableros |
| ├─ Matplotlib | Visualización estática de alta calidad |
| ├─ Bokeh | Visualización interactiva web (opcional) |
| ├─ Libertades | Números en grupos |
| ├─ Movimientos | Orden de jugada |
| └─ Heatmaps | Mapas de calor de libertades |
| **GameNavigator** | Navegación interactiva |
| ├─ Controles | Slider + botones de navegación |
| ├─ Tabs | Sistema escalable de múltiples vistas |
| └─ Callbacks | Actualización sincronizada |
| **Funciones Export** | Exportación de visualizaciones |
| ├─ PNG | Imágenes de alta calidad (hasta 600 DPI) |
| ├─ GIF | Animaciones completas |
| └─ Comparación | Vista lado a lado |

---

## 🔮 Próximas Características

- ⚡ Integración con modelo cuántico (Tab de Energía)
- 🧠 Análisis de influencia y territorio
- 📊 Gráficas de ventaja durante la partida
- 🎯 Detección automática de joseki
- 💾 Exportación a formato personalizado

---

## 📚 Recursos

**Descargar partidas profesionales:**
- [GoKifu](https://gokifu.com/) - Partidas de torneos
- [Waltheri](http://ps.waltheri.net/) - Base de datos masiva
- [OGS](https://online-go.com/) - Partidas en línea

**Formato SGF:**
- [Especificación SGF](https://www.red-bean.com/sgf/)

---

## 📄 Licencia

El **código fuente** de este repositorio se distribuye bajo **Apache-2.0**.  
La **documentación, notebooks explicativos e imágenes** en `notebooks/` y `assets/` se distribuyen bajo **CC BY 4.0**.

- SPDX (código): `Apache-2.0`
- SPDX (docs): `CC-BY-4.0`

Consulta los archivos [`LICENSE`](./LICENSE) y [`LICENSE-docs`](./LICENSE-docs) para los términos completos.

> Nota: Si reutilizas partes de este proyecto, conserva los avisos de copyright y menciona la atribución correspondiente para los materiales bajo CC BY 4.0.


---

**Creado con ❤️ para la comunidad de Go**
