# Reto OXXO — Optimización de Planogramas

**Reto MA2008B — Optimización No Lineal · ITESM**

Optimización del acomodo de productos en muebles de cuarto frío de tiendas OXXO.
Dada una lista de productos con sus frentes ya decididos y un mueble con un número
fijo de charolas, se determina **en qué charola y en qué orden** colocar cada
producto, maximizando la **calidad del acomodo**: aprovechamiento de espacio,
coherencia de tamaño y agrupación de marca.

El problema se resuelve con un **algoritmo genético** (Modelo 1 y su extensión
Modelo 2) y se compara contra un **modelo de optimización no lineal** (codificación
*random-key* resuelta por *Differential Evolution*), sobre planogramas reales
reconstruidos del histórico.

---

## Estructura del proyecto

```
RETO_OXXO_OPTINL/
├── preprocessing/      Reconstrucción de planogramas y simulación de pesos
├── genetic_algorithm/  Modelos de acomodo por algoritmo genético (GA1, GA2)
├── mathematical_model/ Modelo no lineal y comparación contra el GA
├── data/               Datos de entrada y artefactos generados por el pipeline
├── outputs/            Resultados: gráficas, tablas y CSV
└── evidencias/         Enunciados del reto y reportes entregables (PDF/LaTeX)
```

### `preprocessing/`
- **`pipeline_.py`** — Convierte el histórico crudo (`data/Ejemplo.csv`, ~497k filas)
  en planogramas individuales. Agrupa por `SEGMENTO + MUEBLE + TAMAÑO + PLANOGRUPO`,
  reconstruye cada planograma por **bloques-producto** (cada `UBICACION_BANDEJA` es un
  frente; se colapsan frentes consecutivos del mismo UPC), y estima los pesos
  `w1, w2, w3` por simulación de Montecarlo (1000 escenarios). Genera los artefactos
  `oxxo_*` en `data/`.

### `genetic_algorithm/`
- **`ga1.py`** — Modelo 1 (no extendido). Maximiza
  `w1·aprovechamiento − w2·incoherencia + w3·marca`. Permutación + decodificador
  *first-fit*, cruce OX, mutación swap/inserción, torneo y elitismo.
- **`ga2.py`** — Modelo 2 (extendido). Añade el término `w4` de consistencia con la
  ubicación esperada histórica `E[charola]` y un multiplicador de importancia
  (placeholder de demanda/margen).
- El formato a resolver se elige con la variable `FORMATO` o la variable de entorno
  `OXXO_FORMATO`.

### `mathematical_model/`
- **`nlp_model.py`** — Modelo no lineal *random-key* (variables continuas en `[0,1]`)
  resuelto con `scipy.optimize.differential_evolution`, usando la **misma** función
  objetivo y decodificador que el GA. Corre Histórico vs. GA vs. NLP sobre las mismas
  instancias y genera las tablas y figuras de comparación.
- **`math_model.py`** — Funciones auxiliares de reporte (tablas LaTeX, CSV, gráficas).

### `data/`
- **`Ejemplo.csv`** — Histórico crudo (~45 MB). **No versionado** (ver `.gitignore`);
  colócalo aquí para correr el pipeline.
- **`ejemplo_planograma.csv`** — Dataset pequeño de demostración.
- **`oxxo_instance_*.csv`** — Bloques-producto de cada planograma (dimensiones,
  frentes, marca, `E[charola]`).
- **`oxxo_locdist_*.json`** — Distribución de probabilidad de charola por producto.
- **`oxxo_weights_*.json`** — Pesos `w1, w2, w3` simulados, con desviaciones y un
  campo de validación.

### `outputs/`
- `resultados_ga1.png`, `resultados_ga2.png` — Convergencia del fitness de cada GA.
- `convergencia_*.png`, `criterios_*.png` — Comparación GA vs. NLP.
- `planograma_bco.png` — Planograma reconstruido por el GA.
- `comparison_tables.tex`, `comparison_results.csv` — Tablas de comparación.
- `expected_locations_*.csv` — Ubicación esperada por producto.

### `evidencias/`
- `RETO_MA2008B.pdf`, `RETO_MA2008B_E2.pdf` — Enunciados del reto.
- `RETO_MA2008B_E3.pdf` — Reporte final del Entregable 3.
- `reporte_e3.tex` — Fuente LaTeX del reporte.

---

## Cómo ejecutar

Requiere **Python 3.13** con `numpy`, `pandas`, `scipy`, `Pillow` y `matplotlib`.

```bash
# 1. Reconstruir planogramas y simular pesos (genera data/oxxo_*)
python preprocessing/pipeline_.py

# 2. Resolver con el algoritmo genético
python genetic_algorithm/ga1.py            # Modelo 1
python genetic_algorithm/ga2.py            # Modelo 2 (extendido)
OXXO_FORMATO=OFC_CF_5.5_Refrescos python genetic_algorithm/ga1.py   # otro formato

# 3. Modelo no lineal + comparación (genera tablas y figuras en outputs/)
python mathematical_model/nlp_model.py BCO_CF_4.0_Refrescos
```

Todas las rutas están ancladas a la raíz del proyecto, así que los scripts corren
desde cualquier directorio.
