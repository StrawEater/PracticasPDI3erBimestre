# Guía del Docente: Gestión y Evaluación del TP Final de Rompecabezas PDI
**Cátedra de Procesamiento Digital de Imágenes**  
*Documento de uso interno para el equipo de profesores y ayudantes.*

---

## 1. Visión General del Marco de Trabajo

Este framework ha sido diseñado para simplificar la administración, evaluación y reproducibilidad del TP Final:

1. **Generador reproducible (`src/generator.py`):** Permite cortar cualquier imagen en una grilla $R \times C$, desordenarla y opcionalmente rotar piezas, almacenando de forma segura la verdad fundamental (`ground_truth.json`).
2. **Evaluador estandarizado (`src/evaluator.py`):** Calcula métricas formales de la literatura científica (Direct Accuracy, Neighbor Pair Accuracy, Largest Connected Component) y genera mapas de error visual en PNG.
3. **Solver de Referencia / Baseline (`src/baseline_solver.py`):** Implementa un piso técnico básico (SSD en espacio Lab con búsqueda voraz) para que los alumnos tengan un punto de comparación claro.
4. **Evaluador por Lotes (`src/batch_evaluator.py`):** Permite evaluar las entregas de todos los grupos de la cursada en un solo comando y genera una planilla CSV con las notas cuantitativas.
5. **Material para Estudiantes (`student_template/`):** Estructura lista para comprimir y distribuir a los alumnos, incluyendo un notebook interactivo (`notebook_guia.ipynb`).

---

## 2. Flujo de Trabajo Docente

### Fase 1: Publicación del TP (Inicio de la actividad)
1. Comprimir la carpeta `student_template/` y la carpeta `dataset_ejemplos/` (asegurándose de no incluir archivos `prediction*.json` generados durante pruebas si se desea que empiecen limpios).
2. Entregar a los alumnos `ENUNCIADO.md` y `RUBRICA.md`.
3. Compartir el script `src/evaluator.py` y `src/metrics.py` (o toda la carpeta `src/`) para que los alumnos puedan autoevaluar su progreso localmente.

### Fase 2: Creación del Conjunto Secreto de Evaluación (Para el examen/entrega)
Para evitar que los alumnos "sobreajusten" (overfitting) sus heurísticas a los ejemplos públicos, se recomienda generar un conjunto de prueba oculto con nuevas imágenes, rotaciones leves o modulación periódica:

```bash
# Ejemplo 1: Rompecabezas 3x3 fácil (imagen natural con encastres)
python src/generator.py --image ../imagenes/pecesito.jpg --rows 3 --cols 3 --output dataset_secreto/eval_3x3_facil --seed 999

# Ejemplo 2: Rompecabezas 3x3 con rotación ortogonal y leve continua ("rotalas un poco")
python src/generator.py --image ../imagenes/cueva.jpg --rows 3 --cols 3 --rotate --slight-rotation --jitter-degrees 12.0 --output dataset_secreto/eval_3x3_rotado --seed 777

# Ejemplo 3: Rompecabezas 3x3 con filtro de rayas periódicas y píxel marcador de ancla
python src/generator.py --image ../imagenes/barco.jpg --rows 3 --cols 3 --add-stripes --stripe-period 8 --add-marker --output dataset_secreto/eval_3x3_rayas_marcador --seed 888
```

> **Importante para los docentes:** Al distribuir este conjunto de evaluación a los alumnos para la entrega final, compartan la carpeta con las piezas (`pieces/`) pero **NO** incluyan `original.png` ni `ground_truth.json`. Conserven esos archivos en el repositorio de la cátedra para la corrección.

---

### Fase 3: Corrección Automatizada por Lotes

Cuando los alumnos envían sus carpetas de entrega (por ejemplo, dentro de una carpeta `entregas_alumnos/`), la estructura debe ser:

```text
entregas_alumnos/
├── grupo_01/
│   ├── prediction_eval_3x3_facil.json
│   ├── prediction_eval_4x4_medio.json
│   └── prediction_eval_3x3_rotado.json
├── grupo_02/
│   └── ...
```

Para procesar todas las entregas automáticamente y consolidar las notas en un CSV:

```bash
python src/batch_evaluator.py --test-set dataset_secreto/ --submissions entregas_alumnos/ --output-csv calificaciones_finales.csv
```

El script imprimirá el progreso de cada grupo y generará `calificaciones_finales.csv` con columnas:
`Grupo`, `[Puzzle]_NeighborAcc%`, `[Puzzle]_DirectAcc%`, `[Puzzle]_LCC%`, `Promedio_NeighborAcc%`.

---

## 3. Preguntas Clave para la Defensa Oral de los Alumnos

Para verificar la autoría individual y comprensión de los conceptos de Procesamiento Digital de Imágenes durante el coloquio o defensa:

1. **Sobre Binarización y Contornos:**
   - *¿Cómo implementaron la binarización de las piezas para separar la silueta del fondo negro? ¿Por qué conviene aplicar una operación morfológica de cierre posterior al umbralizado (ej. Otsu)?*
   - *¿Cómo detectaron las 4 esquinas de la grilla base y qué criterio utilizaron para clasificar los lados en PLANO, MACHO y HEMBRA?*
2. **Sobre Matcheo de Encastres y Curvas Específicas:**
   - *¿Cómo discriminaron entre las distintas familias de curvas (`standard`, `circular`, `random`)?*
   - *¿Qué porcentaje de comparaciones lograron podar al imponer la restricción física de que solo un lado MACHO puede unirse a un lado HEMBRA de la misma curvatura?*
3. **Sobre Análisis de Rayas en Frecuencia (Fourier 2D) y Gradientes:**
   - *¿Cómo utilizaron la Transformada 2D de Fourier (`np.fft.fft2`) o los gradientes direccionales de Sobel para estimar el ángulo de inclinación a partir del filtro de rayas horizontales?*
   - *¿Por qué las rayas horizontales generan picos en el eje vertical en el dominio de la frecuencia? ¿Cómo se relaciona el espaciado entre rayas con la distancia al origen en el espectro de potencia?*
4. **Sobre Rotaciones Leves y Deskewing:**
   - *¿Cómo implementaron la rectificación geométrica (deskewing) de piezas ligeramente rotadas antes de calcular las tiras de borde? ¿Qué tipo de interpolación utilizaron (`INTER_LINEAR` vs `INTER_NEAREST`)?*
5. **Sobre Espacios de Color y Gradientes:**
   - *¿Qué diferencia observaron al calcular distancias de bordes en RGB respecto a CIE-Lab o HSV? ¿Por qué la distancia euclidiana en RGB es subóptima para medir similitud visual percibida?*
6. **Sobre el Algoritmo de Ensamble y Piezas Ancla:**
   - *¿Cómo aprovecharon la detección del píxel marcador característico para fijar la pieza inicial y orientar el algoritmo voraz / MRV hacia afuera?*

---

## 4. Tipologías de Imágenes y Casos Límite a Tener en Cuenta

- **Zonas homogéneas (Cielo / Fondo marino / Paredes lisas):** En imágenes como `algas.jpg` o `cueva.jpg`, varios bordes tienen colores casi idénticos. Los alumnos que identifiquen bordes con baja varianza o que prioricen ubicar primero las piezas con mayor información (alto contraste / bordes fuertes) obtendrán mejores resultados.
- **Patrones periódicos o repetitivos:** Si una imagen contiene franjas o texturas repetidas, la función de costo local puede tener mínimos locales engañosos. Esto permite evaluar si el grupo diseñó mecanismos de coherencia global.
