# Rúbrica de Evaluación Oficial - TP Final PDI
**Cátedra de Procesamiento Digital de Imágenes**  
**Escala de Calificación:** 0 a 10 puntos (Aprobación con nota $\ge 4$, Promoción con nota $\ge 7$)

---

## 1. Desglose Porcentual de Criterios

| Criterio | Ponderación | Puntaje Máximo |
| :--- | :---: | :---: |
| **1. Desempeño Cuantitativo en Dataset de Prueba** | 40% | 4.0 pts |
| **2. Métodos y Fundamentación de PDI** | 30% | 3.0 pts |
| **3. Calidad del Código y Reproducibilidad** | 15% | 1.5 pts |
| **4. Informe Técnico y Análisis Crítico** | 15% | 1.5 pts |
| **Total** | **100%** | **10.0 pts** |

---

## 2. Matriz Detallada de Evaluación

### Criterio 1: Desempeño Cuantitativo en Dataset Oculto (4.0 puntos)
Se evaluará la solución sobre un conjunto de prueba secreto preparado por la cátedra conteniendo instancias de diferentes dimensiones ($3 \times 3$, $4 \times 4$, $5 \times 5$) con y sin rotación.

- **Sobresaliente (3.5 - 4.0 pts):**
  - Supera ampliamente el Baseline en todas las instancias de Nivel 1 (Neighbor Accuracy $> 75\%$).
  - Resuelve con alta precisión instancias con rotación ortogonal (Nivel 2).
  - Componente conexa mayor $> 80\%$ en la mayoría de los casos.
- **Muy Bueno (2.8 - 3.4 pts):**
  - Supera el Baseline de la cátedra en Nivel 1 (Neighbor Accuracy $> 55\%$).
  - Implementa soporte para rotaciones (Nivel 2) con resultados parciales o en grillas chicas ($3 \times 3$).
- **Regular / Aprobado Mínimo (1.6 - 2.7 pts):**
  - Alcanza un desempeño comparable al Baseline en Nivel 1 (Neighbor Accuracy entre $35\%$ y $50\%$).
  - No resuelve rotaciones o tiene errores de convergencia en grillas mayores a $3 \times 3$.
- **Insuficiente (0.0 - 1.5 pts):**
  - Desempeño inferior al Baseline o código que genera colocaciones inválidas / no converge.

---

### Criterio 2: Métodos y Fundamentación de PDI (3.0 puntos)
Evaluación de las herramientas de procesamiento de imágenes incorporadas: binarización, contornos, clasificación geométrica, análisis en dominio frecuencial (Fourier/Sobel) y análisis cromático/gradientes.

- **Sobresaliente (2.7 - 3.0 pts):**
  - Implementa con éxito el pipeline completo de PDI:
    1. Binarización robusta (Otsu / adaptativa con operaciones morfológicas de cierre).
    2. Detección de contornos precisos y segmentación correcta de las 4 esquinas base.
    3. Clasificación morfológica acertada de los lados en `PLANO`, `MACHO` y `HEMBRA`, y discriminación de familias de curvas (`standard`, `circular`, `random`).
    4. Detección de orientación en frecuencia (FFT 2D) o gradientes direccionales ante patrones de rayas periódicas, rectificando rotaciones continuas/leves.
    5. Matcheo de forma geométrica complementaria estricta combinado con continuidad cromática (CIE-Lab) y gradientes (Sobel).
  - Justifica matemáticamente la elección de cada filtro y métrica de disimilitud.
- **Muy Bueno (2.1 - 2.6 pts):**
  - Implementa binarización, clasificación morfológica y comparación de perfiles 1D.
  - Detecta la orientación de rayas o estima rotaciones con precisión adecuada.
  - La combinación de forma y textura es sólida y permite descartar uniones incompatibles.
- **Regular (1.2 - 2.0 pts):**
  - Binarización básica o fallos menores en la detección de esquinas / clasificación de pestañas.
  - Se limita a comparaciones locales sin aprovechar completamente la complementariedad geométrica ni el análisis frecuencial.
- **Insuficiente (0.0 - 1.1 pts):**
  - No implementa binarización ni análisis de contornos, o el método falla en separar la silueta del fondo.

---

### Criterio 3: Calidad de Software, Modularidad y Reproducibilidad (1.5 puntos)

- **Sobresaliente (1.4 - 1.5 pts):**
  - Código modular, limpio, completamente legible, tipado con type hints y documentado con docstrings informativos.
  - Sigue estrictamente la interfaz CLI requerida (`--puzzle-dir`, `--output-json`).
  - Ejecución fluida sin advertencias ni librerías obsoletas; complejidad computacional controlada.
- **Muy Bueno (1.0 - 1.3 pts):**
  - Código bien organizado y fácil de seguir. Cumple con la interfaz de entrada/salida y es reproducible.
- **Regular (0.6 - 0.9 pts):**
  - Funciona pero presenta código desordenado, variables poco descriptivas o dificultad menor para reproducir la ejecución.
- **Insuficiente (0.0 - 0.5 pts):**
  - El código arroja excepciones no controladas al ejecutarse o no respeta el formato `prediction.json`.

---

### Criterio 4: Informe Técnico y Análisis Crítico (1.5 puntos)

- **Sobresaliente (1.4 - 1.5 pts):**
  - Redacción clara, concisa y rigurosa en formato paper/artículo científico.
  - Incluye tablas cuantitativas comparativas bien estructuradas frente al baseline y gráficos de evolución.
  - **Análisis crítico destacado:** Identifica con precisión casos de falla (ej. cielos despejados, patrones periódicos, texturas repetitivas) y propone soluciones fundamentadas.
- **Muy Bueno (1.0 - 1.3 pts):**
  - Informe completo y ordenado. Presenta los resultados experimentales y explica la metodología adecuadamente.
- **Regular (0.6 - 0.9 pts):**
  - Informe descriptivo pero superficial; faltan comparaciones sistemáticas o análisis de causas de error.
- **Insuficiente (0.0 - 0.5 pts):**
  - Informe incompleto, desprolijo o ausente.

---

## 3. Puntos Extra / Bonificaciones (Hasta +1.0 punto adicional)

Se otorgará bonificación sobre la nota final a los grupos que implementen exitosamente:
- **Detección y Anclaje Automático de Píxeles Marcadores:** Identificación de marcadores fiduciales o dead pixels para inicializar la búsqueda a partir de una pieza ancla fija.
- **Deskewing de Alta Precisión Sub-grado:** Estimación angular continua de inclinación con error inferior a $1^\circ$ mediante picos en el espectro de potencia de Fourier 2D.
- **Inferencia de dimensiones:** El algoritmo determina automáticamente las filas y columnas ($R \times C$) óptimas sin recibir el valor a priori.
- **Algoritmos avanzados de optimización:** Algoritmos Genéticos, Programación Entera Mixta (MIP) o Enfriamiento Simulado (Simulated Annealing) bien adaptados al ensamble global.
