# Trabajo Práctico Final: Resolución Automatizada de Rompecabezas mediante PDI
**Cátedra de Procesamiento Digital de Imágenes (PDI)**  
**Modalidad:** Grupal (hasta 2 integrantes) o Individual  

---

## 1. Introducción y Motivación

La reconstrucción de imágenes fragmentadas a partir de piezas desordenadas es un problema clásico y desafiante de visión por computadora y procesamiento digital de imágenes. Posee aplicaciones directas de alto impacto en el mundo real:
- **Arqueología digital:** Restauración de frescos, mosaicos, cerámicas y manuscritos antiguos fragmentados.
- **Ciencias forenses e inteligencia:** Reconstrucción automatizada de documentos triturados (*shredded documents*).
- **Medicina y Biología:** Ensamblado de cortes histológicos y mosaicos de imágenes de microscopía confocal o electrónica.
- **Astronomía y Teledetección:** Composición de mosaicos satelitales (*stitching*) con solapamientos y condiciones de iluminación heterogéneas.

El objetivo de este trabajo práctico es que los estudiantes diseñen, implementen y evalúen un sistema capaz de resolver rompecabezas rectangulares basándose exclusivamente en el análisis de las señales visuales presentes en las piezas (color, bordes, gradientes y texturas).

---

## 2. Formulación del Problema

Se provee un conjunto de $N = R \times C$ piezas cuadradas o rectangulares pertenecientes a una imagen original desconocida. Las piezas han sido:
1. Recortadas de forma contigua en una cuadrícula regular de $R$ filas y $C$ columnas.
2. Desordenadas aleatoriamente (a cada pieza se le asigna un identificador numérico `piece_id`).
3. Opcionalmente rotadas en múltiplos de 90° ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).

El sistema desarrollado por el grupo debe determinar:
- La posición de cada pieza en la grilla reconstructora: celda $(r, c)$ para $0 \le r < R$ y $0 \le c < C$.
- La rotación horaria aplicada a cada pieza para orientarla correctamente.

## 3. Características de las Piezas, Encastres y Señales Visuales Sintéticas

Las piezas del rompecabezas cuentan con **propiedades geométricas y señales visuales de procesamiento avanzado**:
1. **Silueta y Encastres Geométricos:**
   - Cada pieza es una silueta poligonal irregular recortada sobre un **fondo negro puro** `(0, 0, 0)`.
   - Los bordes internos de cada pieza poseen encastres de tipo **Macho** (pestaña que sobresale hacia afuera) o **Hembra** (hendidura o hueco que entra hacia adentro).
   - Los bordes perimetrales exteriores de la imagen original son **Planos** (líneas rectas sin encastre).
   - **Diversidad de Funciones de Curva Específicas:** Los encastres no son idénticos; se generan utilizando distintas funciones analíticas:
     * `'standard'`: Bulbo cosenoidal clásico con cuello estrecho.
     * `'circular'`: Saliente redondeada / semicircular abovedada.
     * `'random'`: Saliente asimétrica u ondulada con armónicos impares.
   - **Regla física de encastre:** Un borde `MACHO` únicamente puede encajar con un borde `HEMBRA` que comparta exactamente la misma función de curva geométrica complementaria.

2. **Filtro de Rayas Horizontales Periódicas (Análisis de Frecuencia y Orientación):**
   - Ciertas instancias de rompecabezas incluyen un patrón de modulación periódica horizontal ($I(y, x) \cdot (1 - \alpha \sin^2(\pi y / T))$).
   - Cuando una pieza está rotada o inclinada, las rayas cambian de orientación respecto a la horizontal.
   - Los alumnos deben aprovechar este patrón para estimar la orientación de la pieza mediante **Transformada 2D de Fourier (FFT)** o **histogramas angulares de gradientes Sobel**, permitiendo rectificar la orientación antes del ensamble.

3. **Rotaciones Leves y Continuas ("Rotalas un poco"):**
   - Además de las rotaciones ortogonales (0°, 90°, 180°, 270°), algunas piezas pueden presentar inclinaciones leves continuas ($\pm 5^\circ$ a $\pm 15^\circ$).
   - Los estudiantes deben implementar un paso de **enderezado (deskewing)** geométrico previo al matcheo de bordes.

4. **Píxel Marcador Distintivo (Pieza Ancla):**
   - Algunas instancias incluyen un marcador calibrado (píxel o parche de 3x3 de color distintivo de alto contraste).
   - La detección de este marcador permite identificar una pieza clave y utilizarla como semilla/ancla de la reconstrucción.

---

## 4. Ejes Obligatorios de Procesamiento Digital de Imágenes

Se requiere que los estudiantes implementen un pipeline riguroso de PDI dividido en las siguientes etapas:

1. **Binarización y Segmentación de la Silueta:**
   - Aislar la pieza completa (máscara binaria blanca con valor $255$) del fondo exterior (negro con valor $0$).
   - Aplicar métodos de umbralización como el algoritmo de **Otsu** (`cv2.THRESH_OTSU`) o umbralización adaptativa.
   - Limpieza de micro-huecos mediante **morfología matemática** (operaciones de cierre morfológico con `cv2.morphologyEx`).

2. **Extracción y Segmentación de Contornos:**
   - Detección de los bordes externos de la silueta mediante algoritmos de seguimiento de contornos (`cv2.findContours`).
   - Identificación de las 4 esquinas de la grilla rectangular base de la pieza.
   - Segmentación del contorno continuo en sus 4 lados orientados: Norte, Sur, Este y Oeste.

3. **Clasificación Morfológica de Encastres y Funciones de Curva:**
   - Medir la desviación y signo de la curva de cada lado respecto a la recta base que une sus esquinas.
   - Clasificar cada uno de los 4 lados en:
     - `PLANO`: Desviación despreciable ($\le 5\%$ del largo del borde).
     - `MACHO`: Protrusión saliente hacia afuera de la pieza.
     - `HEMBRA`: Hendidura entrante hacia el interior de la pieza.
   - Discriminar el tipo de perfil geométrico (`standard`, `circular`, `random`) mediante características de forma (plenitud, asimetría, momentos).

4. **Detección de Orientación por Frecuencia / Gradientes y Deskewing:**
   - En piezas con modulación o rayas, calcular la Transformada 2D de Fourier (`np.fft.fft2`) o gradientes espaciales de Sobel (`np.arctan2(Gy, Gx)`).
   - Estimar el ángulo de inclinación $\theta$ de la pieza respecto al patrón horizontal de referencia.
   - Aplicar una transformación afín de rotación (`cv2.warpAffine`) para rectificar la pieza a su orientación ortogonal antes del matcheo de bordes.

5. **Matcheo Geométrico de Encastres y Afinidad Visual:**
   - Poda dura: Descartar inmediatamente combinaciones imposibles (ej. Macho con Macho, Hembra con Hembra, perfiles de distinta familia o bordes Planos hacia el interior).
   - Medir la distancia cuadrática media (MSE) entre la curva 1D del macho y la curva complementaria de la hembra.
   - Integrar la continuidad cromática en espacio perceptual **CIE-Lab** y coherencia de gradientes espaciales (**Sobel**).

6. **Algoritmo de Reconstrucción (Búsqueda Voraz con Backtracking y Anclaje):**
   - Ensamblar la grilla de piezas basándose en la matriz de afinidad geométrica y visual.
   - Soportar la selección de pieza ancla (si se detecta el píxel marcador distintivo) como raíz de la búsqueda.
   - Búsqueda informada guiada por incertidumbre (MRV - Most Constrained Variable) con poda de backtracking acotada.

---

## 5. Métricas Oficiales de Evaluación

Las soluciones se evaluarán con el script provisto por la cátedra (`src/evaluator.py`), el cual calcula las métricas estándar de la literatura científica (Gallagher 2012):

1. **Direct Placement Accuracy (Exactitud Directa):**
   $$\text{Acc}_{\text{direct}} = \frac{\text{Piezas en posición exacta } (r, c) \text{ y rotación correcta}}{\text{Total de piezas}} \times 100\%$$
2. **Neighbor Pair Accuracy (Exactitud de Pares de Vecindad):**
   $$\text{Acc}_{\text{neighbor}} = \frac{\text{Pares de piezas vecinas (H y V) asignadas correctamente}}{\text{Total de pares de bordes adyacentes}} \times 100\%$$
   *Nota:* Esta métrica es invariante a traslaciones globales y es la métrica principal para evaluar la calidad del emparejamiento de bordes.
3. **Largest Connected Component (LCC):**
   Porcentaje del tamaño del fragmento conexo reconstruido correctamente más grande.

---

## 6. Estructura de Entregables

Cada grupo debe entregar:

1. **Código Fuente:**
   - Basado en la plantilla provista en `student_template/`.
   - Script ejecutable `mi_solver.py` que reciba los parámetros:
     ```bash
     python mi_solver.py --puzzle-dir <ruta_al_puzzle> --output-json <ruta_prediccion.json>
     ```
   - Archivo `requirements.txt` o indicación de dependencias.
2. **Resultados Cuantitativos:**
   - Archivos `prediction.json` generados sobre el conjunto de prueba público (`dataset_ejemplos/`).
3. **Informe Técnico (PDF, máximo 6 páginas):**
   - Formato estándar de dos columnas (estilo IEEE/SADIO).
   - Estructura:
     1. Resumen (Abstract).
     2. Descripción de la metodología de PDI implementada (espacios de color, cálculo de bordes, función de costo).
     3. Algoritmo de ensamble y resolución del puzzle.
     4. Resultados experimentales y tablas comparativas frente al Baseline de la cátedra.
     5. Análisis cualitativo y discusión de fallos (¿en qué tipo de imágenes falló el método y por qué?).
     6. Conclusiones y líneas de trabajo futuro.

---

## 7. Cronograma y Fechas Clave

- **Publicación del Trabajo Práctico:** [Fecha de Lanzamiento]
- **Sesión de Consultas y Validación de Baseline:** [Semana +2]
- **Entrega Final de Código e Informe:** [Fecha Límite Oficial]
- **Defensas Orales / Demostraciones en Vivo:** [Semana de Cierre]
