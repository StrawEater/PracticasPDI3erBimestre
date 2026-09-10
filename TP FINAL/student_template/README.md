# Plantilla de Inicio para el Estudiante - TP Final PDI

¡Bienvenidos al Trabajo Práctico Final de **Procesamiento Digital de Imágenes**!

Este directorio contiene la estructura base para comenzar a desarrollar su propia solución para el problema de reconstrucción de rompecabezas.

---

## Contenido de este directorio

1. **`mi_solver.py`**: Esqueleto en Python modularizado con funciones clave y comentarios `TODO` listos para completar.
2. **`notebook_guia.ipynb`**: Notebook interactivo que explica conceptualmente el problema, cómo inspeccionar los bordes de las piezas, medir diferencias de color y evaluar los primeros resultados.

---

## Cómo empezar

### 1. Explorar el Notebook Guía
Abran `notebook_guia.ipynb` en Jupyter Lab / VS Code. Les mostrará:
- Cómo cargar las piezas desde disco.
- Cómo extraer las tiras de bordes (Norte, Sur, Este, Oeste).
- Cómo calcular distancias en diferentes espacios de color (RGB vs. CIE-Lab vs. HSV).
- Cómo armar la estructura de salida `prediction.json`.

### 2. Implementar su método en `mi_solver.py`
En `mi_solver.py` deberán diseñar e implementar:
1. **Extracción y representación de características de bordes**:
   - Consideren gradientes (Sobel, Canny), perfiles de intensidad, texturas (LBP), o franjas de más de 1 píxel de espesor para continuidad de patrones.
2. **Medidas de afinidad/compatibilidad**:
   - SSD, SAD, correlación cruzada normalizada (NCC), o distancia Mahalanobis.
3. **Estrategia de búsqueda y ensamble global**:
   - Algoritmo Voraz mejorado, Árbol Generador Mínimo (MST), búsqueda informada con heurísticas, etc.
4. **Manejo de rotaciones (Nivel 2)**:
   - Probar las 4 rotaciones ortogonales posibles para cada pieza.

### 3. Ejecutar su Solver
Para ejecutar su solución sobre uno de los puzzles de prueba:

```bash
python mi_solver.py --puzzle-dir ../dataset_ejemplos/puzzle_3x3_facil --output-json prediction_mi_grupo.json
```

### 4. Evaluar su Desempeño
Pueden autoevaluar su predicción ejecutando el evaluador oficial:

```bash
python ../src/evaluator.py --ground-truth ../dataset_ejemplos/puzzle_3x3_facil/ground_truth.json --prediction prediction_mi_grupo.json --visualize reporte_mi_grupo.png
```

---

## Formato requerido para `prediction.json`

Su solver debe generar un archivo JSON con la siguiente estructura:

```json
{
    "grid": [
        [0, 3, 1],
        [4, 2, 8],
        [7, 5, 6]
    ],
    "rotations": {
        "0": 0,
        "1": 90,
        "2": 0,
        "3": 270,
        "4": 180,
        "5": 0,
        "6": 0,
        "7": 90,
        "8": 0
    }
}
```

- `"grid"`: Matriz $R \times C$ con los identificadores enteros (`piece_id`) colocados en cada posición.
- `"rotations"`: Diccionario que indica la rotación horaria (en grados: `0`, `90`, `180`, `270`) que aplicaron a cada pieza para reconstruir la imagen.
