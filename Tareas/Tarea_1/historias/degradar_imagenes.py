"""
degradar_imagenes.py

Aplica transformaciones de degradación (oscurecer, sobre-exponer, reducir
contraste) a las imágenes del banco usando el espacio de color YCrCb.

Todas las operaciones se realizan SÓLO sobre el canal Y (luminancia),
dejando intactos Cr y Cb (crominancia), lo que preserva los colores
originales al reconvertir a BGR.

Cada imagen recibe aleatoriamente:
  - Ninguna transformación  (se copia tal cual)
  - Una transformación
  - Combinación de varias (ej: oscurecer + bajo contraste)

Uso:
    python degradar_imagenes.py
    python degradar_imagenes.py --entrada otra/ruta --salida otra/salida
    python degradar_imagenes.py --seed 42 --verbose
"""

import argparse
import random
import cv2
import numpy as np
from itertools import combinations
from pathlib import Path

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Transformaciones (operan en YCrCb sobre el canal Y)
# ---------------------------------------------------------------------------

def oscurecer(imagen: np.ndarray, factor: float = 0.35) -> np.ndarray:
    """Reduce la luminancia multiplicando Y por factor (< 1)."""
    ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = np.clip(y.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def sobreexponer(imagen: np.ndarray, factor: float = 1.65) -> np.ndarray:
    """Satura la luminancia multiplicando Y por factor (> 1)."""
    ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = np.clip(y.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


def reducir_contraste(imagen: np.ndarray, alpha: float = 65) -> np.ndarray:
    """
    Comprime el rango dinámico del canal Y hacia el punto medio (128).

        new_Y = 128 + (Y - 128) * alpha    (alpha < 1 → bajo contraste)
    """
    ycrcb = cv2.cvtColor(imagen, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = np.clip(128.0 + ((y.astype(np.float32) - 128.0) / 128) * alpha, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)


TRANSFORMACIONES: dict[str, callable] = {
    "oscurecer": oscurecer,
    "sobreexponer": sobreexponer,
    "bajo_contraste": reducir_contraste,
}

# Nota: oscurecer y sobreexponer son opuestos; combinarlos no tiene sentido
# visual, por eso los excluimos de las combinaciones mixtas.
COMBINACIONES_VALIDAS: list[tuple[str, ...]] = [
    # sin transformación
    (),
    # individuales
    ("oscurecer",),
    ("sobreexponer",),
    ("bajo_contraste",),
    # # dobles compatibles
    ("oscurecer", "bajo_contraste"),
    ("sobreexponer", "bajo_contraste"),
]

# Pesos de muestreo para cada combinación (suma no necesita ser 1)
PESOS = [
    0.10,   # sin cambio
    0.20,   # oscurecer
    0.20,   # sobreexponer
    0.20,   # bajo_contraste
    0.15,  # oscurecer + bajo_contraste
    0.15,  # sobreexponer + bajo_contraste
]


# ---------------------------------------------------------------------------
# Lógica de procesamiento
# ---------------------------------------------------------------------------

def aplicar_combinacion(imagen: np.ndarray, nombres: tuple[str, ...]) -> np.ndarray:
    resultado = imagen.copy()
    for nombre in nombres:
        resultado = TRANSFORMACIONES[nombre](resultado)
    return resultado


def etiqueta(nombres: tuple[str, ...]) -> str:
    return " + ".join(nombres) if nombres else "sin_cambio"


def recolectar_imagenes(directorio: Path) -> list[Path]:
    return sorted(
        p for p in directorio.rglob("*")
        if p.suffix.lower() in EXTENSIONES
    )


def procesar(entrada: Path, salida: Path, seed: int | None, verbose: bool) -> None:
    imagenes = recolectar_imagenes(entrada)
    if not imagenes:
        print(f"No se encontraron imágenes en: {entrada}")
        return

    rng = random.Random(seed)
    contadores: dict[str, int] = {etiqueta(c): 0 for c in COMBINACIONES_VALIDAS}

    for ruta_src in imagenes:
        combinacion = rng.choices(COMBINACIONES_VALIDAS, weights=PESOS, k=1)[0]
        label = etiqueta(combinacion)

        img = cv2.imread(str(ruta_src))
        if img is None:
            print(f"  [WARN] No se pudo leer: {ruta_src.name}")
            continue

        img_result = aplicar_combinacion(img, combinacion)

        ruta_relativa = ruta_src.relative_to(entrada)
        ruta_dst = salida / ruta_relativa
        ruta_dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ruta_dst), img_result)

        contadores[label] += 1
        if verbose:
            print(f"  [{label:>30}]  {ruta_relativa}")

    total = sum(contadores.values())
    print(f"\nProcesadas {total} imágenes:")
    for label, cantidad in contadores.items():
        print(f"  {label:>30}: {cantidad:>4}")
    print(f"\nResultados guardados en: {salida}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    base = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Degrada imágenes usando el espacio YCrCb.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=base / "imagenes",
        help="Carpeta raíz con las imágenes originales. (default: ./imagenes)",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=base / "imagenes_modificadas",
        help="Carpeta destino. (default: ./imagenes_modificadas)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla para reproducibilidad (default: aleatorio).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra la transformación aplicada a cada archivo.",
    )
    args = parser.parse_args()

    print(f"Entrada : {args.entrada}")
    print(f"Salida  : {args.salida}")
    print(f"Seed    : {args.seed if args.seed is not None else 'aleatorio'}\n")

    procesar(args.entrada, args.salida, args.seed, args.verbose)


if __name__ == "__main__":
    main()
