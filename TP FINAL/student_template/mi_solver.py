"""
=============================================================================
TRABAJO PRÁCTICO FINAL - PROCESAMIENTO DIGITAL DE IMÁGENES
Plantilla Oficial del Estudiante: Rompecabezas con Encastres (Macho / Hembra)
=============================================================================

Integrantes del Grupo:
- [Nombre y Apellido - Legajo]
- [Nombre y Apellido - Legajo]

El flujo de trabajo requerido por la cátedra comprende:
  ---------------------------------------------------------------------------
  PASO 1: BINARIZACIÓN LIMPIA DE LA PIEZA
          - Separar la silueta de la pieza (255) del fondo negro puro (0).
          - Aplicar umbralización y cierre/relleno morfológico para no crear falsos huecos.
  ---------------------------------------------------------------------------
  PASO 2: DETECCIÓN DE CONTORNOS Y ESQUINAS NOMINALES
          - Extraer el contorno exterior con cv2.findContours.
          - Detectar las 4 esquinas base (TL, TR, BR, BL).
          - Segmentar el contorno en los 4 bordes: Norte, Sur, Este y Oeste.
  ---------------------------------------------------------------------------
  PASO 3: SEÑALES 1D Y CLASIFICACIÓN TOPOLÓGICA (Fourier & Álgebra Lineal)
          - Representar cada borde como una señal 1D s(t) de desviación perpendicular.
          - Clasificar cada borde en 'PLANO', 'MACHO' o 'HEMBRA'.
          - Clasificar la pieza según su número de bordes planos:
            * ESQUINA: 2 lados planos perpendiculares (exactamente 4 piezas en la grilla).
            * BORDE: 1 lado plano (marco perimetral).
            * INTERIOR: 0 lados planos (relleno central).
  ---------------------------------------------------------------------------
  PASO 4: MATRIZ DE CORRELACIÓN POR PRODUCTO INTERNO (Tipo Fourier)
          - Para dos lados enfrentados (uno MACHO y uno HEMBRA), calcular:
            rho = <s_A, -s_B(1-t)> / (||s_A|| * ||s_B||)
          - Da 1.0 para encastre exacto, ~0.0 para encastres desfasados/ortogonales,
            y <= 0 para colisiones (macho-macho o hembra-hembra).
          - Combinar opcionalmente con color (CIE-Lab) y gradientes (Sobel).
  ---------------------------------------------------------------------------
  PASO 5: RECONSTRUCCIÓN INTUITIVA
          - Ensamblar la grilla aprovechando las restricciones topológicas
            (Esquinas -> Marco Perimetral -> Interior) maximizando la correlación acumulada.
  ---------------------------------------------------------------------------
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

# Permitir importar utilidades de la cátedra
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from src.utils import (
        load_image,
        save_image,
        save_json,
        rotate_image_clockwise,
        rotate_image_expand,
        assemble_puzzle
    )
    from src.shape_matcher import (
        binarize_piece,
        extract_external_contour,
        detect_corners_and_split_sides,
        compute_jigsaw_shape_compatibility,
        analyze_piece_shape
    )
    from src.stripe_analyzer import (
        detect_stripe_orientation,
        rectify_piece_rotation
    )
    from src.affinity_matcher import compute_all_pairwise_relations
    from src.reconstruction import reconstruct_from_relations
except ImportError:
    pass


class MiSolverRompecabezas:
    def __init__(self, puzzle_dir: str):
        self.puzzle_dir = puzzle_dir
        self.pieces = {} # {piece_id: np.ndarray (RGB)}
        self.rows = 0
        self.cols = 0
        self.allow_rotations = False
        self._load_pieces()

    def _load_pieces(self) -> None:
        """Carga todas las piezas del rompecabezas desde la carpeta pieces/."""
        pieces_dir = os.path.join(self.puzzle_dir, "pieces")
        if not os.path.isdir(pieces_dir):
            raise FileNotFoundError(f"No se encontró la carpeta: {pieces_dir}")
            
        for f in os.listdir(pieces_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                p_id = int(f.replace("piece_", "").split(".")[0])
                bgr = cv2.imread(os.path.join(pieces_dir, f))
                self.pieces[p_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
        num_pieces = len(self.pieces)
        side = int(np.sqrt(num_pieces))
        self.rows = side
        self.cols = side
        
        gt_path = os.path.join(self.puzzle_dir, "ground_truth.json")
        if os.path.isfile(gt_path):
            with open(gt_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
                self.rows = meta.get("rows", self.rows)
                self.cols = meta.get("cols", self.cols)
                self.allow_rotations = meta.get("allow_rotations", False)

    # =========================================================================
    # TODO 1: BINARIZACIÓN DE LA PIEZA (FONDO NEGRO vs. PIEZA BLANCA)
    # =========================================================================
    def binarizar(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        [TODO 1]: Separa la pieza completa (255) del fondo negro (0).
        Sugerencia:
        - Convertir a escala de grises.
        - Aplicar cv2.threshold con cv2.THRESH_OTSU.
        - Aplicar operaciones morfológicas (cv2.morphologyEx) para cerrar micro-huecos.
        """
        # Implementación base:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    # =========================================================================
    # TODO 2: DETECCIÓN DE CONTORNOS Y SEGMENTACIÓN DE LADOS
    # =========================================================================
    def extraer_contorno_y_esquinas(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        [TODO 2]: Extrae el contorno exterior y detecta las 4 esquinas nominales.
        Utilizar cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE).
        Retorna:
            - contour: puntos (N, 2) del contorno exterior.
            - corners: diccionario con las 4 esquinas {'TL', 'TR', 'BR', 'BL'}.
        """
        # Pueden utilizar la implementación provista por la cátedra o personalizarla:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        main_contour = max(contours, key=cv2.contourArea).squeeze(axis=1)
        
        # Detección de esquinas extremas
        mid_x, mid_y = np.mean(main_contour, axis=0)
        q_tl = [p for p in main_contour if p[0] <= mid_x and p[1] <= mid_y]
        q_tr = [p for p in main_contour if p[0] >= mid_x and p[1] <= mid_y]
        q_br = [p for p in main_contour if p[0] >= mid_x and p[1] >= mid_y]
        q_bl = [p for p in main_contour if p[0] <= mid_x and p[1] >= mid_y]
        
        c_tl = min(q_tl, key=lambda p: p[0]**2 + p[1]**2)
        c_tr = min(q_tr, key=lambda p: (p[0] - binary_mask.shape[1])**2 + p[1]**2)
        c_br = min(q_br, key=lambda p: (p[0] - binary_mask.shape[1])**2 + (p[1] - binary_mask.shape[0])**2)
        c_bl = min(q_bl, key=lambda p: p[0]**2 + (p[1] - binary_mask.shape[0])**2)
        
        corners = {"TL": c_tl, "TR": c_tr, "BR": c_br, "BL": c_bl}
        return main_contour, corners

    # =========================================================================
    # TODO 3: CLASIFICACIÓN DE ENCASTRES Y CURVAS ESPECÍFICAS
    # =========================================================================
    def clasificar_lados(self, img_rgb: np.ndarray) -> Dict[str, Any]:
        """
        [TODO 3]: Clasifica cada lado (N, E, S, W) en:
        - Tipo morfológico: 'PLANO', 'MACHO' o 'HEMBRA'.
        - Función de curva específica:
            * 'standard': bulbo clásico con cuello cosenoidal.
            * 'circular': saliente abovedada/semicircular.
            * 'random': saliente asimétrica / ondulada.
        """
        return analyze_piece_shape(img_rgb)

    # =========================================================================
    # TODO 4: DETECCIÓN DE ORIENTACIÓN POR RAYAS Y RECTIFICACIÓN (DESKEWING)
    # =========================================================================
    def estimar_orientacion_rayas(self, img_rgb: np.ndarray) -> float:
        """
        [TODO 4]: Estima el ángulo de inclinación de la pieza respecto al filtro de rayas horizontales.
        Técnicas recomendadas:
        - Dominio espacial: Histograma ponderado de direcciones del gradiente Sobel (arctan2(Gy, Gx)).
        - Dominio frecuencial: Transformada 2D de Fourier (np.fft.fft2) y detección de picos espectrales.
        Retorna el ángulo en grados [-90, +90] respecto a la horizontal.
        """
        return detect_stripe_orientation(img_rgb)

    def rectificar_inclinacion_leve(self, img_rgb: np.ndarray, angulo_grados: float) -> np.ndarray:
        """
        Rota la pieza alrededor de su centro para enderezarla a posición horizontal.
        """
        rectificada, _ = rectify_piece_rotation(img_rgb, -angulo_grados)
        return rectificada

    # =========================================================================
    # TODO 5: MATRIZ DE RELACIÓN (FORMA + TEXTURA/COLOR + ORIENTACIÓN DE RAYAS)
    # =========================================================================
    def calcular_matrices_de_relacion(self) -> Dict[str, Any]:
        """
        [TODO 5]: Construye la matriz de costo/afinidad para todos los pares.
        Reglas clave:
        - Regla dura: MACHO solo encaja con HEMBRA.
        - Compatibilidad de funciones de curva: bulbo standard con standard, circular con circular.
        - Coherencia de rayas: ambas piezas deben compartir la misma orientación de modulación.
        - Continuidad cromática (CIE-Lab) y coherencia de gradientes (Sobel).
        """
        print("   [PDI] Calculando compatibilidad morfológica de encastres y continuidad...")
        return compute_all_pairwise_relations(self.pieces, allow_rotations=self.allow_rotations)

    # =========================================================================
    # TODO 6: RECONSTRUCCIÓN MEDIANTE GREEDY / BACKTRACKING
    # =========================================================================
    def resolver(self) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        [TODO 6]: Ensambla el rompecabezas a partir de las matrices de relación.
        Aplica Búsqueda Voraz guiada por incertidumbre (MRV) y Backtracking acotado.
        """
        # Rectificar piezas si tienen inclinaciones leves
        self.tilts = {}
        for p_id, tile in self.pieces.items():
            tilt = self.estimar_orientacion_rayas(tile)
            self.tilts[p_id] = tilt
            is_ortho = (abs(tilt) < 2.0) or (abs(abs(tilt) - 90.0) < 5.0)
            if not is_ortho:
                self.pieces[p_id] = self.rectificar_inclinacion_leve(tile, tilt)
                
        relations = self.calcular_matrices_de_relacion()
        
        print("   [BÚSQUEDA] Ensamblando grilla con Búsqueda Voraz + Backtracking...")
        grid, rotations = reconstruct_from_relations(
            relations=relations,
            rows=self.rows,
            cols=self.cols,
            beam_width=3,
            max_backtracks=500
        )
        return grid, rotations

    def export_prediction(self, output_path: str, grid: list, rotations: dict) -> None:
        """Exporta el archivo prediction.json para el evaluador oficial."""
        data = {
            "grid": grid,
            "rotations": {str(k): int(v) for k, v in rotations.items()},
            "continuous_tilts": {str(k): round(float(v), 2) for k, v in getattr(self, "tilts", {}).items()}
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[OK] Predicción guardada exitosamente en: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Solver de Rompecabezas PDI con Encastres - Plantilla Alumnos")
    parser.add_argument("--puzzle-dir", type=str, required=True, help="Carpeta del rompecabezas")
    parser.add_argument("--output-json", type=str, default="prediction.json", help="Archivo de predicción de salida")
    
    args = parser.parse_args()
    solver = MiSolverRompecabezas(args.puzzle_dir)
    grid, rotations = solver.resolver()
    solver.export_prediction(args.output_json, grid, rotations)


if __name__ == "__main__":
    main()
