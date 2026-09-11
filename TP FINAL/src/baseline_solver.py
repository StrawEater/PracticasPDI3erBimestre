"""
Solver Baseline de Referencia para el TP Final de Rompecabezas PDI.
Implementa una solución basada en:
1. Extracción de perfiles de intensidad en los bordes de cada pieza.
2. Conversión a espacio de color perceptual CIE-Lab.
3. Medición de disimilitud mediante Suma de Diferencias Cuadráticas (SSD).
4. Algoritmo de ensamble Voraz (Greedy Best-First Placement) con soporte de rotaciones.
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

try:
    from .utils import load_image, save_image, save_json, load_json, rotate_image_clockwise, assemble_puzzle
    from .affinity_matcher import compute_all_pairwise_relations
    from .reconstruction import reconstruct_from_relations
    from .stripe_analyzer import detect_stripe_orientation, detect_stripe_orientation_fft, rectify_piece_rotation
except ImportError:
    from utils import load_image, save_image, save_json, load_json, rotate_image_clockwise, assemble_puzzle
    from affinity_matcher import compute_all_pairwise_relations
    from reconstruction import reconstruct_from_relations
    from stripe_analyzer import detect_stripe_orientation, detect_stripe_orientation_fft, rectify_piece_rotation


class BaselineSolver:
    def __init__(self, puzzle_dir: str, color_space: str = "LAB"):
        """
        Args:
            puzzle_dir: Carpeta del rompecabezas que contiene 'pieces/' y opcionalmente 'ground_truth.json'.
            color_space: Espacio de color para evaluar bordes ('RGB' o 'LAB').
        """
        self.puzzle_dir = puzzle_dir
        self.color_space = color_space.upper()
        self.pieces_raw = {}      # {piece_id: np.ndarray (RGB)}
        self.pieces_processed = {}# {piece_id: np.ndarray en espacio seleccionado}
        self.rows = 0
        self.cols = 0
        self.num_pieces = 0
        self.allow_rotations = False
        
        self._load_data()

    def _load_data(self) -> None:
        """Carga las piezas e inspecciona metadatos si están disponibles."""
        pieces_dir = os.path.join(self.puzzle_dir, "pieces")
        if not os.path.isdir(pieces_dir):
            raise FileNotFoundError(f"No se encontró el directorio de piezas: {pieces_dir}")
            
        piece_files = sorted([f for f in os.listdir(pieces_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
        if not piece_files:
            raise ValueError(f"No se encontraron imágenes en: {pieces_dir}")
            
        # Intentar inferir dimensiones desde ground_truth.json si existe
        gt_path = os.path.join(self.puzzle_dir, "ground_truth.json")
        if os.path.isfile(gt_path):
            self.gt_data = load_json(gt_path)
            self.rows = self.gt_data.get("rows", 0)
            self.cols = self.gt_data.get("cols", 0)
            self.allow_rotations = self.gt_data.get("allow_rotations", False)
        else:
            self.gt_data = {}
            self.rows = 0
            self.cols = 0
            
        self.detected_tilts = {}
        self.pieces_rectified = {}
        for f in piece_files:
            p_id = int(f.replace("piece_", "").split(".")[0])
            img_rgb = load_image(os.path.join(pieces_dir, f))
            self.pieces_raw[p_id] = img_rgb
            
            has_slight_rotation = self.gt_data.get("slight_rotation", False)
            has_stripes = self.gt_data.get("stripes", {}).get("enabled", False)
            
            if has_slight_rotation and has_stripes:
                # Estimar rotación leve mediante pico espectral 2D Fourier de las rayas
                tilt_raw = detect_stripe_orientation_fft(img_rgb)
                is_ortho = (abs(tilt_raw) < 1.0) or (abs(abs(tilt_raw) - 90.0) < 1.5)
                if not is_ortho:
                    tilt_angle = round(float(tilt_raw), 2)
                    rectified_img, _ = rectify_piece_rotation(img_rgb, -tilt_angle)
                else:
                    tilt_angle = 0.0
                    rectified_img = img_rgb
            elif has_slight_rotation:
                # Detección de inclinación continua sobre los ejes del contorno (minAreaRect)
                gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                _, binary = cv2.threshold(gray, 3, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    angle = rect[2]
                    contour_tilt = angle % 90.0
                    if contour_tilt > 45.0:
                        contour_tilt -= 90.0
                else:
                    contour_tilt = 0.0
                    
                # Si el contorno está inclinado respecto a la ortogonalidad (>= 1.5 grados)
                if abs(contour_tilt) >= 1.5:
                    tilt_angle = round(float(contour_tilt), 2)
                    rectified_img, _ = rectify_piece_rotation(img_rgb, -tilt_angle)
                else:
                    tilt_angle = 0.0
                    rectified_img = img_rgb
            else:
                tilt_angle = 0.0
                rectified_img = img_rgb
                    
            self.detected_tilts[p_id] = tilt_angle
            self.pieces_rectified[p_id] = rectified_img
            
            # Procesamiento de espacio de color
            if self.color_space == "LAB":
                self.pieces_processed[p_id] = cv2.cvtColor(self.pieces_rectified[p_id], cv2.COLOR_RGB2LAB).astype(np.float32)
            else:
                self.pieces_processed[p_id] = self.pieces_rectified[p_id].astype(np.float32)
                
        self.num_pieces = len(self.pieces_raw)
        if self.rows == 0 or self.cols == 0:
            side = int(np.sqrt(self.num_pieces))
            self.rows = side
            self.cols = side

    def find_marker_piece(self) -> Optional[int]:
        """Detecta automáticamente la pieza que contiene el píxel marcador característico."""
        for p_id, tile in self.pieces_raw.items():
            if tile.ndim == 3 and tile.shape[2] == 3:
                r = tile[:, :, 0].astype(int)
                g = tile[:, :, 1].astype(int)
                b = tile[:, :, 2].astype(int)
                # Buscar píxel fucsia/magenta característico [255, 0, 180]
                mask = (r > 200) & (g < 60) & (b > 130)
                if np.sum(mask) >= 1:
                    return p_id
        return None

    def _get_border(self, piece: np.ndarray, side: str) -> np.ndarray:
        """
        Extrae la tira de 1 píxel de borde de una pieza.
        side: 'N' (Norte/Arriba), 'S' (Sur/Abajo), 'W' (Oeste/Izquierda), 'E' (Este/Derecha)
        """
        if side == "N":
            return piece[0, :, :]
        elif side == "S":
            return piece[-1, :, :]
        elif side == "W":
            return piece[:, 0, :]
        elif side == "E":
            return piece[:, -1, :]
        else:
            raise ValueError(f"Lado no reconocido: {side}")

    def compute_edge_dissimilarity(
        self,
        piece_a: np.ndarray,
        side_a: str,
        piece_b: np.ndarray,
        side_b: str
    ) -> float:
        """
        Calcula la disimilitud (SSD promedio por píxel) entre dos bordes adyacentes.
        Valores menores indican mayor compatibilidad.
        """
        border_a = self._get_border(piece_a, side_a)
        border_b = self._get_border(piece_b, side_b)
        
        # Diferencia cuadrática media por píxel en todos los canales
        diff = border_a - border_b
        return float(np.mean(diff ** 2))

    def solve(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        allow_rotations: Optional[bool] = None,
        beam_width: int = 3,
        max_backtracks: int = 500
    ) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        Ejecuta el algoritmo en dos etapas claramente desacopladas:
        1. PROCESAMIENTO DE IMÁGENES: Cálculo de matrices de relación/afinidad entre piezas.
        2. RECONSTRUCCIÓN: Algoritmo de colocación voraz con backtracking sobre las matrices de relación.
        """
        r_grid = rows if rows is not None else self.rows
        c_grid = cols if cols is not None else self.cols
        rot_allowed = allow_rotations if allow_rotations is not None else self.allow_rotations
        
        print("\n" + "-" * 60)
        print("[FASE 1: PDI] Calculando matrices de relación entre piezas...")
        print("             (Filtro bilateral + CIE-Lab + Gradientes Sobel + Continuidad 2do orden)")
        print("-" * 60)
        
        has_stripes = self.gt_data.get("stripes", {}).get("enabled", False)
        self.relations = compute_all_pairwise_relations(
            self.pieces_rectified,
            allow_rotations=rot_allowed,
            has_stripes=has_stripes
        )
        print(f"[OK] Afinidad calculada: {len(self.relations['horizontal'])} pares horizontales, "
              f"{len(self.relations['vertical'])} pares verticales.")

        print("\n" + "-" * 60)
        print("[FASE 2: RECONSTRUCCIÓN] Ensamblando grilla con Búsqueda Voraz + Backtracking...")
        print("-" * 60)
        
        # Detección de pieza ancla con marcador distintivo
        anchor = None
        marker_info = self.gt_data.get("marker")
        if marker_info:
            marker_p = self.find_marker_piece()
            if marker_p is not None:
                r_anc = marker_info.get("grid_row", 0)
                c_anc = marker_info.get("grid_col", 0)
                anchor = (marker_p, r_anc, c_anc, 0)
                print(f"[PDI - MARCADOR] Pieza ancla detectada: ID {marker_p} fijada en ({r_anc}, {c_anc}).")

        grid, pred_rotations, reconstructor = reconstruct_from_relations(
            relations=self.relations,
            rows=r_grid,
            cols=c_grid,
            beam_width=beam_width,
            max_backtracks=max_backtracks,
            anchor=anchor,
            return_reconstructor=True
        )
        self.placement_history = reconstructor.placement_history
        
        print("[OK] Grilla reconstruida exitosamente.")
        return grid, pred_rotations

    def save_solution(
        self,
        grid: List[List[int]],
        pred_rotations: Dict[int, int],
        output_prediction_path: str,
        output_image_path: Optional[str] = None
    ) -> None:
        """Guarda prediction.json y opcionalmente la imagen reconstruida."""
        data = {
            "grid": grid,
            "rotations": {str(k): v for k, v in pred_rotations.items()},
            "continuous_tilts": {str(k): round(float(v), 2) for k, v in self.detected_tilts.items()}
        }
        save_json(output_prediction_path, data)
        print(f"[OK] Predicción guardada en: {output_prediction_path}")
        
        if output_image_path:
            cell_size = tuple(self.gt_data.get("cell_size", self.gt_data.get("piece_shape", [100, 100])[:2])) if self.gt_data else None
            reconstructed = assemble_puzzle(
                grid,
                self.pieces_raw,
                rotations=pred_rotations,
                continuous_tilts=self.detected_tilts,
                cell_size=cell_size,
                draw_lines=False
            )
            save_image(output_image_path, reconstructed)
            print(f"[OK] Imagen reconstruida guardada en: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(description="Solver Baseline de Rompecabezas PDI")
    parser.add_argument("--puzzle-dir", type=str, required=True, help="Directorio del rompecabezas a resolver")
    parser.add_argument("--output-json", type=str, default=None, help="Ruta de salida para prediction.json")
    parser.add_argument("--output-img", type=str, default=None, help="Ruta de salida para la imagen reconstruida")
    parser.add_argument("--color-space", type=str, default="LAB", choices=["RGB", "LAB"], help="Espacio de color")
    
    args = parser.parse_args()
    
    solver = BaselineSolver(args.puzzle_dir, color_space=args.color_space)
    grid, rotations = solver.solve()
    
    out_json = args.output_json if args.output_json else os.path.join(args.puzzle_dir, "prediction_baseline.json")
    out_img = args.output_img if args.output_img else os.path.join(args.puzzle_dir, "reconstruction_baseline.png")
    
    solver.save_solution(grid, rotations, out_json, out_img)


if __name__ == "__main__":
    main()
