"""
Generador de instancias de rompecabezas a partir de imágenes.
Recorta una imagen en una grilla de R x C piezas, las desordena de forma pseudoaleatoria
y opcionalmente aplica rotaciones ortogonales (0°, 90°, 180°, 270°).
Genera el conjunto de piezas y el archivo oficial ground_truth.json.
"""

import os
import argparse
import random
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

try:
    from .utils import load_image, save_image, save_json, rotate_image_clockwise, rotate_image_expand
    from .jigsaw_geometry import JigsawGridGeometry
    from .stripe_analyzer import apply_horizontal_stripes
except ImportError:
    from utils import load_image, save_image, save_json, rotate_image_clockwise, rotate_image_expand
    from jigsaw_geometry import JigsawGridGeometry
    from stripe_analyzer import apply_horizontal_stripes


def infer_grid_dimensions(num_pieces: int) -> Tuple[int, int]:
    """Infiere las dimensiones (rows, cols) más cuadradas posibles para un número de piezas dado."""
    side = int(np.isqrt(num_pieces))
    if side * side == num_pieces:
        return side, side
    # Buscar el par de factores más cercano a un cuadrado
    best_r, best_c = 1, num_pieces
    for r in range(side, 0, -1):
        if num_pieces % r == 0:
            best_r = r
            best_c = num_pieces // r
            break
    return best_r, best_c


class PuzzleGenerator:
    def __init__(
        self,
        image_path: str,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        num_pieces: Optional[int] = None,
        cut_type: str = "jigsaw",
        allow_rotations: bool = False,
        slight_rotation: bool = False,
        max_jitter_degrees: float = 15.0,
        add_stripes: bool = False,
        stripe_period: int = 8,
        stripe_amplitude: float = 0.28,
        add_marker: bool = False,
        marker_pos: Optional[Tuple[int, int]] = None,
        seed: Optional[int] = None
    ):
        self.image_path = image_path
        if num_pieces is not None and (rows is None or cols is None):
            self.rows, self.cols = infer_grid_dimensions(num_pieces)
        else:
            self.rows = rows if rows is not None else 3
            self.cols = cols if cols is not None else 3
            
        self.cut_type = cut_type.lower()
        self.allow_rotations = allow_rotations
        self.slight_rotation = slight_rotation
        self.max_jitter_degrees = max_jitter_degrees
        self.add_stripes = add_stripes
        self.stripe_period = stripe_period
        self.stripe_amplitude = stripe_amplitude
        self.add_marker = add_marker
        self.marker_pos = marker_pos
        self.seed = seed
        
    def generate(self, output_dir: str) -> Dict[str, Any]:
        """
        Ejecuta la generación del rompecabezas y guarda los archivos en output_dir.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            
        os.makedirs(output_dir, exist_ok=True)
        pieces_dir = os.path.join(output_dir, "pieces")
        os.makedirs(pieces_dir, exist_ok=True)
        for f in os.listdir(pieces_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                try:
                    os.remove(os.path.join(pieces_dir, f))
                except OSError:
                    pass
        
        # 1. Cargar imagen original
        full_img = load_image(self.image_path)
        h, w, c = full_img.shape
        
        # 2. Ajustar dimensiones para que sean exactamente divisibles por rows y cols
        piece_h = h // self.rows
        piece_w = w // self.cols
        
        if (self.allow_rotations or self.slight_rotation) and piece_h != piece_w:
            side = min(piece_h, piece_w)
            piece_h = side
            piece_w = side
            
        crop_h = piece_h * self.rows
        crop_w = piece_w * self.cols
        
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        cropped_img = full_img[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()
        
        # 2.1 Aplicar filtro de rayas horizontales si está habilitado
        if self.add_stripes:
            cropped_img = apply_horizontal_stripes(
                cropped_img,
                period=self.stripe_period,
                amplitude=self.stripe_amplitude
            )
            
        # 2.2 Insertar píxel marcador si está habilitado
        marker_info = None
        if self.add_marker:
            if self.marker_pos is not None:
                mx, my = self.marker_pos
            else:
                # Colocar en el centro de una pieza interior (o pieza 0)
                target_r = min(1, self.rows - 1)
                target_c = min(1, self.cols - 1)
                mx = int((target_c + 0.5) * piece_w)
                my = int((target_r + 0.5) * piece_h)
                
            mx = max(2, min(crop_w - 3, mx))
            my = max(2, min(crop_h - 3, my))
            
            # Píxel marcador distintivo (magenta/fucsia brillante)
            marker_color = [255, 0, 180]
            # Parche de 3x3 para garantizar visibilidad e invariancia a interpolación
            cropped_img[my - 1:my + 2, mx - 1:mx + 2] = marker_color
            
            target_piece_r = my // piece_h
            target_piece_c = mx // piece_w
            marker_info = {
                "x": int(mx),
                "y": int(my),
                "grid_row": int(target_piece_r),
                "grid_col": int(target_piece_c),
                "color_rgb": marker_color
            }
        
        save_image(os.path.join(output_dir, "original.png"), cropped_img)
        
        # 3. Recortar piezas en orden natural (Jigsaw o Cuadrada)
        raw_pieces = []
        original_coords = []
        original_edge_types = []
        original_edge_curves = []
        original_offsets = []
        
        if self.cut_type == "jigsaw":
            jigsaw = JigsawGridGeometry(self.rows, self.cols, crop_h, crop_w, seed=self.seed)
            for r in range(self.rows):
                for col in range(self.cols):
                    tile, mask, offset = jigsaw.extract_piece_image(cropped_img, r, col, padding=55)
                    edge_types = jigsaw.get_piece_edge_types(r, col)
                    edge_curves = jigsaw.get_piece_edge_curves(r, col)
                    raw_pieces.append(tile)
                    original_coords.append((r, col))
                    original_edge_types.append(edge_types)
                    original_edge_curves.append(edge_curves)
                    original_offsets.append(offset)
        else:
            for r in range(self.rows):
                for col in range(self.cols):
                    tile = cropped_img[r * piece_h:(r + 1) * piece_h, col * piece_w:(col + 1) * piece_w].copy()
                    edge_types = {"N": "PLANO" if r == 0 else "RECTO",
                                  "S": "PLANO" if r == self.rows - 1 else "RECTO",
                                  "W": "PLANO" if col == 0 else "RECTO",
                                  "E": "PLANO" if col == self.cols - 1 else "RECTO"}
                    edge_curves = {"N": "none", "S": "none", "W": "none", "E": "none"}
                    raw_pieces.append(tile)
                    original_coords.append((r, col))
                    original_edge_types.append(edge_types)
                    original_edge_curves.append(edge_curves)
                    original_offsets.append((col * piece_w, r * piece_h))
                
        num_pieces = len(raw_pieces)
        
        # 4. Desordenar (shuffling)
        shuffled_indices = list(range(num_pieces))
        random.shuffle(shuffled_indices)
        
        # 5. Generar y guardar cada pieza con su rotación opcional
        pieces_info = {}
        solution_grid = [[-1 for _ in range(self.cols)] for _ in range(self.rows)]
        
        for piece_id, orig_idx in enumerate(shuffled_indices):
            orig_r, orig_c = original_coords[orig_idx]
            orig_offset = original_offsets[orig_idx]
            tile = raw_pieces[orig_idx].copy()
            edge_types = original_edge_types[orig_idx]
            edge_curves = original_edge_curves[orig_idx]
            
            applied_ortho = 0
            applied_jitter = 0.0
            current_edge_types = edge_types.copy()
            current_edge_curves = edge_curves.copy()
            
            # Rotación ortogonal (0, 90, 180, 270)
            if self.allow_rotations:
                applied_ortho = random.choice([0, 90, 180, 270])
                tile = rotate_image_clockwise(tile, applied_ortho)
                
                # Rotar orientación de los lados
                steps = (applied_ortho % 360) // 90
                sides = ["N", "E", "S", "W"]
                current_edge_types = {
                    sides[i]: edge_types[sides[(i - steps) % 4]] for i in range(4)
                }
                current_edge_curves = {
                    sides[i]: edge_curves[sides[(i - steps) % 4]] for i in range(4)
                }
                
            # Rotación leve/continua adicional ("rotalas un poco")
            if self.slight_rotation:
                sign = random.choice([1.0, -1.0])
                applied_jitter = round(sign * random.uniform(4.0, max(5.0, self.max_jitter_degrees)), 2)
                # Rotar con expansión dinámica de lienzo y margen negro holgado (cero recorte)
                tile = rotate_image_expand(tile, applied_jitter, padding=40)
                
            piece_filename = f"piece_{piece_id:03d}.png"
            save_image(os.path.join(pieces_dir, piece_filename), tile)
            
            restoration_ortho = (360 - applied_ortho) % 360
            total_applied_rot = round(applied_ortho + applied_jitter, 2)
            
            pieces_info[str(piece_id)] = {
                "piece_id": piece_id,
                "filename": piece_filename,
                "correct_row": orig_r,
                "correct_col": orig_c,
                "applied_rotation": applied_ortho,
                "applied_jitter_degrees": applied_jitter,
                "total_applied_rotation": total_applied_rot,
                "restoration_rotation": restoration_ortho,
                "restoration_jitter_degrees": -applied_jitter,
                "original_edge_types": edge_types,
                "edge_types": current_edge_types,
                "original_edge_curves": edge_curves,
                "edge_curves": current_edge_curves,
                "crop_offset": [int(orig_offset[1]), int(orig_offset[0])],
                "cell_offset": [int(orig_offset[1] - orig_r * piece_h), int(orig_offset[0] - orig_c * piece_w)]
            }
            
            solution_grid[orig_r][orig_c] = piece_id
            
        # 6. Guardar ground truth
        metadata = {
            "source_image": os.path.basename(self.image_path),
            "rows": self.rows,
            "cols": self.cols,
            "num_pieces": num_pieces,
            "piece_shape": [piece_h, piece_w, cropped_img.shape[2]],
            "cell_size": [piece_h, piece_w],
            "cut_type": self.cut_type,
            "allow_rotations": self.allow_rotations,
            "slight_rotation": self.slight_rotation,
            "stripes": {
                "enabled": self.add_stripes,
                "period": self.stripe_period,
                "amplitude": self.stripe_amplitude,
                "orientation_degrees": 0.0 # Rayas horizontales de referencia
            },
            "marker": marker_info,
            "seed": self.seed,
            "solution_grid": solution_grid,
            "pieces_info": pieces_info
        }
        
        save_json(os.path.join(output_dir, "ground_truth.json"), metadata)
        
        print(f"[OK] Rompecabezas generado con éxito en '{output_dir}':")
        print(f"     - Modo de corte: {self.cut_type.upper()} (Encastres Macho/Hembra sobre fondo negro)")
        print(f"     - Dimensiones: {self.rows}x{self.cols} ({num_pieces} piezas)")
        print(f"     - Rotaciones ortogonales: {self.allow_rotations}")
        print(f"     - Rotaciones leves continuas: {self.slight_rotation} (max: {self.max_jitter_degrees}°)")
        print(f"     - Filtro de rayas horizontales: {self.add_stripes}")
        print(f"     - Píxel marcador incorporado: {self.add_marker}")
        print(f"     - Semilla aleatoria: {self.seed}")
        
        return metadata


def create_puzzle(
    image_path: str,
    output_dir: str,
    num_pieces: Optional[int] = None,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    cut_type: str = "jigsaw",
    allow_rotations: bool = False,
    slight_rotation: bool = False,
    max_jitter_degrees: float = 15.0,
    add_stripes: bool = False,
    stripe_period: int = 8,
    stripe_amplitude: float = 0.28,
    add_marker: bool = False,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Función de conveniencia para crear automáticamente un rompecabezas a partir de una imagen.
    """
    generator = PuzzleGenerator(
        image_path=image_path,
        rows=rows,
        cols=cols,
        num_pieces=num_pieces,
        cut_type=cut_type,
        allow_rotations=allow_rotations,
        slight_rotation=slight_rotation,
        max_jitter_degrees=max_jitter_degrees,
        add_stripes=add_stripes,
        stripe_period=stripe_period,
        stripe_amplitude=stripe_amplitude,
        add_marker=add_marker,
        seed=seed
    )
    return generator.generate(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generador de Rompecabezas PDI para TP Final")
    parser.add_argument("--image", type=str, required=True, help="Ruta a la imagen de entrada")
    parser.add_argument("--num-pieces", type=int, default=None, help="Número total de piezas deseadas (ej. 9, 12, 16)")
    parser.add_argument("--rows", type=int, default=None, help="Número de filas de la grilla (opcional)")
    parser.add_argument("--cols", type=int, default=None, help="Número de columnas de la grilla (opcional)")
    parser.add_argument("--cut-type", type=str, default="jigsaw", choices=["jigsaw", "square"], help="Tipo de corte: 'jigsaw' o 'square'")
    parser.add_argument("--rotate", action="store_true", help="Permitir rotaciones ortogonales (0, 90, 180, 270 grados)")
    parser.add_argument("--slight-rotation", action="store_true", help="Aplicar rotaciones leves continuas ('rotalas un poco')")
    parser.add_argument("--jitter-degrees", type=float, default=15.0, help="Máximo ángulo en grados para rotaciones leves")
    parser.add_argument("--add-stripes", action="store_true", help="Aplicar filtro de rayas horizontales periódicas")
    parser.add_argument("--stripe-period", type=int, default=8, help="Período espacial de las rayas en píxeles")
    parser.add_argument("--add-marker", action="store_true", help="Insertar un píxel marcador distintivo en la imagen completa")
    parser.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria para reproducibilidad (default: 42)")
    parser.add_argument("--output", type=str, required=True, help="Directorio donde guardar el rompecabezas")
    
    args = parser.parse_args()
    
    create_puzzle(
        image_path=args.image,
        output_dir=args.output,
        num_pieces=args.num_pieces,
        rows=args.rows,
        cols=args.cols,
        cut_type=args.cut_type,
        allow_rotations=args.rotate,
        slight_rotation=args.slight_rotation,
        max_jitter_degrees=args.jitter_degrees,
        add_stripes=args.add_stripes,
        stripe_period=args.stripe_period,
        add_marker=args.add_marker,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
