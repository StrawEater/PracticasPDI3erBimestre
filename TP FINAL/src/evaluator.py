"""
Evaluador de soluciones para el TP Final de Rompecabezas PDI.
Compara el archivo de predicción del alumno (prediction.json) contra ground_truth.json,
calcula métricas cuantitativas y genera un reporte visual comparativo.
"""

import os
import argparse
from typing import Dict, Any, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from .utils import load_json, load_image, assemble_puzzle, rotate_image_clockwise
    from .metrics import evaluate_puzzle
except ImportError:
    from utils import load_json, load_image, assemble_puzzle, rotate_image_clockwise
    from metrics import evaluate_puzzle


class PuzzleEvaluator:
    def __init__(self, ground_truth_path: str):
        self.gt_path = ground_truth_path
        self.gt_data = load_json(ground_truth_path)
        self.puzzle_dir = os.path.dirname(os.path.abspath(ground_truth_path))
        
    def evaluate(self, prediction_path: str, save_visual_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evalúa un archivo prediction.json y genera métricas y gráfico opcional.
        """
        pred_data = load_json(prediction_path)
        pred_grid = pred_data.get("grid")
        if pred_grid is None:
            raise ValueError("El archivo de predicción debe contener la clave 'grid' con la matriz de piezas.")
            
        pred_rotations_raw = pred_data.get("rotations", {})
        # Normalizar claves de rotación a int
        pred_rotations = {int(k): int(v) for k, v in pred_rotations_raw.items()} if pred_rotations_raw else None
        
        gt_grid = self.gt_data["solution_grid"]
        pieces_info = self.gt_data.get("pieces_info", {})
        gt_rotations = {int(k): v.get("restoration_rotation", 0) for k, v in pieces_info.items()}
        
        # Validar consistencia de dimensiones
        rows_pred = len(pred_grid)
        cols_pred = len(pred_grid[0]) if rows_pred > 0 else 0
        if rows_pred != self.gt_data["rows"] or cols_pred != self.gt_data["cols"]:
            raise ValueError(
                f"Dimensiones de predicción ({rows_pred}x{cols_pred}) no coinciden con ground truth "
                f"({self.gt_data['rows']}x{self.gt_data['cols']})."
            )
            
        # Calcular métricas
        results = evaluate_puzzle(
            pred_grid=pred_grid,
            gt_grid=gt_grid,
            pred_rotations=pred_rotations,
            gt_rotations=gt_rotations
        )
        
        # Métricas avanzadas: error angular de rotación leve / continua
        has_stripes = self.gt_data.get("stripes", {}).get("enabled", False)
        has_slight_rot = self.gt_data.get("slight_rotation", False)
        
        pred_tilts = pred_data.get("continuous_tilts", {})
        if (has_stripes or has_slight_rot) and pred_tilts:
            angular_errors = []
            for p_id_str, pred_tilt in pred_tilts.items():
                p_info = pieces_info.get(p_id_str, {})
                gt_jitter = p_info.get("applied_jitter_degrees", 0.0)
                err = abs(float(pred_tilt) - float(gt_jitter))
                angular_errors.append(err)
            results["mean_angular_error"] = round(float(np.mean(angular_errors)), 2) if angular_errors else 0.0
        
        # Validación de pieza ancla con marcador
        marker_info = self.gt_data.get("marker")
        if marker_info:
            target_r = marker_info.get("grid_row")
            target_c = marker_info.get("grid_col")
            target_p = gt_grid[target_r][target_c] if target_r is not None and target_c is not None else None
            placed_correctly = (pred_grid[target_r][target_c] == target_p) if target_p is not None else False
            results["marker_piece_correct"] = placed_correctly
        
        # Imprimir reporte formateado
        self._print_report(results, prediction_path)
        
        # Generar gráfico visual si se solicita
        if save_visual_path:
            self._generate_visual_report(
                pred_grid, pred_rotations, gt_grid, gt_rotations, results, save_visual_path,
                pred_tilts=pred_tilts
            )
            
        return results

    def _print_report(self, results: Dict[str, Any], pred_path: str) -> None:
        direct = results["direct"]
        neigh = results["neighbor"]
        lcc = results["lcc"]
        
        print("\n" + "=" * 65)
        print("         REPORTE DE EVALUACIÓN OFICIAL - ROMPECABEZAS PDI         ")
        print("=" * 65)
        print(f" Archivo evaluado: {os.path.basename(pred_path)}")
        print("-" * 65)
        print("  1. EXACTITUD DIRECTA (Direct Placement Accuracy):")
        print(f"     - Piezas en posición correcta:   {direct['correct_pieces']}/{direct['total_pieces']} ({direct['direct_position_accuracy']}%)")
        print(f"     - Piezas con orientación OK:     {direct['direct_full_accuracy']}%")
        print("-" * 65)
        print("  2. EXACTITUD DE VECINDAD (Neighbor Pair Accuracy - Métrica Clave):")
        print(f"     - Pares de vecinos correctos:    {neigh['correct_pairs']}/{neigh['total_pairs']} ({neigh['neighbor_accuracy']}%)")
        print(f"     - Adyacencia Horizontal:         {neigh['horizontal_accuracy']}%")
        print(f"     - Adyacencia Vertical:           {neigh['vertical_accuracy']}%")
        print("-" * 65)
        print("  3. COMPONENTE CONEXA MAYOR (Largest Connected Component):")
        print(f"     - Mayor fragmento continuo:      {lcc['largest_connected_component']}/{lcc['total_pieces']} piezas ({lcc['lcc_percentage']}%)")
        if "marker_piece_correct" in results or results.get("mean_angular_error", 0.0) > 0:
            print("-" * 65)
            print("  4. PROCESAMIENTO AVANZADO (Rayas, Rotación Leve y Marcador):")
            if "mean_angular_error" in results:
                print(f"     - Error angular medio (Deskewing): {results['mean_angular_error']}°")
            if "marker_piece_correct" in results:
                print(f"     - Pieza ancla con marcador correcta: {'SÍ' if results['marker_piece_correct'] else 'NO'}")
            print("=" * 65 + "\n")

    def _generate_visual_report(
        self,
        pred_grid: list,
        pred_rotations: dict,
        gt_grid: list,
        gt_rotations: dict,
        results: dict,
        save_path: str,
        pred_tilts: Optional[dict] = None
    ) -> None:
        """
        Carga las piezas y crea una figura comparativa de 3 paneles:
        1. Original (Ground Truth)
        2. Reconstrucción Predicha
        3. Mapa de aciertos/errores de colocación
        """
        pieces_dir = os.path.join(self.puzzle_dir, "pieces")
        if not os.path.isdir(pieces_dir):
            print(f"[AVISO] No se encontró el directorio de piezas en {pieces_dir}. Se omite el reporte visual.")
            return

        # Cargar todas las piezas
        pieces = {}
        for piece_file in os.listdir(pieces_dir):
            if piece_file.endswith((".png", ".jpg", ".jpeg")):
                try:
                    p_id = int(piece_file.replace("piece_", "").split(".")[0])
                    pieces[p_id] = load_image(os.path.join(pieces_dir, piece_file))
                except ValueError:
                    continue

        pieces_info = self.gt_data.get("pieces_info", {})
        cell_size = tuple(self.gt_data.get("cell_size", self.gt_data.get("piece_shape", [100, 100])[:2]))
        gt_tilts = {int(k): float(v.get("applied_jitter_degrees", 0.0)) for k, v in pieces_info.items()}
        pred_tilts_dict = {int(k): float(v) for k, v in pred_tilts.items()} if pred_tilts else None

        # Ensamblar GT (perfectamente orientado y encastrado)
        gt_img = assemble_puzzle(
            gt_grid, pieces,
            rotations=gt_rotations,
            continuous_tilts=gt_tilts,
            pieces_info=pieces_info,
            cell_size=cell_size,
            draw_lines=False
        )
        
        # Ensamblar Predicción (rotado a posición predicha y encastrado de ranuras y salientes)
        pred_img = assemble_puzzle(
            pred_grid, pieces,
            rotations=pred_rotations,
            continuous_tilts=pred_tilts_dict,
            pieces_info=pieces_info,
            cell_size=cell_size,
            draw_lines=False
        )

        # Crear mapa de error visual sobre la imagen predicha ensamblada
        rows = len(gt_grid)
        cols = len(gt_grid[0])
        tile_h, tile_w = cell_size
        error_map = pred_img.copy()

        for r in range(rows):
            for c in range(cols):
                p_pred = pred_grid[r][c]
                p_gt = gt_grid[r][c]
                is_pos_correct = (p_pred == p_gt)
                
                rot_ok = True
                if pred_rotations is not None and gt_rotations is not None:
                    rot_ok = (pred_rotations.get(p_pred, 0) % 360 == gt_rotations.get(p_gt, 0) % 360)
                
                if is_pos_correct and rot_ok:
                    tint = np.array([0, 200, 0], dtype=np.float32) # verde (correcto)
                elif is_pos_correct:
                    tint = np.array([220, 180, 0], dtype=np.float32) # amarillo (rotación errónea)
                else:
                    tint = np.array([220, 0, 0], dtype=np.float32) # rojo (ubicación errónea)
                    
                y0 = r * tile_h
                y1 = min(error_map.shape[0], (r + 1) * tile_h)
                x0 = c * tile_w
                x1 = min(error_map.shape[1], (c + 1) * tile_w)
                
                cell_region = error_map[y0:y1, x0:x1]
                cell_mask = np.any(cell_region > 5, axis=-1)
                if np.any(cell_mask):
                    blended = 0.65 * cell_region[cell_mask].astype(np.float32) + 0.35 * tint
                    cell_region[cell_mask] = np.clip(blended, 0, 255).astype(np.uint8)

        # Crear figura
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        axes[0].imshow(gt_img)
        axes[0].set_title("1. Solución Original (Ground Truth)", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        neigh_acc = results["neighbor"]["neighbor_accuracy"]
        axes[1].imshow(pred_img)
        axes[1].set_title(f"2. Reconstrucción Propuesta\n(Vecindad: {neigh_acc}%)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        
        pos_acc = results["direct"]["direct_position_accuracy"]
        axes[2].imshow(error_map)
        axes[2].set_title(f"3. Mapa de Aciertos\n(Posición directa: {pos_acc}%)", fontsize=12, fontweight="bold")
        axes[2].axis("off")
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Reporte visual guardado en: '{save_path}'")


def main():
    parser = argparse.ArgumentParser(description="Evaluador Oficial de Rompecabezas PDI")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ruta a ground_truth.json")
    parser.add_argument("--prediction", type=str, required=True, help="Ruta a prediction.json")
    parser.add_argument("--visualize", type=str, default=None, help="Ruta para guardar gráfico de evaluación (PNG)")
    
    args = parser.parse_args()
    evaluator = PuzzleEvaluator(args.ground_truth)
    evaluator.evaluate(args.prediction, args.visualize)


if __name__ == "__main__":
    main()
