"""
Demostración Integral de Punta a Punta para el TP Final de Rompecabezas PDI.
Ejecuta de forma secuencial y automatizada:
1. Generación de un rompecabezas de prueba a partir de una imagen.
2. Ejecución del solver baseline provisto por la cátedra.
3. Evaluación cuantitativa y generación del reporte visual comparativo.

Uso:
    python demo.py
"""

import os
import sys

# Agregar la raíz de TP FINAL al sys.path para importación limpia
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.generator import PuzzleGenerator
from src.baseline_solver import BaselineSolver
from src.evaluator import PuzzleEvaluator


def main():
    print("=" * 70)
    print("      DEMOSTRACIÓN INTEGRAL: FRAMEWORK TP FINAL - ROMPECABEZAS PDI     ")
    print("=" * 70)
    
    # 1. Rutas
    repo_root = os.path.dirname(CURRENT_DIR)
    sample_image = os.path.join(repo_root, "imagenes", "casita.jpg")
    if not os.path.isfile(sample_image):
        # Fallback a otra imagen si no existe
        alt_images = [f for f in os.listdir(os.path.join(repo_root, "imagenes")) if f.endswith((".jpg", ".png"))]
        if alt_images:
            sample_image = os.path.join(repo_root, "imagenes", alt_images[0])
        else:
            raise FileNotFoundError("No se encontró ninguna imagen en la carpeta 'imagenes'.")
            
    demo_output_dir = os.path.join(CURRENT_DIR, "demo_run")
    print(f"\n[PASO 1] Generando rompecabezas de prueba 3x3 avanzado:")
    print("         - Encastres específicos: Macho/Hembra (Standard, Circular, Random)")
    print("         - Filtro de rayas horizontales periódicas")
    print("         - Rotaciones leves continuas ('rotalas un poco')")
    print("         - Píxel marcador distintivo (fiducial marker)")
    print(f"         Imagen base: '{sample_image}'")
    
    generator = PuzzleGenerator(
        image_path=sample_image,
        rows=3,
        cols=3,
        cut_type="jigsaw",
        allow_rotations=False,
        slight_rotation=True,
        max_jitter_degrees=10.0,
        add_stripes=True,
        stripe_period=8,
        stripe_amplitude=0.32,
        add_marker=True,
        seed=101
    )
    generator.generate(demo_output_dir)
    
    print("\n[PASO 2] Ejecutando Solver Baseline de la cátedra...")
    solver = BaselineSolver(demo_output_dir, color_space="LAB")
    grid, rotations = solver.solve()
    
    pred_json_path = os.path.join(demo_output_dir, "prediction.json")
    reconstructed_img_path = os.path.join(demo_output_dir, "reconstruction_baseline.png")
    solver.save_solution(grid, rotations, pred_json_path, reconstructed_img_path)
    
    print("\n[PASO 3] Evaluando resultados con el Evaluador Oficial...")
    gt_json_path = os.path.join(demo_output_dir, "ground_truth.json")
    visual_report_path = os.path.join(demo_output_dir, "eval_report.png")
    
    evaluator = PuzzleEvaluator(gt_json_path)
    results = evaluator.evaluate(pred_json_path, save_visual_path=visual_report_path)
    
    print("\n" + "=" * 70)
    print("                   RESUMEN EJECUTIVO DE LA DEMO                     ")
    print("=" * 70)
    print(f" Neighbor Accuracy alcanzado por Baseline: {results['neighbor']['neighbor_accuracy']}%")
    print(f" Componente conexa mayor:                  {results['lcc']['largest_connected_component']}/9 piezas")
    print(f" Reporte visual generado en:\n   -> {visual_report_path}")
    print("=" * 70)
    print("\n[ÉXITO] Todo el pipeline del TP Final está funcionando correctamente.\n")


if __name__ == "__main__":
    main()
