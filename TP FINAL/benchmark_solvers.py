"""
Script oficial de Benchmark y Evaluación Comparativa para el TP Final de Rompecabezas PDI.
Ejecuta el solver en la suite completa de datasets oficiales:
- puzzle_2x2_intro
- puzzle_3x3_facil
- puzzle_3x3_rotado
- puzzle_4x4_medio
- puzzle_10x10_desafio

Mide tiempos detallados (extracción, afinidad, reconstrucción, total),
calcula métricas cuantitativas formales (Direct Accuracy, Neighbor Accuracy, Error Angular),
y genera animaciones paso a paso en formato GIF.
"""

import os
import sys
import time
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.baseline_solver import BaselineSolver
from src.evaluator import PuzzleEvaluator
from src.puzzle_animator import create_solving_animation


def run_benchmarks(generate_animations: bool = True):
    dataset_base = os.path.join(CURRENT_DIR, "dataset_ejemplos")
    
    datasets = [
        {"name": "puzzle_2x2_intro", "rows": 2, "cols": 2, "fps": 3},
        {"name": "puzzle_3x3_facil", "rows": 3, "cols": 3, "fps": 5},
        {"name": "puzzle_3x3_rotado", "rows": 3, "cols": 3, "fps": 5},
        {"name": "puzzle_4x4_medio", "rows": 4, "cols": 4, "fps": 7},
        {"name": "puzzle_10x10_desafio", "rows": 10, "cols": 10, "fps": 12},
    ]
    
    print("=" * 80)
    print("        BENCHMARK OFICIAL DE SOLVER PDI - EVALUACIÓN Y TIEMPOS        ")
    print("=" * 80)
    
    records = []
    
    for ds in datasets:
        ds_name = ds["name"]
        ds_dir = os.path.join(dataset_base, ds_name)
        gt_path = os.path.join(ds_dir, "ground_truth.json")
        pred_path = os.path.join(ds_dir, "prediction_baseline.json")
        recon_path = os.path.join(ds_dir, "reconstruction_baseline.png")
        gif_path = os.path.join(ds_dir, "resolucion_animada.gif")
        mp4_path = os.path.join(ds_dir, "resolucion_animada.mp4")
        
        if not os.path.isdir(ds_dir) or not os.path.isfile(gt_path):
            print(f"[SKIP] No se encontró el dataset: {ds_name}")
            continue
            
        print(f"\n>> Procesando: {ds_name} ({ds['rows']}x{ds['cols']} = {ds['rows']*ds['cols']} piezas)")
        
        # 1. Carga y preprocesamiento
        t0 = time.time()
        solver = BaselineSolver(ds_dir)
        t_load = time.time() - t0
        
        # 2. Resolución completa
        t_solve_start = time.time()
        grid, rotations = solver.solve()
        t_solve_total = time.time() - t_solve_start
        
        solver.save_solution(grid, rotations, pred_path, recon_path)
        
        # 3. Evaluación de métricas
        evaluator = PuzzleEvaluator(gt_path)
        metrics = evaluator.evaluate(pred_path)
        
        direct_acc = metrics["direct"]["direct_full_accuracy"]
        neighbor_acc = metrics["neighbor"]["neighbor_accuracy"]
        lcc_pct = metrics["lcc"]["lcc_percentage"]
        mean_angle_err = metrics.get("mean_angular_error", 0.0)
        
        # 4. Generación de animación (MP4 y GIF)
        anim_time = 0.0
        if generate_animations and solver.placement_history:
            t_anim_start = time.time()
            c_size = tuple(solver.gt_data.get("cell_size")) if solver.gt_data and solver.gt_data.get("cell_size") else None
            
            # Generar video MP4 fluido
            create_solving_animation(
                pieces_raw=solver.pieces_raw,
                placement_history=solver.placement_history,
                rows=solver.rows,
                cols=solver.cols,
                output_path=mp4_path,
                cell_size=c_size,
                fps=ds.get("fps", 10),
                continuous_tilts=solver.detected_tilts
            )
            # Generar GIF para reportes
            create_solving_animation(
                pieces_raw=solver.pieces_raw,
                placement_history=solver.placement_history,
                rows=solver.rows,
                cols=solver.cols,
                output_path=gif_path,
                cell_size=c_size,
                fps=ds.get("fps", 10),
                continuous_tilts=solver.detected_tilts
            )
            anim_time = time.time() - t_anim_start
            
        rec = {
            "dataset": ds_name,
            "grid": f"{ds['rows']}x{ds['cols']}",
            "pieces": ds["rows"] * ds["cols"],
            "t_load_s": round(t_load, 2),
            "t_solve_s": round(t_solve_total, 2),
            "direct_acc": direct_acc,
            "neighbor_acc": neighbor_acc,
            "lcc_pct": lcc_pct,
            "angle_err": mean_angle_err,
            "gif_path": gif_path if generate_animations else None
        }
        records.append(rec)
        
    print("\n" + "=" * 80)
    print("                    TABLA DE RESULTADOS FINALES                    ")
    print("=" * 80)
    print(f"{'Dataset':<24} | {'Grilla':<6} | {'Piezas':<6} | {'Tiempo':<8} | {'Direct Acc':<10} | {'Neighbor Acc':<12} | {'Rot Error':<10}")
    print("-" * 88)
    for r in records:
        print(f"{r['dataset']:<24} | {r['grid']:<6} | {r['pieces']:<6} | {r['t_solve_s']:<6.2f} s | {r['direct_acc']:<9.1f}% | {r['neighbor_acc']:<11.1f}% | {r['angle_err']:<6.2f} deg")
    print("=" * 88)
    return records


if __name__ == "__main__":
    run_benchmarks(generate_animations=True)
