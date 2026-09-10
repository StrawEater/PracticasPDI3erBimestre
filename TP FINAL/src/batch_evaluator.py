"""
Herramienta Docente: Evaluador por Lotes (Batch Evaluator)
Recorre una carpeta con las entregas de los estudiantes (o múltiples puzzles de evaluación),
ejecuta la evaluación automática de cada una y genera un resumen en consola y archivo CSV con las notas.
"""

import os
import sys
import csv
import argparse
from typing import List, Dict, Any

try:
    from .evaluator import PuzzleEvaluator
    from .utils import load_json
except ImportError:
    from evaluator import PuzzleEvaluator
    from utils import load_json


def evaluate_student_submissions(
    eval_puzzles_dir: str,
    submissions_dir: str,
    output_csv_path: str
) -> None:
    """
    Args:
        eval_puzzles_dir: Carpeta que contiene los rompecabezas de prueba (cada uno con su ground_truth.json).
        submissions_dir: Carpeta donde cada subcarpeta es un grupo de alumnos (e.g. grupo_01, grupo_02)
                         conteniendo las predicciones para cada puzzle (e.g. pred_puzzle1.json).
        output_csv_path: Archivo CSV donde se consolidarán los resultados.
    """
    puzzle_names = [d for d in os.listdir(eval_puzzles_dir) if os.path.isdir(os.path.join(eval_puzzles_dir, d))]
    puzzle_names.sort()
    
    if not puzzle_names:
        print(f"[ERROR] No se encontraron carpetas de puzzles en: {eval_puzzles_dir}")
        return

    student_groups = [d for d in os.listdir(submissions_dir) if os.path.isdir(os.path.join(submissions_dir, d))]
    student_groups.sort()
    
    if not student_groups:
        print(f"[ERROR] No se encontraron carpetas de grupos en: {submissions_dir}")
        return

    print("=" * 80)
    print("           EVALUACIÓN DOCENTE POR LOTES (BATCH EVALUATION)           ")
    print(f" Puzzles de prueba: {len(puzzle_names)} ({', '.join(puzzle_names)})")
    print(f" Grupos a evaluar:  {len(student_groups)}")
    print("=" * 80)

    results_table = []
    
    for group in student_groups:
        group_dir = os.path.join(submissions_dir, group)
        group_row = {"Grupo": group}
        total_neighbor_acc = 0.0
        puzzles_evaluated = 0
        
        print(f"\n>> Evaluando entrega: {group}")
        
        for p_name in puzzle_names:
            gt_file = os.path.join(eval_puzzles_dir, p_name, "ground_truth.json")
            # Buscar si el estudiante entregó prediction.json o prediction_{p_name}.json
            possible_pred_names = [
                f"prediction_{p_name}.json",
                f"{p_name}.json",
                "prediction.json"
            ]
            pred_file = None
            for candidate in possible_pred_names:
                candidate_path = os.path.join(group_dir, candidate)
                if os.path.isfile(candidate_path):
                    pred_file = candidate_path
                    break
                    
            if pred_file and os.path.isfile(gt_file):
                try:
                    evaluator = PuzzleEvaluator(gt_file)
                    # Silenciamos reporte visual en batch a menos que sea necesario
                    metrics = evaluator.evaluate(pred_file, save_visual_path=None)
                    neigh_acc = metrics["neighbor"]["neighbor_accuracy"]
                    direct_acc = metrics["direct"]["direct_position_accuracy"]
                    lcc = metrics["lcc"]["lcc_percentage"]
                    
                    group_row[f"{p_name}_NeighborAcc%"] = neigh_acc
                    group_row[f"{p_name}_DirectAcc%"] = direct_acc
                    group_row[f"{p_name}_LCC%"] = lcc
                    
                    total_neighbor_acc += neigh_acc
                    puzzles_evaluated += 1
                    print(f"   - {p_name}: Neighbor={neigh_acc}% | Direct={direct_acc}% | LCC={lcc}%")
                except Exception as e:
                    print(f"   - {p_name}: [ERROR al evaluar] {e}")
                    group_row[f"{p_name}_NeighborAcc%"] = 0.0
            else:
                print(f"   - {p_name}: [NO ENTREGADO]")
                group_row[f"{p_name}_NeighborAcc%"] = 0.0

        avg_acc = (total_neighbor_acc / len(puzzle_names)) if puzzle_names else 0.0
        group_row["Promedio_NeighborAcc%"] = round(avg_acc, 2)
        results_table.append(group_row)

    # Guardar en CSV
    if results_table:
        fieldnames = list(results_table[0].keys())
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results_table:
                writer.writerow(r)
                
        print("\n" + "=" * 80)
        print(f"[ÉXITO] Planilla consolidada generada en: '{output_csv_path}'")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluador en Lote de Entregas Estudiantiles - Cátedra PDI")
    parser.add_argument("--test-set", type=str, required=True, help="Carpeta con los rompecabezas oficiales")
    parser.add_argument("--submissions", type=str, required=True, help="Carpeta con las entregas de los estudiantes")
    parser.add_argument("--output-csv", type=str, default="calificaciones_puzzles.csv", help="Ruta para el archivo CSV de salida")
    
    args = parser.parse_args()
    evaluate_student_submissions(args.test_set, args.submissions, args.output_csv)


if __name__ == "__main__":
    main()
