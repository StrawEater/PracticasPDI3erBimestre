"""
Métricas formales de evaluación para rompecabezas basadas en la literatura científica
(e.g., Gallagher 2012, Pomeranz et al. 2011).

Incluye:
1. Direct Placement Accuracy (Exactitud de Colocación Directa)
2. Neighbor Pair Accuracy (Exactitud de Pares de Vecindad Horizontal y Vertical)
3. Largest Connected Component (Tamaño de la mayor componente conexa correctamente reconstruida)
"""

from typing import Dict, List, Any
import numpy as np


def compute_direct_accuracy(
    pred_grid: List[List[int]],
    gt_grid: List[List[int]],
    pred_rotations: Dict[int, int] = None,
    gt_rotations: Dict[int, int] = None
) -> Dict[str, float]:
    """
    Calcula el porcentaje de piezas colocadas en la celda (r, c) exacta.
    Si hay rotaciones, evalúa también si la rotación aplicada coincide con la requerida.
    """
    rows = len(gt_grid)
    cols = len(gt_grid[0])
    total_pieces = rows * cols
    
    correct_pos = 0
    correct_full = 0
    
    for r in range(rows):
        for c in range(cols):
            p_pred = pred_grid[r][c]
            p_gt = gt_grid[r][c]
            
            if p_pred == p_gt:
                correct_pos += 1
                
                # Chequear rotación si aplica
                if pred_rotations is not None and gt_rotations is not None:
                    rot_pred = pred_rotations.get(p_pred, 0) % 360
                    rot_gt = gt_rotations.get(p_gt, 0) % 360
                    if rot_pred == rot_gt:
                        correct_full += 1
                else:
                    correct_full += 1
                    
    pos_acc = (correct_pos / total_pieces) * 100.0
    full_acc = (correct_full / total_pieces) * 100.0
    
    return {
        "direct_position_accuracy": round(pos_acc, 2),
        "direct_full_accuracy": round(full_acc, 2),
        "correct_pieces": correct_full,
        "total_pieces": total_pieces
    }


def compute_neighbor_accuracy(
    pred_grid: List[List[int]],
    gt_grid: List[List[int]],
    pred_rotations: Dict[int, int] = None,
    gt_rotations: Dict[int, int] = None
) -> Dict[str, float]:
    """
    Calcula la exactitud de vecindad (Neighbor Pair Accuracy).
    Evalúa si dos piezas que están juntas en la predicción estaban realmente juntas
    en el ground truth en la misma relación relativa (Horizontal o Vertical).
    
    Esta métrica es invariante a traslaciones globales y refleja fielmente
    la calidad del ensamble de los bordes.
    """
    rows = len(gt_grid)
    cols = len(gt_grid[0])
    
    # 1. Construir conjunto de pares adyacentes en Ground Truth
    # Claves: ('H', pieza_izq, pieza_der) y ('V', pieza_arriba, pieza_abajo)
    gt_horizontal_pairs = set()
    gt_vertical_pairs = set()
    
    for r in range(rows):
        for c in range(cols):
            p_curr = gt_grid[r][c]
            rot_curr = gt_rotations.get(p_curr, 0) % 360 if gt_rotations else 0
            
            # Vecino a la derecha
            if c + 1 < cols:
                p_right = gt_grid[r][c + 1]
                rot_right = gt_rotations.get(p_right, 0) % 360 if gt_rotations else 0
                gt_horizontal_pairs.add((p_curr, rot_curr, p_right, rot_right))
                
            # Vecino abajo
            if r + 1 < rows:
                p_down = gt_grid[r + 1][c]
                rot_down = gt_rotations.get(p_down, 0) % 360 if gt_rotations else 0
                gt_vertical_pairs.add((p_curr, rot_curr, p_down, rot_down))
                
    total_h_pairs = len(gt_horizontal_pairs)
    total_v_pairs = len(gt_vertical_pairs)
    total_pairs = total_h_pairs + total_v_pairs
    
    # 2. Contar pares acertados en la predicción
    correct_h = 0
    correct_v = 0
    
    for r in range(rows):
        for c in range(cols):
            p_curr = pred_grid[r][c]
            rot_curr = pred_rotations.get(p_curr, 0) % 360 if pred_rotations else 0
            
            if c + 1 < cols:
                p_right = pred_grid[r][c + 1]
                rot_right = pred_rotations.get(p_right, 0) % 360 if pred_rotations else 0
                pair = (p_curr, rot_curr, p_right, rot_right)
                # Si no se evalúa rotación, comparar solo IDs
                if pred_rotations is None or gt_rotations is None:
                    if any(p[0] == p_curr and p[2] == p_right for p in gt_horizontal_pairs):
                        correct_h += 1
                else:
                    if pair in gt_horizontal_pairs:
                        correct_h += 1
                        
            if r + 1 < rows:
                p_down = pred_grid[r + 1][c]
                rot_down = pred_rotations.get(p_down, 0) % 360 if pred_rotations else 0
                pair = (p_curr, rot_curr, p_down, rot_down)
                if pred_rotations is None or gt_rotations is None:
                    if any(p[0] == p_curr and p[2] == p_down for p in gt_vertical_pairs):
                        correct_v += 1
                else:
                    if pair in gt_vertical_pairs:
                        correct_v += 1
                        
    correct_total = correct_h + correct_v
    neighbor_acc = (correct_total / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    h_acc = (correct_h / total_h_pairs) * 100.0 if total_h_pairs > 0 else 0.0
    v_acc = (correct_v / total_v_pairs) * 100.0 if total_v_pairs > 0 else 0.0
    
    return {
        "neighbor_accuracy": round(neighbor_acc, 2),
        "horizontal_accuracy": round(h_acc, 2),
        "vertical_accuracy": round(v_acc, 2),
        "correct_pairs": correct_total,
        "total_pairs": total_pairs
    }


def compute_largest_connected_component(
    pred_grid: List[List[int]],
    gt_grid: List[List[int]]
) -> Dict[str, Any]:
    """
    Calcula el tamaño del mayor bloque de piezas conexas contiguas correctamente ordenadas.
    Utiliza BFS para encontrar la componente conexa más grande en el grafo de aciertos.
    """
    rows = len(gt_grid)
    cols = len(gt_grid[0])
    
    # Mapear cada pieza a su posición en GT
    gt_pos = {}
    for r in range(rows):
        for c in range(cols):
            gt_pos[gt_grid[r][c]] = (r, c)
            
    # Construir grafo de adyacencias correctas entre celdas de la grilla predicha
    visited = set()
    max_component_size = 0
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited:
                continue
                
            # Iniciar BFS para esta componente
            queue = [(r, c)]
            comp = set([(r, c)])
            visited.add((r, c))
            
            while queue:
                curr_r, curr_c = queue.pop(0)
                p_curr = pred_grid[curr_r][curr_c]
                gt_r_curr, gt_c_curr = gt_pos[p_curr]
                
                # Vecinos en la grilla predicha
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr_r + dr, curr_c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        p_neighbor = pred_grid[nr][nc]
                        gt_r_neigh, gt_c_neigh = gt_pos[p_neighbor]
                        
                        # La relación relativa en la predicción debe coincidir con la de GT
                        if (gt_r_neigh - gt_r_curr == dr) and (gt_c_neigh - gt_c_curr == dc):
                            visited.add((nr, nc))
                            comp.add((nr, nc))
                            queue.append((nr, nc))
                            
            if len(comp) > max_component_size:
                max_component_size = len(comp)
                
    total_pieces = rows * cols
    lcc_percentage = (max_component_size / total_pieces) * 100.0
    
    return {
        "largest_connected_component": max_component_size,
        "lcc_percentage": round(lcc_percentage, 2),
        "total_pieces": total_pieces
    }


def evaluate_puzzle(
    pred_grid: List[List[int]],
    gt_grid: List[List[int]],
    pred_rotations: Dict[int, int] = None,
    gt_rotations: Dict[int, int] = None
) -> Dict[str, Any]:
    """
    Función integral que calcula todas las métricas oficiales.
    """
    direct = compute_direct_accuracy(pred_grid, gt_grid, pred_rotations, gt_rotations)
    neighbor = compute_neighbor_accuracy(pred_grid, gt_grid, pred_rotations, gt_rotations)
    lcc = compute_largest_connected_component(pred_grid, gt_grid)
    
    return {
        "direct": direct,
        "neighbor": neighbor,
        "lcc": lcc
    }
