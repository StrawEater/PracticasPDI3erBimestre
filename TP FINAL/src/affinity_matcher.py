"""
Módulo de procesamiento de imágenes para la extracción de características de borde
y cálculo de matrices de afinidad / compatibilidad entre piezas.

Combina:
1. Limpieza y preprocesamiento de filtros espaciales (Bilateral / Gaussiano).
2. Representación en espacio perceptual CIE-Lab.
3. Continuidad espacial y extrapolación de perfiles (análisis de tiras interiores).
4. Coherencia de dirección y magnitud de gradientes espaciales (Sobel).
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any

try:
    from .utils import rotate_image_clockwise
    from .shape_matcher import analyze_piece_shape, compute_jigsaw_shape_compatibility
    from .stripe_analyzer import detect_stripe_orientation
except ImportError:
    from utils import rotate_image_clockwise
    from shape_matcher import analyze_piece_shape, compute_jigsaw_shape_compatibility
    from stripe_analyzer import detect_stripe_orientation


def preprocess_piece(
    piece_rgb: np.ndarray,
    use_bilateral: bool = True,
    gaussian_ksize: int = 3
) -> Dict[str, np.ndarray]:
    """
    Aplica limpieza y preprocesamiento a una pieza:
    - Filtro bilateral o gaussiano para suavizado de ruido preservando aristas.
    - Conversión a espacio de color perceptual CIE-Lab.
    - Cálculo de gradientes espaciales (Sobel) en magnitud y orientación angular.
    """
    # 1. Limpieza de ruido con filtro bilateral (preserva bordes mejor que Gaussiano)
    if use_bilateral:
        denoised_rgb = cv2.bilateralFilter(piece_rgb, d=5, sigmaColor=35, sigmaSpace=35)
    else:
        denoised_rgb = cv2.GaussianBlur(piece_rgb, (gaussian_ksize, gaussian_ksize), 0)
        
    # 2. Conversión a CIE-Lab (donde distancias euclidianas son perceptualmente uniformes)
    lab = cv2.cvtColor(denoised_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # 3. Canal de luminancia (L) para gradientes estructurales
    gray_l = lab[:, :, 0]
    
    # Gradientes espaciales con operador de Sobel
    grad_x = cv2.Sobel(gray_l, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_l, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    # Orientación del gradiente en radianes [-pi, pi]
    angle = np.arctan2(grad_y, grad_x)
    
    return {
        "rgb": piece_rgb,
        "denoised_rgb": denoised_rgb,
        "lab": lab,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "magnitude": magnitude,
        "angle": angle
    }


def extract_edge_profile(
    piece_data: Dict[str, np.ndarray],
    side: str
) -> Dict[str, np.ndarray]:
    """
    Extrae la tira de borde real y la franja interior para evaluar continuidad cromática y de gradientes.
    Soporta tanto piezas rectangulares estándar como piezas de rompecabezas con fondo negro.
    side: 'N' (Arriba), 'S' (Abajo), 'W' (Izquierda), 'E' (Derecha).
    """
    lab = piece_data["lab"]
    mag = piece_data["magnitude"]
    ang = piece_data["angle"]
    rgb = piece_data["rgb"]
    
    # Comprobar si tiene fondo negro (jigsaw)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    mask = (gray > 5)
    
    if not np.any(mask):
        return {
            "lab": np.zeros((50, 3), dtype=np.float32),
            "inner_lab": np.zeros((50, 3), dtype=np.float32),
            "magnitude": np.zeros(50, dtype=np.float32),
            "angle": np.zeros(50, dtype=np.float32)
        }
        
    has_black_border = bool(np.mean(mask) < 0.95)
    
    if not has_black_border:
        if side == "N":
            return {"lab": lab[0, :, :], "inner_lab": lab[1, :, :], "magnitude": mag[0, :], "angle": ang[0, :]}
        elif side == "S":
            return {"lab": lab[-1, :, :], "inner_lab": lab[-2, :, :], "magnitude": mag[-1, :], "angle": ang[-1, :]}
        elif side == "W":
            return {"lab": lab[:, 0, :], "inner_lab": lab[:, 1, :], "magnitude": mag[:, 0], "angle": ang[:, 0]}
        elif side == "E":
            return {"lab": lab[:, -1, :], "inner_lab": lab[:, -2, :], "magnitude": mag[:, -1], "angle": ang[:, -1]}
            
    # Para piezas de rompecabezas con silueta: muestrear a lo largo del límite real no nulo
    valid_cols = np.where(np.any(mask, axis=0))[0]
    valid_rows = np.where(np.any(mask, axis=1))[0]
    
    if side == "N":
        y = np.argmax(mask[:, valid_cols], axis=0)
        x = valid_cols
        y_in = np.clip(y + 2, 0, lab.shape[0] - 1)
        x_in = x
    elif side == "S":
        y = mask.shape[0] - 1 - np.argmax(mask[::-1, valid_cols], axis=0)
        x = valid_cols
        y_in = np.clip(y - 2, 0, lab.shape[0] - 1)
        x_in = x
    elif side == "W":
        y = valid_rows
        x = np.argmax(mask[valid_rows, :], axis=1)
        y_in = y
        x_in = np.clip(x + 2, 0, lab.shape[1] - 1)
    elif side == "E":
        y = valid_rows
        x = mask.shape[1] - 1 - np.argmax(mask[valid_rows, ::-1], axis=1)
        y_in = y
        x_in = np.clip(x - 2, 0, lab.shape[1] - 1)
    else:
        raise ValueError(f"Lado no reconocido: {side}")
        
    return {
        "lab": lab[y, x],
        "inner_lab": lab[y_in, x_in],
        "magnitude": mag[y, x],
        "angle": ang[y, x]
    }


def resample_1d_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Re-muestrea un array 1D o 2D (canales) a una longitud fija mediante interpolación lineal."""
    if len(arr) == target_len:
        return arr
    t_orig = np.linspace(0, 1, len(arr))
    t_new = np.linspace(0, 1, target_len)
    if arr.ndim == 1:
        return np.interp(t_new, t_orig, arr).astype(arr.dtype)
    else:
        channels = [np.interp(t_new, t_orig, arr[:, c]) for c in range(arr.shape[1])]
        return np.stack(channels, axis=-1).astype(arr.dtype)


def compute_border_dissimilarity(
    profile_a: Dict[str, np.ndarray],
    profile_b: Dict[str, np.ndarray],
    weights: Tuple[float, float, float] = (0.6, 0.25, 0.15)
) -> float:
    """
    Calcula la disimilitud combinando:
    1. Match de color en CIE-Lab (diferencia euclidiana en la frontera).
    2. Continuidad espacial de 2do orden (extrapolación del perfil interior hacia el vecino).
    3. Coherencia estructural de gradientes (magnitud y alineación angular).
    
    Retorna un valor escalar >= 0. Menor costo = mayor afinidad.
    """
    w_color, w_cont, w_grad = weights
    
    lab_a = profile_a["lab"]
    lab_b = profile_b["lab"]
    
    if len(lab_a) != len(lab_b):
        common_len = min(len(lab_a), len(lab_b))
        lab_a = resample_1d_array(lab_a, common_len)
        lab_b = resample_1d_array(lab_b, common_len)
        inner_a = resample_1d_array(profile_a["inner_lab"], common_len)
        inner_b = resample_1d_array(profile_b["inner_lab"], common_len)
        mag_a = resample_1d_array(profile_a["magnitude"], common_len)
        mag_b = resample_1d_array(profile_b["magnitude"], common_len)
        ang_a = resample_1d_array(profile_a["angle"], common_len)
        ang_b = resample_1d_array(profile_b["angle"], common_len)
    else:
        inner_a = profile_a["inner_lab"]
        inner_b = profile_b["inner_lab"]
        mag_a = profile_a["magnitude"]
        mag_b = profile_b["magnitude"]
        ang_a = profile_a["angle"]
        ang_b = profile_b["angle"]
    
    # --- 1. Match cromático en la frontera inmediata ---
    diff_color = lab_a - lab_b
    cost_color = float(np.mean(np.sqrt(np.sum(diff_color ** 2, axis=-1))))
    
    # --- 2. Continuidad de imagen (extrapolación lineal) ---
    extrap_a = 2.0 * lab_a - inner_a
    diff_extrap_a = extrap_a - lab_b
    
    extrap_b = 2.0 * lab_b - inner_b
    diff_extrap_b = extrap_b - lab_a
    
    cost_cont = float(0.5 * (
        np.mean(np.sqrt(np.sum(diff_extrap_a ** 2, axis=-1))) +
        np.mean(np.sqrt(np.sum(diff_extrap_b ** 2, axis=-1)))
    ))
    
    # --- 3. Coherencia de magnitud y orientación de gradientes ---
    diff_mag = np.abs(mag_a - mag_b)
    cost_mag = float(np.mean(diff_mag))
    
    ang_diff = np.abs(ang_a - ang_b)
    ang_diff = np.minimum(ang_diff, 2 * np.pi - ang_diff)
    cost_ang = float(np.mean(ang_diff))
    
    cost_grad = cost_mag * 0.7 + (cost_ang * 25.0) * 0.3
    
    # Costo final ponderado
    total_cost = (w_color * cost_color) + (w_cont * cost_cont) + (w_grad * cost_grad)
    return float(total_cost)


def compute_all_pairwise_relations(
    pieces: Dict[int, np.ndarray],
    allow_rotations: bool = False,
    has_stripes: bool = False
) -> Dict[str, Any]:
    """
    Calcula la matriz/tensor completo de afinidad entre todos los pares de piezas
    para las direcciones HORIZONTAL (Este-Oeste) y VERTICAL (Sur-Norte).
    """
    piece_ids = sorted(list(pieces.keys()))
    rotations = [0, 90, 180, 270] if allow_rotations else [0]
    
    # Detectar si las piezas corresponden a formato jigsaw (fondo negro > 2% de píxeles)
    sample_img = pieces[piece_ids[0]]
    is_jigsaw = bool(np.mean(sample_img <= 5) > 0.02)
    
    # 1. Preprocesar y extraer perfiles de cada pieza en cada orientación
    profiles = {} # (p_id, rot) -> {'N': ..., 'S': ..., 'W': ..., 'E': ...}
    shapes = {}   # (p_id, rot) -> resultado de analyze_piece_shape
    stripe_angles = {} # (p_id, rot) -> ángulo estimado de las rayas
    
    for p_id in piece_ids:
        raw_img = pieces[p_id]
        for rot in rotations:
            rotated_img = rotate_image_clockwise(raw_img, rot)
            preprocessed = preprocess_piece(rotated_img, use_bilateral=True)
            
            profiles[(p_id, rot)] = {
                "N": extract_edge_profile(preprocessed, "N"),
                "S": extract_edge_profile(preprocessed, "S"),
                "W": extract_edge_profile(preprocessed, "W"),
                "E": extract_edge_profile(preprocessed, "E")
            }
            
            if has_stripes:
                try:
                    stripe_angles[(p_id, rot)] = detect_stripe_orientation(rotated_img)
                except Exception:
                    stripe_angles[(p_id, rot)] = 0.0
                
            if is_jigsaw:
                try:
                    shapes[(p_id, rot)] = analyze_piece_shape(rotated_img)
                except Exception:
                    pass
            
    # 2. Computar afinidades para todos los pares ordenados (p_a != p_b)
    horiz_relations = {}
    vert_relations = {}
    
    for p_a in piece_ids:
        for rot_a in rotations:
            prof_a = profiles[(p_a, rot_a)]
            shape_a = shapes.get((p_a, rot_a))
            ang_a = stripe_angles.get((p_a, rot_a), 0.0)
            
            for p_b in piece_ids:
                if p_a == p_b:
                    continue
                for rot_b in rotations:
                    prof_b = profiles[(p_b, rot_b)]
                    shape_b = shapes.get((p_b, rot_b))
                    ang_b = stripe_angles.get((p_b, rot_b), 0.0)
                    
                    # Coherencia de orientación de rayas periódicas (solo si has_stripes es True)
                    if has_stripes:
                        stripe_diff = abs(ang_a - ang_b)
                        stripe_diff = min(stripe_diff, 180.0 - stripe_diff)
                        stripe_penalty = (stripe_diff * 4.0) if stripe_diff > 18.0 else 0.0
                    else:
                        stripe_penalty = 0.0
                    
                    # Horizontal: p_a a la izquierda de p_b (Borde Este de A vs Borde Oeste de B)
                    if shape_a and shape_b:
                        shape_h = compute_jigsaw_shape_compatibility(shape_a["sides"]["E"], shape_b["sides"]["W"])
                        if shape_h >= 1e5:
                            h_cost = 1e6 # Incompatible: ambos macho, ambos hembra o borde plano
                        else:
                            color_h = compute_border_dissimilarity(prof_a["E"], prof_b["W"])
                            h_cost = (shape_h * 10.0) + (0.1 * min(color_h, 300.0))
                    else:
                        h_cost = compute_border_dissimilarity(prof_a["E"], prof_b["W"])
                    horiz_relations[(p_a, rot_a, p_b, rot_b)] = h_cost + stripe_penalty
                    
                    # Vertical: p_a arriba de p_b (Borde Sur de A vs Borde Norte de B)
                    if shape_a and shape_b:
                        shape_v = compute_jigsaw_shape_compatibility(shape_a["sides"]["S"], shape_b["sides"]["N"])
                        if shape_v >= 1e5:
                            v_cost = 1e6
                        else:
                            color_v = compute_border_dissimilarity(prof_a["S"], prof_b["N"])
                            v_cost = (shape_v * 10.0) + (0.1 * min(color_v, 300.0))
                    else:
                        v_cost = compute_border_dissimilarity(prof_a["S"], prof_b["N"])
                    vert_relations[(p_a, rot_a, p_b, rot_b)] = v_cost + stripe_penalty
                    
    return {
        "horizontal": horiz_relations,
        "vertical": vert_relations,
        "piece_ids": piece_ids,
        "rotations": rotations,
        "shapes": shapes,
        "is_jigsaw": is_jigsaw
    }
