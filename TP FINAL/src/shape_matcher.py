"""
Módulo de Procesamiento de Imágenes para la Extracción, Clasificación y Correlación de Bordes:
1. Binarización limpia (Fondo Negro '0' vs. Pieza '255') con relleno morfológico de huecos.
2. Extracción del contorno externo y detección precisa de las 4 esquinas de la grilla.
3. Partición del contorno en 4 señales 1D (Norte, Este, Sur, Oeste) de desviación perpendicular.
4. Clasificación topológica de la pieza: ESQUINA (2 lados planos), BORDE (1 lado plano), INTERIOR (0 planos).
5. Cálculo de la matriz de correlación / producto interno normalizado (estilo Fourier) entre bordes de piezas:
   da 1.0 para match perfecto, ~0.0 para no-match / desfasado en altura, y 0.0 para incompatibles.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def binarize_piece(
    img_rgb: np.ndarray,
    fixed_threshold: int = 0
) -> np.ndarray:
    """
    Binariza la imagen de la pieza aislando la silueta completa (255)
    del fondo negro (0), con relleno de agujeros para evitar que sombras oscuras
    dentro de la textura de la imagen creen ranuras falsas.
    """
    if img_rgb.ndim == 3:
        max_channel = np.max(img_rgb, axis=2)
    else:
        max_channel = img_rgb
        
    _, binary = cv2.threshold(max_channel, fixed_threshold, 255, cv2.THRESH_BINARY)
    
    # Relleno de agujeros interiores mediante floodFill desde el borde exterior
    h, w = binary.shape
    mask_flood = np.zeros((h + 2, w + 2), np.uint8)
    bin_inv = cv2.floodFill(binary.copy(), mask_flood, (0, 0), 255)[1]
    bin_filled = cv2.bitwise_not(bin_inv)
    binary = cv2.bitwise_or(binary, bin_filled)
    
    return binary


def extract_external_contour(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extrae el contorno exterior de mayor área de la máscara binaria.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No se detectó ningún contorno en la máscara binaria provista.")
    main_contour = max(contours, key=cv2.contourArea)
    return main_contour.squeeze(axis=1)


def detect_jigsaw_corners(
    contour_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detecta las 4 esquinas base (TL, TR, BR, BL) de la pieza rectangular nominal.
    """
    mid_x, mid_y = np.mean(contour_pts, axis=0)
    
    q_tl = [p for p in contour_pts if p[0] <= mid_x and p[1] <= mid_y]
    q_tr = [p for p in contour_pts if p[0] >= mid_x and p[1] <= mid_y]
    q_br = [p for p in contour_pts if p[0] >= mid_x and p[1] >= mid_y]
    q_bl = [p for p in contour_pts if p[0] <= mid_x and p[1] >= mid_y]
    
    min_x, max_x = np.min(contour_pts[:, 0]), np.max(contour_pts[:, 0])
    min_y, max_y = np.min(contour_pts[:, 1]), np.max(contour_pts[:, 1])
    
    c_tl = min(q_tl, key=lambda p: (p[0] - min_x)**2 + (p[1] - min_y)**2)
    c_tr = min(q_tr, key=lambda p: (p[0] - max_x)**2 + (p[1] - min_y)**2)
    c_br = min(q_br, key=lambda p: (p[0] - max_x)**2 + (p[1] - max_y)**2)
    c_bl = min(q_bl, key=lambda p: (p[0] - min_x)**2 + (p[1] - max_y)**2)
    
    return np.array(c_tl), np.array(c_tr), np.array(c_br), np.array(c_bl)


def detect_corners_and_split_sides(
    contour_pts: np.ndarray,
    binary_mask: np.ndarray,
    num_samples: int = 100
) -> Dict[str, Any]:
    """
    Segmenta el contorno en los 4 lados orientados (N, E, S, W),
    extrae la señal 1D de desviación perpendicular regularizada y clasifica cada lado.
    """
    c_tl, c_tr, c_br, c_bl = detect_jigsaw_corners(contour_pts)
    
    def get_idx(pt):
        d = np.sum((contour_pts - pt)**2, axis=1)
        return int(np.argmin(d))
        
    i_tl = get_idx(c_tl)
    i_tr = get_idx(c_tr)
    i_br = get_idx(c_br)
    i_bl = get_idx(c_bl)
    
    n = len(contour_pts)
    shifted = np.roll(contour_pts, -i_tl, axis=0)
    i_tr_s = (i_tr - i_tl) % n
    i_br_s = (i_br - i_tl) % n
    i_bl_s = (i_bl - i_tl) % n
    
    # Determinar sentido de recorrido (horario o antihorario)
    if i_bl_s < i_tr_s:
        # Antihorario: TL -> BL -> BR -> TR -> TL
        curve_w = shifted[0 : i_bl_s + 1]             # TL -> BL
        curve_s = shifted[i_bl_s : i_br_s + 1]        # BL -> BR
        curve_e = shifted[i_br_s : i_tr_s + 1][::-1] # TR -> BR
        curve_n = np.vstack([shifted[i_tr_s:], shifted[0:1]])[::-1] # TL -> TR
    else:
        # Horario: TL -> TR -> BR -> BL -> TL
        curve_n = shifted[0 : i_tr_s + 1]              # TL -> TR
        curve_e = shifted[i_tr_s : i_br_s + 1]         # TR -> BR
        curve_s = shifted[i_br_s : i_bl_s + 1][::-1]  # BL -> BR
        curve_w = np.vstack([shifted[i_bl_s:], shifted[0:1]])[::-1] # BL -> TL
        
    def extract_side_signal(curve: np.ndarray, side_name: str) -> Dict[str, Any]:
        p0 = curve[0].astype(np.float32)
        p1 = curve[-1].astype(np.float32)
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length == 0:
            return {
                "type": "PLANO",
                "profile": np.zeros(num_samples, dtype=np.float32),
                "norm": 0.0,
                "length": 0.0
            }
            
        u = vec / length
        if side_name == "N": normal_unit = np.array([0.0, -1.0])
        elif side_name == "S": normal_unit = np.array([0.0, 1.0])
        elif side_name == "W": normal_unit = np.array([-1.0, 0.0])
        elif side_name == "E": normal_unit = np.array([1.0, 0.0])
        else: normal_unit = np.array([u[1], -u[0]], dtype=np.float32)
        
        rel = curve.astype(np.float32) - p0
        # Proyección sobre la tangente (0 a length)
        proj_t = np.dot(rel, u)
        # Desviación perpendicular (señal de bulbo)
        dev = np.dot(rel, normal_unit)
        
        # Muestreo regularizado en t in [0, 1]
        t_orig = np.linspace(0, 1, len(curve))
        t_target = np.linspace(0, 1, num_samples)
        profile = np.interp(t_target, t_orig, dev).astype(np.float32)
        
        max_dev = float(np.max(np.abs(profile)))
        mean_dev = float(np.mean(profile))
        
        # Umbral para clasificar como plano
        if max_dev < length * 0.05:
            stype = "PLANO"
            profile = np.zeros(num_samples, dtype=np.float32)
        elif mean_dev > 0:
            stype = "MACHO"
        else:
            stype = "HEMBRA"
            
        norm = float(np.linalg.norm(profile))
        return {
            "type": stype,
            "profile": profile,
            "norm": norm,
            "length": length,
            "max_dev": max_dev,
            "mean_dev": mean_dev
        }

    info_n = extract_side_signal(curve_n, "N")
    info_e = extract_side_signal(curve_e, "E")
    info_s = extract_side_signal(curve_s, "S")
    info_w = extract_side_signal(curve_w, "W")
    
    # Clasificación topológica global de la pieza
    types = [info_n["type"], info_e["type"], info_s["type"], info_w["type"]]
    num_flat = sum(1 for t in types if t == "PLANO")
    if num_flat == 2:
        topology = "CORNER"
    elif num_flat == 1:
        topology = "BORDER"
    else:
        topology = "INTERIOR"
        
    return {
        "N": info_n,
        "E": info_e,
        "S": info_s,
        "W": info_w,
        "topology": topology,
        "num_flat": num_flat,
        "corners": {"TL": c_tl, "TR": c_tr, "BR": c_br, "BL": c_bl}
    }


def analyze_piece_shape(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Función principal de análisis morfológico de una pieza:
    1. Binariza limpiamente la pieza (fondo negro 0, pieza 255).
    2. Extrae contorno externo.
    3. Segmenta los 4 lados y calcula sus señales 1D y clasificación topológica.
    """
    binary = binarize_piece(img_rgb)
    contour = extract_external_contour(binary)
    sides_info = detect_corners_and_split_sides(contour, binary)
    
    return {
        "binary_mask": binary,
        "contour": contour,
        "sides": sides_info,
        "topology": sides_info["topology"]
    }


def get_rotated_sides(sides_dict: Dict[str, Any], rot_k: int) -> Dict[str, Any]:
    """
    Devuelve los lados de una pieza rotada por rot_k * 90° en sentido horario.
    rot_k in {0, 1, 2, 3}.
    """
    if rot_k % 4 == 0:
        return sides_dict
        
    k = rot_k % 4
    sides_order = ["N", "E", "S", "W"]
    rotated = {}
    for i, side in enumerate(sides_order):
        orig_side = sides_order[(i - k) % 4]
        rotated[side] = sides_dict[orig_side]
    return rotated


def compute_edge_correlation(
    side_a: Dict[str, Any],
    side_b: Dict[str, Any]
) -> float:
    """
    Calcula el producto interno normalizado (correlación tipo Fourier)
    entre la señal 1D del borde A y el borde B:
    - Retorna 1.0 si encastran perfectamente (macho con hembra coincidente en altura y forma).
    - Retorna ~0.0 si los bulbos están desfasados en altura o son ortogonales.
    - Retorna 0.0 si son incompatibles (macho-macho, hembra-hembra o bordes planos interiores).
    """
    type_a = side_a["type"]
    type_b = side_b["type"]
    
    # Dos bordes planos no encajan en el interior
    if type_a == "PLANO" or type_b == "PLANO":
        return 0.0
        
    # Deben ser complementarios estricto (uno MACHO y uno HEMBRA)
    is_complementary = (type_a == "MACHO" and type_b == "HEMBRA") or (type_a == "HEMBRA" and type_b == "MACHO")
    if not is_complementary:
        return 0.0
        
    prof_a = side_a["profile"]
    prof_b = side_b["profile"]
    
    # Las curvas ya están segmentadas en la misma dirección canónica a lo largo de la costura
    # (Izquierda -> Derecha para N y S; Arriba -> Abajo para E y W).
    # Su normal exterior es opuesta a la de A (-prof_b).
    prof_b_comp = -prof_b
    
    norm_a = side_a.get("norm", float(np.linalg.norm(prof_a)))
    norm_b = side_b.get("norm", float(np.linalg.norm(prof_b)))
    
    if norm_a < 1e-4 or norm_b < 1e-4:
        return 0.0
        
    # Producto interno continuo discretizado <prof_a, prof_b_comp>
    dot = float(np.dot(prof_a, prof_b_comp))
    rho = dot / (norm_a * norm_b)
    
    if rho <= 0.0:
        return 0.0
        
    # Ratio de amplitudes para premiar profundidades idénticas
    amp_ratio = min(norm_a, norm_b) / max(norm_a, norm_b)
    
    correlation = float(rho * amp_ratio)
    return max(0.0, min(1.0, correlation))


def compute_jigsaw_shape_compatibility(
    side_a: Dict[str, Any],
    side_b: Dict[str, Any]
) -> float:
    """
    Función de compatibilidad/costo para el affinity matcher:
    Convierte la correlación [0, 1] en un costo de disimilitud:
    - 0.0 para match perfecto.
    - 1e6 para incompatibles.
    """
    corr = compute_edge_correlation(side_a, side_b)
    if corr <= 1e-4:
        return 1e6
    # Costo inversamente proporcional a la correlación
    return float((1.0 - corr) * 100.0)
