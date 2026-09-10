"""
Módulo de Procesamiento de Imágenes para la Extracción y Matcheo de Formas de Piezas:
1. Binarización de piezas (Fondo Negro '0' vs. Pieza Completa '255') mediante Otsu o umbralización.
2. Extracción de contornos externos y detección de esquinas del encastre.
3. Partición del contorno en 4 lados (Norte, Sur, Este, Oeste).
4. Clasificación morfológica de cada lado: 'PLANO', 'MACHO' (pestaña) o 'HEMBRA' (hueco).
5. Matcheo de forma de bordes complementarios.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def binarize_piece(
    img_rgb: np.ndarray,
    method: str = "otsu",
    fixed_threshold: int = 5
) -> np.ndarray:
    """
    Binariza la imagen de la pieza aislando la silueta completa (blanco = 255)
    del fondo negro (negro = 0).
    
    Args:
        img_rgb: Imagen RGB de la pieza (fondo negro 0,0,0).
        method: 'otsu' para binarización automática o 'threshold' con valor fijo.
        fixed_threshold: Umbral para método fijo.
        
    Returns:
        Máscara binaria uint8 (255 para la pieza, 0 para el fondo).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    if method.lower() == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, fixed_threshold, 255, cv2.THRESH_BINARY)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
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
    Detecta las 4 esquinas base (TL, TR, BR, BL) de la pieza.
    Las esquinas son los vértices de la grilla rectangular de la pieza.
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
    binary_mask: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """
    Segmenta el contorno en los 4 lados orientados (N, E, S, W)
    y clasifica morfológicamente cada lado como PLANO, MACHO o HEMBRA.
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
    
    # Determinar si el contorno recorre en sentido horario o antihorario
    # Si i_bl_s < i_tr_s, recorre antihorario (TL -> BL -> BR -> TR)
    if i_bl_s < i_tr_s:
        # Antihorario
        curve_w = shifted[0 : i_bl_s + 1]
        curve_s = shifted[i_bl_s : i_br_s + 1]
        curve_e = shifted[i_br_s : i_tr_s + 1]
        curve_n = np.vstack([shifted[i_tr_s:], shifted[0:1]])
        # Invertir para que todos los lados se recorran consistentemente de inicio a fin nominal
        curve_w = curve_w[::-1] # BL -> TL
        curve_s = curve_s       # BL -> BR
        curve_e = curve_e[::-1] # BR -> TR
        curve_n = curve_n[::-1] # TL -> TR
    else:
        # Horario: TL -> TR -> BR -> BL -> TL
        curve_n = shifted[0 : i_tr_s + 1]
        curve_e = shifted[i_tr_s : i_br_s + 1]
        curve_s = shifted[i_br_s : i_bl_s + 1]
        curve_w = np.vstack([shifted[i_bl_s:], shifted[0:1]])
        
    def classify_edge(curve: np.ndarray, side_name: str) -> Tuple[str, np.ndarray, str]:
        p0 = curve[0].astype(np.float32)
        p1 = curve[-1].astype(np.float32)
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length == 0:
            return "PLANO", np.zeros(50), "none"
            
        u = vec / length
        n_outward = np.array([u[1], -u[0]], dtype=np.float32)
        
        # Ajustar signo del normal hacia afuera según lado
        if side_name == "N": normal_unit = np.array([0.0, -1.0])
        elif side_name == "S": normal_unit = np.array([0.0, 1.0])
        elif side_name == "W": normal_unit = np.array([-1.0, 0.0])
        elif side_name == "E": normal_unit = np.array([1.0, 0.0])
        else: normal_unit = n_outward
        
        rel = curve.astype(np.float32) - p0
        trans = np.dot(rel, normal_unit)
        
        # Resamplear a 50 puntos
        t_orig = np.linspace(0, 1, len(curve))
        t_fixed = np.linspace(0, 1, 50)
        profile_50 = np.interp(t_fixed, t_orig, trans)
        
        max_dev = float(np.max(np.abs(profile_50)))
        mean_dev = float(np.mean(profile_50))
        
        if max_dev < length * 0.06:
            stype = "PLANO"
            curve_type = "none"
        elif mean_dev > 0:
            stype = "MACHO"
            curve_type = classify_profile_curve_type(profile_50, stype)
        else:
            stype = "HEMBRA"
            curve_type = classify_profile_curve_type(profile_50, stype)
            
        return stype, profile_50, curve_type

    type_n, prof_n, ctype_n = classify_edge(curve_n, "N")
    type_e, prof_e, ctype_e = classify_edge(curve_e, "E")
    type_s, prof_s, ctype_s = classify_edge(curve_s, "S")
    type_w, prof_w, ctype_w = classify_edge(curve_w, "W")
    
    return {
        "N": {"type": type_n, "profile": prof_n, "curve_type": ctype_n},
        "E": {"type": type_e, "profile": prof_e, "curve_type": ctype_e},
        "S": {"type": type_s, "profile": prof_s, "curve_type": ctype_s},
        "W": {"type": type_w, "profile": prof_w, "curve_type": ctype_w},
        "corners": {"TL": c_tl, "TR": c_tr, "BR": c_br, "BL": c_bl}
    }


def classify_profile_curve_type(profile_50: np.ndarray, stype: str) -> str:
    """
    Clasifica la función geométrica del borde ('standard', 'circular', 'random' o 'none')
    a partir del perfil 1D del encastre analizando plenitud y asimetría.
    """
    if stype == "PLANO":
        return "none"
        
    p = np.abs(profile_50)
    max_val = float(np.max(p))
    if max_val < 1e-4:
        return "none"
        
    p_norm = p / max_val
    active_idx = np.where(p_norm > 0.15)[0]
    if len(active_idx) < 4:
        return "standard"
        
    sub_p = p_norm[active_idx[0]:active_idx[-1] + 1]
    half = len(sub_p) // 2
    left = sub_p[:half]
    right = sub_p[-half:][::-1]
    
    asymmetry = float(np.mean(np.abs(left - right)))
    fullness = float(np.mean(sub_p))
    
    if asymmetry > 0.13:
        return "random"
    elif fullness > 0.68:
        return "circular"
    else:
        return "standard"


def analyze_piece_shape(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Función principal de análisis morfológico de una pieza:
    1. Binariza la pieza (fondo negro, pieza en blanco).
    2. Extrae contorno externo.
    3. Detecta esquinas y clasifica los 4 lados (PLANO, MACHO, HEMBRA) y sus funciones de curva.
    """
    binary = binarize_piece(img_rgb, method="otsu")
    contour = extract_external_contour(binary)
    sides_info = detect_corners_and_split_sides(contour, binary)
    
    return {
        "binary_mask": binary,
        "contour": contour,
        "sides": sides_info
    }


def compute_jigsaw_shape_compatibility(
    side_a: Dict[str, Any],
    side_b: Dict[str, Any]
) -> float:
    """
    Compara dos lados para verificar compatibilidad de encastre:
    - Si ambos son MACHO o ambos son HEMBRA -> Retorna 1e6 (incompatibles).
    - Si alguno es PLANO -> Retorna 1e6 (bordes planos no encajan internamente).
    - Si uno es MACHO y otro HEMBRA -> Retorna la distancia de perfiles cuadrática
      más una penalización si sus funciones de curva específicas difieren.
    """
    type_a = side_a["type"]
    type_b = side_b["type"]
    
    if type_a == "PLANO" or type_b == "PLANO":
        return 1e6
        
    is_complementary = (type_a == "MACHO" and type_b == "HEMBRA") or (type_a == "HEMBRA" and type_b == "MACHO")
    if not is_complementary:
        return 1e6
        
    # Perfiles complementarios alineados
    prof_a = side_a["profile"]
    prof_b_inv = -side_b["profile"] # Hembra se invierte para comparar con Macho
    
    diff = prof_a - prof_b_inv
    base_mse = float(np.mean(diff ** 2))
    
    # Penalización por discrepancia en la función de curva específica
    c_a = side_a.get("curve_type", "standard")
    c_b = side_b.get("curve_type", "standard")
    are_equiv = (c_a == c_b) or (c_a in ("wide", "random") and c_b in ("wide", "random"))
    curve_penalty = 100.0 if (c_a != "none" and c_b != "none" and not are_equiv) else 0.0
    
    return base_mse + curve_penalty
