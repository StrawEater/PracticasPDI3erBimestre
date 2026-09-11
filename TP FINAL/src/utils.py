"""
Utilidades auxiliares de entrada/salida, transformaciones geométricas y visualización.
Todas las operaciones internas trabajan en formato RGB.
"""

import json
import os
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


def load_image(filepath: str) -> np.ndarray:
    """Carga una imagen desde disco y la convierte a RGB."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No se encontró la imagen en: {filepath}")
    bgr = cv2.imread(filepath)
    if bgr is None:
        raise ValueError(f"No se pudo decodificar la imagen: {filepath}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image(filepath: str, img_rgb: np.ndarray) -> None:
    """Guarda una imagen RGB en disco en formato BGR."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, bgr)


def load_json(filepath: str) -> dict:
    """Carga un archivo JSON con codificación UTF-8."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(filepath: str, data: dict) -> None:
    """Guarda un diccionario como JSON formateado."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def rotate_image_clockwise(img: np.ndarray, angle_degrees: int) -> np.ndarray:
    """
    Rota una imagen en sentido horario por 0, 90, 180 o 270 grados exactos.
    """
    angle = angle_degrees % 360
    if angle == 0:
        return img.copy()
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(f"Ángulo de rotación no soportado: {angle_degrees}. Debe ser múltiplo de 90°.")


def rotate_image_expand(
    img: np.ndarray,
    angle_degrees: float,
    padding: int = 40,
    border_value: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Rota una imagen cualquier ángulo (continuo u ortogonal) expandiendo dinámicamente el lienzo
    para que NUNCA se recorte ningún píxel útil de la pieza ni sus pestañas, asegurando
    un margen perimetral negro holgado alrededor de toda la pieza.
    
    Args:
        img: Imagen en formato RGB (H, W, 3) o monocanal (H, W).
        angle_degrees: Ángulo a rotar en sentido antihorario (convención OpenCV).
                       Para rotar en sentido horario, pasar ángulo negativo.
        padding: Margen adicional de fondo negro en píxeles.
        border_value: Valor del color de fondo exterior.
    """
    if abs(angle_degrees) < 1e-3:
        if padding > 0:
            return cv2.copyMakeBorder(
                img, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=border_value
            )
        return img.copy()
        
    h, w = img.shape[:2]
    rad = np.deg2rad(angle_degrees)
    cos_a = abs(np.cos(rad))
    sin_a = abs(np.sin(rad))
    
    # Bounding box del rectángulo rotado con holgura segura
    new_w = int(np.ceil(h * sin_a + w * cos_a)) + 2 * padding
    new_h = int(np.ceil(h * cos_a + w * sin_a)) + 2 * padding
    
    # Preservar paridad con (w, h) para alineación subpixel idéntica del centro
    if (new_w - w) % 2 != 0:
        new_w += 1
    if (new_h - h) % 2 != 0:
        new_h += 1
        
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_degrees, 1.0)
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    
    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return rotated


def estimate_cell_size(
    pieces: Dict[int, np.ndarray],
    rows: int,
    cols: int,
    pieces_info: Optional[Dict[str, Any]] = None
) -> Tuple[int, int]:
    """
    Determina de forma analítica exacta o subpíxel el tamaño nominal (tile_h, tile_w) de las celdas
    a partir de las 4 esquinas geométricas de las piezas o de su geometría intrínseca.
    """
    if pieces_info is not None and "cell_size" in pieces_info:
        return tuple(pieces_info["cell_size"])
        
    sample = next(iter(pieces.values()))
    is_jigsaw = bool(np.mean(np.all(sample <= 5, axis=-1)) > 0.02) if sample.ndim == 3 else bool(np.mean(sample <= 5) > 0.02)
    if not is_jigsaw:
        return (sample.shape[0], sample.shape[1])
        
    try:
        try:
            from .shape_matcher import binarize_piece, extract_external_contour, detect_jigsaw_corners
        except ImportError:
            from shape_matcher import binarize_piece, extract_external_contour, detect_jigsaw_corners
            
        corner_widths = []
        corner_heights = []
        for p_img in pieces.values():
            try:
                mask = binarize_piece(p_img)
                cnt = extract_external_contour(mask)
                tl, tr, br, bl = detect_jigsaw_corners(cnt)
                w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
                h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
                corner_widths.append(w)
                corner_heights.append(h)
            except Exception:
                continue
                
        if corner_widths and corner_heights:
            dims_min = [min(w, h) for w, h in zip(corner_widths, corner_heights)]
            dims_max = [max(w, h) for w, h in zip(corner_widths, corner_heights)]
            ratio = np.median(dims_max) / max(1.0, np.median(dims_min))
            if ratio < 1.08:
                side = int(round((np.median(dims_min) + np.median(dims_max)) / 2.0))
                return (side, side)
            else:
                w_greater = sum(1 for w, h in zip(corner_widths, corner_heights) if w >= h)
                if w_greater >= len(corner_widths) / 2.0:
                    tile_w = int(round(np.median(dims_max)))
                    tile_h = int(round(np.median(dims_min)))
                else:
                    tile_w = int(round(np.median(dims_min)))
                    tile_h = int(round(np.median(dims_max)))
                return (tile_h, tile_w)
    except Exception:
        pass
        
    # Fallback geométrico exacto por ratio de recorte (crop_w = 1.5*tile_w + 110)
    h_crop, w_crop = sample.shape[:2]
    est_w = max(10, int(round((w_crop - 110) / 1.5)))
    est_h = max(10, int(round((h_crop - 110) / 1.5)))
    return (est_h, est_w)


def assemble_puzzle(
    grid: List[List[int]],
    pieces: Dict[int, np.ndarray],
    rotations: Optional[Dict[int, int]] = None,
    continuous_tilts: Optional[Dict[int, float]] = None,
    pieces_info: Optional[Dict[str, Any]] = None,
    cell_size: Optional[Tuple[int, int]] = None,
    draw_lines: bool = False,
    line_color: Tuple[int, int, int] = (255, 0, 0),
    line_thickness: int = 1
) -> np.ndarray:
    """
    Ensambla la grilla del rompecabezas orientando cada pieza a su posición predicha
    y colocándola de forma que encastren e interbloqueen sus ranuras (sockets) y salientes (tabs)
    sin recortes ni marcos negros espurios.
    
    Args:
        grid: Matriz de tamaño [rows][cols] con los IDs de las piezas.
        pieces: Diccionario {piece_id: imagen_rgb}.
        rotations: Diccionario opcional {piece_id: grados_ortogonales} (0, 90, 180, 270 horario).
        continuous_tilts: Diccionario opcional {piece_id: inclinacion_leve_grados} para deskewing.
        pieces_info: Diccionario opcional con metadatos del ground truth.
        cell_size: Tupla opcional (tile_h, tile_w) de dimensiones nominales de cada celda.
        draw_lines: Si es True, dibuja líneas divisorias de cuadrícula.
    """
    rows = len(grid)
    cols = len(grid[0])
    
    # 1. Orientar cada pieza a su orientación predicha
    oriented_pieces = {}
    for p_id, orig_tile in pieces.items():
        # Solo procesar las piezas que están presentes en la grilla actual
        found = any(p_id in row for row in grid)
        if not found:
            continue
            
        tile = orig_tile.copy()
        
        # A. Enderezar inclinación continua / leve detectada por rayas
        if continuous_tilts is not None and p_id in continuous_tilts:
            tilt = continuous_tilts[p_id]
            if abs(tilt) >= 0.2:
                tile = rotate_image_expand(tile, -tilt, padding=0)
                
        # B. Rotar rotación ortogonal predicha (en sentido horario)
        if rotations is not None and p_id in rotations:
            deg = rotations[p_id] % 360
            if deg != 0:
                tile = rotate_image_clockwise(tile, deg)
                
        oriented_pieces[p_id] = tile

    # 2. Detectar si son piezas de rompecabezas con encastres (jigsaw) o rectangulares (cuadrícula)
    sample_id = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in oriented_pieces:
                sample_id = grid[r][c]
                break
        if sample_id is not None:
            break
            
    if sample_id is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    sample_img = oriented_pieces[sample_id]
    is_jigsaw = bool(np.mean(np.all(sample_img <= 5, axis=-1)) > 0.02) if sample_img.ndim == 3 else bool(np.mean(sample_img <= 5) > 0.02)
    
    # 3. Determinar el tamaño nominal de celda (tile_h, tile_w) con precisión subpíxel
    if cell_size is not None:
        tile_h, tile_w = cell_size
    else:
        tile_h, tile_w = estimate_cell_size(oriented_pieces, rows, cols, pieces_info=pieces_info)

    # 4. Dimensiones del lienzo completo
    canvas_h = rows * tile_h
    canvas_w = cols * tile_w
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # 5. Ensamble por encastre (interlocking jigsaw assembly)
    for r in range(rows):
        for c in range(cols):
            p_id = grid[r][c]
            if p_id not in oriented_pieces:
                continue
                
            tile = oriented_pieces[p_id]
            h_tile, w_tile = tile.shape[:2]
            
            if not is_jigsaw:
                if h_tile != tile_h or w_tile != tile_w:
                    tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
                canvas[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
            else:
                # Máscara estricta de la pieza: preserva zonas oscuras con textura (> 0)
                mask = np.any(tile > 0, axis=-1) if tile.ndim == 3 else (tile > 0)
                if not np.any(mask):
                    continue
                    
                target_cx = (c + 0.5) * tile_w
                target_cy = (r + 0.5) * tile_h
                
                # El centro nominal de la celda de la pieza está en el centro geométrico de la imagen
                cx = w_tile / 2.0
                cy = h_tile / 2.0
                
                tl_x = int(round(target_cx - cx))
                tl_y = int(round(target_cy - cy))
                
                src_y0 = max(0, -tl_y)
                src_x0 = max(0, -tl_x)
                dst_y0 = max(0, tl_y)
                dst_x0 = max(0, tl_x)
                
                src_y1 = min(h_tile, canvas_h - tl_y)
                src_x1 = min(w_tile, canvas_w - tl_x)
                dst_y1 = dst_y0 + (src_y1 - src_y0)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                
                if src_y1 > src_y0 and src_x1 > src_x0:
                    sub_mask = mask[src_y0:src_y1, src_x0:src_x1]
                    sub_tile = tile[src_y0:src_y1, src_x0:src_x1]
                    
                    canvas_region = canvas[dst_y0:dst_y1, dst_x0:dst_x1]
                    # Solo copiamos en píxeles útiles de la pieza y que no destruyan piezas ya colocadas:
                    # Las pestañas salientes (tabs) encastran en las ranuras (sockets) vacías
                    # sin sobrescribir con rectángulos negros ni superponerse arriba de piezas previas.
                    empty_pixels = np.all(canvas_region == 0, axis=-1)
                    write_mask = sub_mask & empty_pixels
                    canvas_region[write_mask] = sub_tile[write_mask]

    # 6. Opcional: líneas divisorias de grilla
    if draw_lines:
        for r in range(1, rows):
            y = r * tile_h
            cv2.line(canvas, (0, y), (canvas_w, y), line_color, line_thickness)
        for c in range(1, cols):
            x = c * tile_w
            cv2.line(canvas, (x, 0), (x, canvas_h), line_color, line_thickness)
            
    return canvas
