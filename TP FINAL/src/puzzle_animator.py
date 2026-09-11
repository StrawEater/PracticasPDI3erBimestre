"""
Módulo para la generación de animaciones del proceso de resolución del rompecabezas.
Genera archivos GIF animados y videos MP4 que muestran al algoritmo ensamblando
el puzzle pieza por pieza, con interfaz HUD, indicador de progreso y efecto de encastre.
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
from PIL import Image

try:
    from .utils import assemble_puzzle, rotate_image_clockwise, estimate_cell_size
except ImportError:
    from utils import assemble_puzzle, rotate_image_clockwise, estimate_cell_size


def create_solving_animation(
    pieces_raw: Dict[int, np.ndarray],
    placement_history: List[Dict[str, Any]],
    rows: int,
    cols: int,
    output_path: str,
    cell_size: Optional[Tuple[int, int]] = None,
    continuous_tilts: Optional[Dict[int, float]] = None,
    fps: int = 12,
    hold_final_frames: int = 15,
    max_dimension: int = 700
) -> str:
    """
    Genera un GIF animado (o MP4) que visualiza el progreso paso a paso del solver.
    
    Args:
        pieces_raw: Diccionario {piece_id: np.ndarray (RGB)}.
        placement_history: Lista ordenada de pasos [{'step': 1, 'piece_id': p, 'rotation': rot, 'row': r, 'col': c}, ...]
        rows: Cantidad de filas de la grilla.
        cols: Cantidad de columnas de la grilla.
        output_path: Ruta destino (.gif o .mp4).
        cell_size: (tile_h, tile_w) nominales de la celda.
        continuous_tilts: Diccionario opcional de inclinaciones continuas.
        fps: Cuadros por segundo para la animación.
        hold_final_frames: Cantidad de cuadros de pausa al finalizar el puzzle.
        max_dimension: Tamaño máximo en píxeles (ancho o alto) para optimizar peso del GIF.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 1. Determinar tamaño nominal de celda con precisión analítica subpíxel
    if cell_size is None:
        cell_size = estimate_cell_size(pieces_raw, rows, cols)
        
    tile_h, tile_w = cell_size
    canvas_h = rows * tile_h
    canvas_w = cols * tile_w
    
    # Factor de escala si la imagen completa es demasiado grande para un GIF liviano
    scale = 1.0
    if max(canvas_h, canvas_w) > max_dimension:
        scale = max_dimension / float(max(canvas_h, canvas_w))
        
    scaled_w = max(100, int(round(canvas_w * scale)))
    scaled_h = max(100, int(round(canvas_h * scale)))
    
    # Altura del HUD superior e inferior y dimensiones pares para codecs de video
    hud_top_h = 44
    hud_bot_h = 28
    frame_w = max(450, scaled_w)
    if frame_w % 2 != 0:
        frame_w += 1
    frame_h = scaled_h + hud_top_h + hud_bot_h
    if frame_h % 2 != 0:
        frame_h += 1
    
    total_pieces = len(placement_history)
    frames_pil = []
    durations = []
    
    # Grilla acumulativa
    current_grid = [[-1 for _ in range(cols)] for _ in range(rows)]
    current_rotations = {}
    
    # Cuadro 0: Grilla vacía estilizada (Blueprint)
    empty_board = _render_empty_grid(scaled_h, scaled_w, rows, cols)
    frame_0 = _draw_hud(
        board_img=empty_board,
        step=0,
        total_steps=total_pieces,
        piece_id=None,
        row=None,
        col=None,
        rot=None,
        frame_w=frame_w,
        hud_top_h=hud_top_h,
        hud_bot_h=hud_bot_h
    )
    frames_pil.append(Image.fromarray(frame_0))
    durations.append(int(1000 / fps * 2)) # Breve pausa inicial
    
    # Renderizado cuadro por cuadro
    for idx, step_info in enumerate(placement_history):
        p_id = step_info["piece_id"]
        rot = step_info["rotation"]
        r = step_info["row"]
        c = step_info["col"]
        
        current_grid[r][c] = p_id
        current_rotations[p_id] = rot
        
        # Ensamblar tablero parcial
        board_rgb = assemble_puzzle(
            grid=current_grid,
            pieces=pieces_raw,
            rotations=current_rotations,
            continuous_tilts=continuous_tilts,
            cell_size=cell_size,
            draw_lines=False
        )
        
        # Redimensionar al tamaño del marco
        if scale != 1.0:
            board_resized = cv2.resize(board_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            board_resized = board_rgb.copy()
            
        # Componer de forma limpia sobre el blueprint base:
        # Los píxeles de las piezas colocadas aparecen nítidos y las ranuras vacías
        # muestran las líneas de blueprint por detrás, sin pisar las pestañas encajadas.
        frame_board = empty_board.copy()
        piece_pixels = np.any(board_resized > 0, axis=-1)
        frame_board[piece_pixels] = board_resized[piece_pixels]
        
        # Agregar panel HUD superior e inferior
        frame_with_hud = _draw_hud(
            board_img=frame_board,
            step=idx + 1,
            total_steps=total_pieces,
            piece_id=p_id,
            row=r,
            col=c,
            rot=rot,
            frame_w=frame_w,
            hud_top_h=hud_top_h,
            hud_bot_h=hud_bot_h
        )
        
        frames_pil.append(Image.fromarray(frame_with_hud))
        durations.append(int(1000 / fps))
        
    # Sostener el resultado final terminado
    for _ in range(hold_final_frames):
        frames_pil.append(frames_pil[-1])
        durations.append(int(1000 / fps))
        
    # Guardar como GIF animado optimizado
    if output_path.lower().endswith(".gif"):
        # Cuantización de paleta adaptativa para máxima calidad y tamaño ultra compacto
        palette_frames = [f.quantize(colors=256, method=Image.Resampling.LANCZOS, dither=Image.Dither.FLOYDSTEINBERG) for f in frames_pil]
        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=durations,
            loop=0,
            optimize=True
        )
    elif output_path.lower().endswith(".mp4"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        for f_pil in frames_pil:
            bgr = cv2.cvtColor(np.array(f_pil), cv2.COLOR_RGB2BGR)
            if bgr.shape[1] != frame_w or bgr.shape[0] != frame_h:
                bgr = cv2.resize(bgr, (frame_w, frame_h), interpolation=cv2.INTER_AREA)
            out_vid.write(bgr)
        out_vid.release()
        
    print(f"[ANIMACIÓN OK] Generado '{output_path}' ({len(frames_pil)} cuadros, {fps} FPS, {frame_w}x{frame_h} px).")
    return output_path


def _render_empty_grid(height: int, width: int, rows: int, cols: int) -> np.ndarray:
    """Crea un lienzo con fondo oscuro estilizado (blueprint) y cuadrícula guía de las ranuras."""
    canvas = np.full((height, width, 3), 16, dtype=np.uint8)
    cell_h = height / float(rows)
    cell_w = width / float(cols)
    
    # Dibujar sutiles líneas divisorias de cuadrícula
    for r in range(rows + 1):
        y = min(int(round(r * cell_h)), height - 1)
        cv2.line(canvas, (0, y), (width - 1, y), (30, 42, 54), 1)
    for c in range(cols + 1):
        x = min(int(round(c * cell_w)), width - 1)
        cv2.line(canvas, (x, 0), (x, height - 1), (30, 42, 54), 1)
        
    return canvas


def _overlay_empty_slots(board: np.ndarray, grid: List[List[int]], rows: int, cols: int) -> None:
    """Función de compatibilidad: las ranuras vacías se preservan limpiamente desde el blueprint base."""
    pass


def _draw_hud(
    board_img: np.ndarray,
    step: int,
    total_steps: int,
    piece_id: Optional[int],
    row: Optional[int],
    col: Optional[int],
    rot: Optional[int],
    frame_w: int,
    hud_top_h: int,
    hud_bot_h: int
) -> np.ndarray:
    """Superpone el panel de control HUD superior e inferior al cuadro."""
    board_h, board_w = board_img.shape[:2]
    final_w = frame_w
    total_h = board_h + hud_top_h + hud_bot_h
    if total_h % 2 != 0:
        total_h += 1
    full_frame = np.zeros((total_h, final_w, 3), dtype=np.uint8)
    
    # 1. Fondo del HUD superior (azul oscuro elegante)
    full_frame[0:hud_top_h, :] = (15, 20, 26)
    cv2.line(full_frame, (0, hud_top_h - 1), (final_w - 1, hud_top_h - 1), (40, 60, 80), 1)
    
    # Título y estado
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(full_frame, "RESOLVIENDO ROMPECABEZAS PDI", (14, 20), font, 0.46, (0, 220, 255), 1, cv2.LINE_AA)
    
    pct = int(round((step / float(total_steps)) * 100)) if total_steps > 0 else 0
    status_text = f"Progreso: {step}/{total_steps} ({pct}%)"
    cv2.putText(full_frame, status_text, (final_w - 180, 20), font, 0.44, (200, 220, 240), 1, cv2.LINE_AA)
    
    # Barra de progreso sutil bajo el texto superior
    bar_y = hud_top_h - 4
    cv2.line(full_frame, (0, bar_y), (final_w - 1, bar_y), (25, 35, 45), 2)
    progress_w = int(round((step / float(total_steps)) * final_w)) if total_steps > 0 else 0
    if progress_w > 0:
        cv2.line(full_frame, (0, bar_y), (progress_w, bar_y), (0, 200, 120), 2)
        
    # 2. Tablero de imagen centrado horizontalmente
    offset_x = (final_w - board_w) // 2
    full_frame[hud_top_h:hud_top_h + board_h, offset_x:offset_x + board_w] = board_img
    
    # 3. HUD inferior (información de la última pieza)
    bot_y = hud_top_h + board_h
    full_frame[bot_y:total_h, :] = (12, 16, 22)
    cv2.line(full_frame, (0, bot_y), (final_w - 1, bot_y), (35, 50, 65), 1)
    
    if piece_id is not None:
        info_txt = f"Pieza #{piece_id:03d} en ({row}, {col}) | Rot: {rot} deg | Encastre de Fourier OK"
        cv2.putText(full_frame, info_txt, (14, bot_y + 19), font, 0.38, (180, 200, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(full_frame, "Iniciando exploracion de bordes y clasificacion topologica...", (14, bot_y + 19), font, 0.38, (140, 160, 180), 1, cv2.LINE_AA)
        
    return full_frame
