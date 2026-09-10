"""
Módulo de Geometría de Rompecabezas con Encastres (Tabs & Blanks / Macho y Hembra).
Genera curvas analíticas de encastre realistas para cada borde interior de la grilla,
construye máscaras poligonales cerradas para cada pieza y las recorta sobre fondo negro (0, 0, 0).
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional


def generate_tab_curve(
    p_start: Tuple[float, float],
    p_end: Tuple[float, float],
    tab_type: int, # +1: Macho, -1: Hembra, 0: Plano
    profile_type: str = "standard", # 'standard', 'circular', 'wide'
    num_points: int = 100,
    tab_depth_ratio: float = 0.20,
    tab_width_ratio: float = 0.32
) -> np.ndarray:
    """
    Genera los puntos (x, y) de la curva de un borde con encastre analítico realista de rompecabezas.
    
    Args:
        p_start, p_end: Coordenadas inicial y final del borde nominal.
        tab_type:
             0: Borde plano (rectilíneo)
            +1: Macho (protrusión hacia afuera)
            -1: Hembra (hendidura hacia adentro)
        profile_type:
            'standard': Bulbo redondeado clásico de rompecabezas con cabeza única suave.
            'circular': Encastre abovedado semicircular redondeado.
            'wide' o 'random': Encastre redondeado ancho estilo Ravensburger (cabeza amplia).
    """
    p0 = np.array(p_start, dtype=np.float32)
    p1 = np.array(p_end, dtype=np.float32)
    
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length == 0 or tab_type == 0:
        # Borde recto
        t = np.linspace(0, 1, num_points)[:, None]
        return p0 + t * vec
        
    u = vec / length # Vector unitario tangente
    # Vector normal unitario (rotación 90° horario del vector tangente)
    n = np.array([u[1], -u[0]], dtype=np.float32)
    
    # Parámetros del encastre de rompecabezas real
    depth = length * tab_depth_ratio * float(tab_type)
    width = length * tab_width_ratio
    center = 0.5 * length
    
    t_vals = np.linspace(0, 1, num_points)
    curve_points = []
    
    for t in t_vals:
        s = t * length # Distancia a lo largo del borde
        dist_from_center = (s - center) / (width / 2.0)
        
        if abs(dist_from_center) <= 1.0:
            x_norm = float(dist_from_center)
            if profile_type == "circular":
                # Perfil semicircular abovedado
                bulb = np.sqrt(max(0.0, 1.0 - x_norm ** 2))
            elif profile_type in ("wide", "random"):
                # Perfil redondeado ancho estilo puzzle europeo clásico
                bulb = np.cos(x_norm * np.pi / 2.0) ** 1.3
            else:
                # Perfil estándar clásico: bulbo redondeado suave de cabeza única
                bulb = np.cos(x_norm * np.pi / 2.0) ** 1.6
                
            offset = depth * bulb
        else:
            offset = 0.0
            
        pt = p0 + s * u + offset * n
        curve_points.append(pt)
        
    return np.array(curve_points, dtype=np.float32)


class JigsawGridGeometry:
    """
    Gestiona la coherencia global de los encastres de una grilla R x C.
    Garantiza que si el borde Este de la pieza (r, c) es Macho,
    el borde Oeste de la pieza (r, c+1) sea el encastre Hembra complementario exacto.
    """
    def __init__(
        self,
        rows: int,
        cols: int,
        image_h: int,
        image_w: int,
        seed: Optional[int] = 42,
        profile_types: Optional[List[str]] = None
    ):
        self.rows = rows
        self.cols = cols
        self.img_h = image_h
        self.img_w = image_w
        
        self.tile_h = image_h // rows
        self.tile_w = image_w // cols
        
        if seed is not None:
            np.random.seed(seed)
            
        # Asignar aleatoriamente la orientación de cada borde interior
        # horiz_tabs[r, c] es el encastre vertical entre (r, c) y (r, c+1) para 0 <= c < cols - 1
        # +1 significa que (r, c) es Macho (penetra en c+1) y (r, c+1) es Hembra
        self.horiz_tabs = np.random.choice([1, -1], size=(rows, cols - 1))
        
        # vert_tabs[r, c] es el encastre horizontal entre (r, c) y (r+1, c) para 0 <= r < rows - 1
        # +1 significa que (r, c) es Macho (penetra en r+1) y (r+1, c) es Hembra
        self.vert_tabs = np.random.choice([1, -1], size=(rows - 1, cols))
        
        # Asignar a cada arista interior una función de perfil específica (standard, circular, random)
        self.allowed_profiles = profile_types if profile_types is not None else ["standard", "circular", "random"]
        self.horiz_profile_types = np.random.choice(self.allowed_profiles, size=(rows, cols - 1))
        self.vert_profile_types = np.random.choice(self.allowed_profiles, size=(rows - 1, cols))

        # Precomputar las costuras canónicas compartidas para garantizar 0 huecos y 0 discrepancia
        self._seams_v = {} # Costuras horizontales entre fila r y r+1
        self._seams_h = {} # Costuras verticales entre columna c y c+1
        self._compute_canonical_seams(num_pts=60)

    def _compute_canonical_seams(self, num_pts: int = 60) -> None:
        """Calcula una única vez cada curva de unión interior compartida."""
        # 1. Costuras horizontales: entre celda (r, c) y (r+1, c)
        for r in range(self.rows - 1):
            for c in range(self.cols):
                p0 = np.array([c * self.tile_w, (r + 1) * self.tile_h], dtype=np.float32)
                p1 = np.array([(c + 1) * self.tile_w, (r + 1) * self.tile_h], dtype=np.float32)
                prof = str(self.vert_profile_types[r, c])
                tab_dir = int(self.vert_tabs[r, c]) # +1: hacia abajo (+y), -1: hacia arriba (-y)
                
                u = (p1 - p0) / float(self.tile_w)
                n = np.array([0.0, 1.0], dtype=np.float32) if tab_dir == 1 else np.array([0.0, -1.0], dtype=np.float32)
                
                length = float(self.tile_w)
                depth = length * 0.20
                width = length * 0.32
                center = 0.5 * length
                
                pts = []
                for t in np.linspace(0.0, 1.0, num_pts):
                    s = t * length
                    dist = (s - center) / (width / 2.0)
                    if abs(dist) <= 1.0:
                        x_norm = float(dist)
                        if prof == "circular":
                            bulb = np.sqrt(max(0.0, 1.0 - x_norm ** 2))
                        elif prof in ("wide", "random"):
                            bulb = np.cos(x_norm * np.pi / 2.0) ** 1.3
                        else:
                            bulb = np.cos(x_norm * np.pi / 2.0) ** 1.6
                    else:
                        bulb = 0.0
                    pts.append(p0 + s * u + (depth * bulb) * n)
                self._seams_v[(r, c)] = np.array(pts, dtype=np.float32)

        # 2. Costuras verticales: entre celda (r, c) y (r, c+1)
        for r in range(self.rows):
            for c in range(self.cols - 1):
                p0 = np.array([(c + 1) * self.tile_w, r * self.tile_h], dtype=np.float32)
                p1 = np.array([(c + 1) * self.tile_w, (r + 1) * self.tile_h], dtype=np.float32)
                prof = str(self.horiz_profile_types[r, c])
                tab_dir = int(self.horiz_tabs[r, c]) # +1: hacia la derecha (+x), -1: hacia la izquierda (-x)
                
                u = (p1 - p0) / float(self.tile_h)
                n = np.array([1.0, 0.0], dtype=np.float32) if tab_dir == 1 else np.array([-1.0, 0.0], dtype=np.float32)
                
                length = float(self.tile_h)
                depth = length * 0.20
                width = length * 0.32
                center = 0.5 * length
                
                pts = []
                for t in np.linspace(0.0, 1.0, num_pts):
                    s = t * length
                    dist = (s - center) / (width / 2.0)
                    if abs(dist) <= 1.0:
                        x_norm = float(dist)
                        if prof == "circular":
                            bulb = np.sqrt(max(0.0, 1.0 - x_norm ** 2))
                        elif prof in ("wide", "random"):
                            bulb = np.cos(x_norm * np.pi / 2.0) ** 1.3
                        else:
                            bulb = np.cos(x_norm * np.pi / 2.0) ** 1.6
                    else:
                        bulb = 0.0
                    pts.append(p0 + s * u + (depth * bulb) * n)
                self._seams_h[(r, c)] = np.array(pts, dtype=np.float32)

    def get_piece_edge_types(self, r: int, c: int) -> Dict[str, str]:
        """
        Devuelve el tipo de borde ('PLANO', 'MACHO', 'HEMBRA') para cada lado de la pieza (r, c).
        """
        # Norte (Arriba)
        if r == 0:
            type_n = "PLANO"
        else:
            type_n = "HEMBRA" if self.vert_tabs[r - 1, c] == 1 else "MACHO"
            
        # Sur (Abajo)
        if r == self.rows - 1:
            type_s = "PLANO"
        else:
            type_s = "MACHO" if self.vert_tabs[r, c] == 1 else "HEMBRA"
            
        # Oeste (Izquierda)
        if c == 0:
            type_w = "PLANO"
        else:
            type_w = "HEMBRA" if self.horiz_tabs[r, c - 1] == 1 else "MACHO"
            
        # Este (Derecha)
        if c == self.cols - 1:
            type_e = "PLANO"
        else:
            type_e = "MACHO" if self.horiz_tabs[r, c] == 1 else "HEMBRA"
            
        return {"N": type_n, "S": type_s, "W": type_w, "E": type_e}

    def get_piece_edge_curves(self, r: int, c: int) -> Dict[str, str]:
        """
        Devuelve el nombre de la función analítica ('none', 'standard', 'circular', 'random')
        de cada lado de la pieza (r, c).
        """
        prof_n = "none" if r == 0 else str(self.vert_profile_types[r - 1, c])
        prof_s = "none" if r == self.rows - 1 else str(self.vert_profile_types[r, c])
        prof_w = "none" if c == 0 else str(self.horiz_profile_types[r, c - 1])
        prof_e = "none" if c == self.cols - 1 else str(self.horiz_profile_types[r, c])
        return {"N": prof_n, "S": prof_s, "W": prof_w, "E": prof_e}

    def get_piece_edges(self, r: int, c: int) -> Dict[str, np.ndarray]:
        """Retorna las 4 curvas orientadas del contorno de la pieza (r, c)."""
        num_pts = 60
        # Norte: de (c*w, r*h) a ((c+1)*w, r*h)
        if r == 0:
            curve_n = np.array([[x, 0.0] for x in np.linspace(c * self.tile_w, (c + 1) * self.tile_w, num_pts)], dtype=np.float32)
        else:
            curve_n = self._seams_v[(r - 1, c)].copy()
            
        # Este: de ((c+1)*w, r*h) a ((c+1)*w, (r+1)*h)
        if c == self.cols - 1:
            curve_e = np.array([[(c + 1) * self.tile_w, y] for y in np.linspace(r * self.tile_h, (r + 1) * self.tile_h, num_pts)], dtype=np.float32)
        else:
            curve_e = self._seams_h[(r, c)].copy()
            
        # Sur: de ((c+1)*w, (r+1)*h) a (c*w, (r+1)*h)
        if r == self.rows - 1:
            curve_s = np.array([[x, (r + 1) * self.tile_h] for x in np.linspace((c + 1) * self.tile_w, c * self.tile_w, num_pts)], dtype=np.float32)
        else:
            curve_s = self._seams_v[(r, c)][::-1].copy()
            
        # Oeste: de (c*w, (r+1)*h) a (c*w, r*h)
        if c == 0:
            curve_w = np.array([[0.0, y] for y in np.linspace((r + 1) * self.tile_h, r * self.tile_h, num_pts)], dtype=np.float32)
        else:
            curve_w = self._seams_h[(r, c - 1)][::-1].copy()
            
        return {"N": curve_n, "E": curve_e, "S": curve_s, "W": curve_w}

    def get_piece_polygon(self, r: int, c: int, num_pts_per_edge: int = 60) -> np.ndarray:
        """
        Genera el contorno poligonal 2D cerrado de la pieza (r, c) en coordenadas de la imagen completa.
        El contorno se recorre en sentido horario: Norte -> Este -> Sur -> Oeste.
        """
        edges = self.get_piece_edges(r, c)
        polygon = np.vstack([edges["N"], edges["E"], edges["S"], edges["W"]])
        return polygon

    def extract_piece_image(
        self,
        full_image_rgb: np.ndarray,
        r: int,
        c: int,
        padding: int = 55
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Recorta la pieza correspondiente a (r, c) con su forma real de encastre sobre fondo negro (0, 0, 0).
        Garantiza un margen perimetral negro espacioso (>= padding px) alrededor de toda la pieza (incluidos
        los bordes planos exteriores) y posiciona el centro nominal de la celda exactamente en el centro
        geométrico de la imagen extraída.
        
        Returns:
            - piece_rgb: Imagen recortada centrada (fondo negro = 0, pieza con textura)
            - piece_mask: Máscara binaria (255 dentro de la pieza, 0 en el fondo)
            - (offset_x, offset_y): Coordenadas globales de la esquina superior izquierda del lienzo
        """
        poly = self.get_piece_polygon(r, c)
        
        # Margen simétrico entero a lo largo de cada eje (holgura segura >= padding px en cada lado)
        pad_x = int(np.ceil(self.tile_w * 0.25)) + padding
        pad_y = int(np.ceil(self.tile_h * 0.25)) + padding
        
        crop_w = self.tile_w + 2 * pad_x
        crop_h = self.tile_h + 2 * pad_y
        
        min_x = c * self.tile_w - pad_x
        max_x = min_x + crop_w
        min_y = r * self.tile_h - pad_y
        max_y = min_y + crop_h
        
        piece_rgb = np.zeros((crop_h, crop_w, full_image_rgb.shape[2]), dtype=full_image_rgb.dtype)
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        
        # Zona de intersección válida con la imagen fuente
        src_x0 = max(0, min_x)
        src_x1 = min(self.img_w, max_x)
        src_y0 = max(0, min_y)
        src_y1 = min(self.img_h, max_y)
        
        dst_x0 = src_x0 - min_x
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y0 = src_y0 - min_y
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        
        if src_x1 > src_x0 and src_y1 > src_y0:
            piece_rgb[dst_y0:dst_y1, dst_x0:dst_x1] = full_image_rgb[src_y0:src_y1, src_x0:src_x1]
            
        # Polígono en coordenadas locales del lienzo centrado
        local_poly = poly.copy()
        local_poly[:, 0] -= min_x
        local_poly[:, 1] -= min_y
        local_poly_int = np.round(local_poly).astype(np.int32)
        
        cv2.fillPoly(mask, [local_poly_int], 255)
        piece_rgb = cv2.bitwise_and(piece_rgb, piece_rgb, mask=mask)
        
        return piece_rgb, mask, (min_x, min_y)
