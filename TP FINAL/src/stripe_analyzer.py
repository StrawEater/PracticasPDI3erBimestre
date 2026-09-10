"""
Módulo de Procesamiento Digital de Imágenes para Análisis de Rayas y Orientación.
Implementa:
1. Generación y aplicación de patrones periódicos de rayas horizontales.
2. Estimación de ángulo de rotación mediante:
   - Histograma ponderado de direcciones del gradiente (Sobel).
   - Espectro de potencia 2D en frecuencia (Transformada de Fourier - 2D FFT).
3. Rectificación geométrica (deskewing) de piezas con rotaciones continuas/leves.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

try:
    from .utils import rotate_image_expand
except ImportError:
    from utils import rotate_image_expand


def apply_horizontal_stripes(
    img_rgb: np.ndarray,
    period: int = 8,
    amplitude: float = 0.30
) -> np.ndarray:
    """
    Aplica una modulación de rayas periódicas horizontales a la imagen completa.
    
    I_mod(y, x) = I(y, x) * (1 - amplitude * sin^2(pi * y / period))
    
    Args:
        img_rgb: Imagen de entrada RGB uint8.
        period: Espaciado espacial de las líneas en píxeles.
        amplitude: Profundidad de modulación (0.0 a 0.8).
        
    Returns:
        Imagen con filtro de rayas horizontales incorporado.
    """
    h, w, c = img_rgb.shape
    y_coords = np.arange(h, dtype=np.float32)[:, None]
    
    # Patrón horizontal 1D repetido a lo largo del ancho
    pattern_1d = 1.0 - amplitude * (np.sin(np.pi * y_coords / float(period)) ** 2)
    pattern_2d = np.repeat(pattern_1d, w, axis=1)[:, :, None]
    
    striped = np.clip(img_rgb.astype(np.float32) * pattern_2d, 0, 255).astype(np.uint8)
    return striped


def detect_stripe_orientation_gradient(
    img_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_bins: int = 180
) -> float:
    """
    Estima el ángulo de inclinación de las rayas (en grados [-90, +90])
    mediante el histograma de orientaciones de gradientes Sobel ponderado por magnitud.
    
    Rayas horizontales puras tienen gradientes normales verticales (90°).
    Si la imagen está rotada theta grados, las rayas y sus normales rotan theta grados.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Máscara interna para evitar el borde de recorte con el fondo negro
    if mask is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        eroded_mask = cv2.erode(mask, kernel)
    else:
        # Excluir píxeles negros de fondo
        eroded_mask = (gray > 10).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        eroded_mask = cv2.erode(eroded_mask, kernel)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    # En coordenadas de imagen (y hacia abajo), rotar antihorario corresponde a -arctan2(gy, gx)
    angle_deg = -np.rad2deg(np.arctan2(gy, gx))
    
    # Mapear a rango axial [0, 180) ya que rayas son simétricas respecto a dirección opuesta
    angle_axial = np.mod(angle_deg, 180.0)
    
    valid_idx = (eroded_mask > 0) & (magnitude > 15.0)
    if not np.any(valid_idx):
        return 0.0
        
    angles_valid = angle_axial[valid_idx]
    mags_valid = magnitude[valid_idx]
    
    # Histograma ponderado
    counts, bin_edges = np.histogram(angles_valid, bins=num_bins, range=(0.0, 180.0), weights=mags_valid)
    
    # El pico dominante de gradiente normal corresponde a la perpendicular de las rayas
    peak_bin = np.argmax(counts)
    peak_normal_angle = 0.5 * (bin_edges[peak_bin] + bin_edges[peak_bin + 1])
    
    # El ángulo de la raya es perpendicular al gradiente (raya_angle = normal - 90°)
    stripe_angle = peak_normal_angle - 90.0
    
    # Normalizar a [-90, +90]
    if stripe_angle > 90.0:
        stripe_angle -= 180.0
    elif stripe_angle < -90.0:
        stripe_angle += 180.0
        
    return float(stripe_angle)


def detect_stripe_orientation_fft(
    img_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    expected_period: int = 8
) -> float:
    """
    Estima el ángulo de inclinación de las rayas en el dominio de Fourier 2D.
    Las rayas periódicas generan picos espectrales a lo largo del eje perpendicular a las mismas.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # Ventana de Hann para reducir fugas espectrales (spectral leakage)
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    if mask is not None:
        pmask = (mask > 0).astype(np.float32)
    else:
        pmask = (gray > 10).astype(np.float32)
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    eroded_mask = cv2.erode(pmask, kernel)
    window = window * eroded_mask
    
    windowed = (gray - np.mean(gray)) * window
    
    # FFT 2D
    f_transform = np.fft.fft2(windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log1p(np.abs(f_shift))
    
    cy, cx = h // 2, w // 2
    target_freq = min(h, w) / float(expected_period)
    
    angles = np.linspace(-90, 89, 180)
    radial_energies = []
    
    for ang in angles:
        ang_rad = np.deg2rad(ang)
        radii = np.linspace(target_freq * 0.6, target_freq * 1.4, 20)
        sample_x = np.clip(np.round(cx + radii * np.cos(ang_rad)).astype(int), 0, w - 1)
        sample_y = np.clip(np.round(cy + radii * np.sin(ang_rad)).astype(int), 0, h - 1)
        
        energy = np.sum(magnitude_spectrum[sample_y, sample_x])
        radial_energies.append(energy)
        
    best_ang_idx = int(np.argmax(radial_energies))
    spectral_peak_angle = angles[best_ang_idx]
    
    # En coordenadas de imagen (y hacia abajo), el ángulo de la raya respecto a la horizontal:
    stripe_angle = -(spectral_peak_angle - 90.0)
    while stripe_angle > 90.0:
        stripe_angle -= 180.0
    while stripe_angle < -90.0:
        stripe_angle += 180.0
        
    return float(stripe_angle)


def detect_stripe_orientation(
    img_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Método combinado robusto para estimar el ángulo de rotación de las rayas (en grados).
    Utiliza el espectro de potencia 2D FFT enfocado en la frecuencia periódica de las rayas.
    """
    return detect_stripe_orientation_fft(img_rgb, mask=mask)


def rectify_piece_rotation(
    img_rgb: np.ndarray,
    angle_degrees: float,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rota la pieza un ángulo especificado para enderezarla (deskewing)
    alrededor de su centro geométrico, manteniendo fondo negro puro.
    
    Args:
        img_rgb: Imagen RGB de la pieza.
        angle_degrees: Ángulo a rotar (en sentido horario).
        mask: Máscara binaria opcional (255 pieza, 0 fondo).
        
    Returns:
        (rectified_img, rectified_mask)
    """
    if abs(angle_degrees) < 0.2:
        if mask is None:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            mask = (gray > 5).astype(np.uint8) * 255
        return img_rgb.copy(), mask.copy()
        
    # Rotar con expansión de lienzo para evitar cualquier recorte perimetral
    rectified_img = rotate_image_expand(img_rgb, angle_degrees, padding=35)
    gray = cv2.cvtColor(rectified_img, cv2.COLOR_RGB2GRAY)
    rectified_mask = (gray > 5).astype(np.uint8) * 255
    
    return rectified_img, rectified_mask
