from __future__ import annotations

import os
from typing import List, Iterable, Optional

import numpy as np
import joblib
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
# ---------- Augmentations específicas para zipper ----------
# --- AUGS SEGURAS EN HxWx1 (float32) ---
import numpy as np
import albumentations as A

import numpy as np
import cv2

def _brain_centroid_radius(x2d: np.ndarray) -> tuple[float,float,float]:
    # x2d: HxW float32 (antes de ToTensorV2)
    h, w = x2d.shape
    # umbral robusto
    thr = np.percentile(x2d, 75)
    mask = (x2d > thr).astype(np.uint8)
    # limpia
    mask = cv2.medianBlur(mask*255, 5)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0.5, 0.5, 0.3  # fallback
    # ignora fondo (índice 0)
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cx, cy = centroids[idx]               # (x, y)
    area = stats[idx, cv2.CC_STAT_AREA]
    r = np.sqrt(area/np.pi)               # radio equivalente
    # normaliza
    return float(cx/w), float(cy/h), float(2*r/max(h,w))


def _brain_margins_connected(
    x2d: np.ndarray,
    min_area_frac: float = 0.02,   # % mínimo del área de la imagen para aceptar una componente
    blur_ks: int = 3,              # suavizado previo para Otsu
    close_ks: int = 5,             # cierre morfológico para rellenar huecos
    stripe_clean: bool = True      # limpia rayas finas (zipper) antes de los CC
):
    """
    Calcula márgenes (top, bottom, left, right) usando connected components.
    x2d: HxW float32 en rango libre (se reescala internamente a [0,1]).

    Devuelve: (top, bottom, left, right) en [0,1].
    """
    assert x2d.ndim == 2, "x2d debe ser HxW"
    H, W = x2d.shape

    # 1) Normaliza a [0,1]
    x = x2d.astype(np.float32)
    x = (x - x.min()) / (x.ptp() + 1e-8)

    # 2) Umbral binario robusto (Otsu). Si falla (todo negro), usa percentil.
    x8 = (x * 255).astype(np.uint8)
    if blur_ks > 1:
        x8 = cv2.GaussianBlur(x8, (blur_ks, blur_ks), 0)

    _, mask = cv2.threshold(x8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mask.mean() < 1.0:  # demasiado vacío → fallback percentil
        thr = max(20, int(np.percentile(x * 255, 70)))
        _, mask = cv2.threshold((x * 255).astype(np.uint8), thr, 255, cv2.THRESH_BINARY)

    # 3) Limpieza morfológica
    if stripe_clean:
        # abre con kernels lineales para quitar rayas finas horizontales/verticales
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kh)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kv)

    if close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


    # 4) Connected components
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        # No encontró nada razonable
        return 1.0, 1.0, 1.0, 1.0

    # 5) Elige la componente válida más grande por área
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx_sorted = np.argsort(-areas)  # descendente
    area_min = min_area_frac * (H * W)

    chosen = None
    for k in idx_sorted:
        idx = k + 1  # porque stats[0] = fondo
        a = stats[idx, cv2.CC_STAT_AREA]
        if a < area_min:
            continue
        # opcional: descartar bbox extremadamente finos (artefactos)
        w_box = stats[idx, cv2.CC_STAT_WIDTH]
        h_box = stats[idx, cv2.CC_STAT_HEIGHT]
        if w_box < 0.15 * W or h_box < 0.15 * H:
            continue
        chosen = idx
        break

    if chosen is None:
        return 1.0, 1.0, 1.0, 1.0

    x0 = stats[chosen, cv2.CC_STAT_LEFT]
    y0 = stats[chosen, cv2.CC_STAT_TOP]
    ww = stats[chosen, cv2.CC_STAT_WIDTH]
    hh = stats[chosen, cv2.CC_STAT_HEIGHT]

    # 6) Márgenes normalizados
    top    = y0 / H
    left   = x0 / W
    bottom = (H - (y0 + hh)) / H
    right  = (W - (x0 + ww)) / W

    # clamp por seguridad
    top    = float(np.clip(top,    0.0, 1.0))
    bottom = float(np.clip(bottom, 0.0, 1.0))
    left   = float(np.clip(left,   0.0, 1.0))
    right  = float(np.clip(right,  0.0, 1.0))
    return top, bottom, left, right


class RandomZipperStripe(A.ImageOnlyTransform):
    """
    Zipper sintético eficiente en memoria.
    - No repite arrays (usa broadcasting).
    - Sin normalización por mean/std (escala fija basada en var(sin)=1/2).
    - Devuelve HxWx1 float32 contiguo.
    """
    def __init__(self, p=0.15, max_amp=0.20, min_period=6, max_period=22, axis="rand"):
        super().__init__( p=p)
        self.max_amp = float(max_amp)
        self.min_period = int(min_period)
        self.max_period = int(max_period)
        self.axis = axis  # "x", "y", "rand"

    def get_transform_init_args_names(self):
        return ("max_amp", "min_period", "max_period", "axis")

    def apply(self, img, **params):
        # img: HxWxC (C=1) o HxW
        x = img.astype(np.float32, copy=False)
        if x.ndim == 2:
            x = x[:, :, None]
        H, W, C = x.shape

        amp    = np.random.uniform(0.06, self.max_amp)
        period = np.random.randint(self.min_period, self.max_period + 1)
        phase  = np.random.uniform(0, 2*np.pi)

        # Elegir eje
        axis = self.axis if self.axis != "rand" else ("x" if np.random.rand() < 0.5 else "y")

        # Escala para que el seno tenga std ~ 0.5 sin calcular mean/std por imagen:
        # std(sin)=1/sqrt(2) ~ 0.707 -> multiplicamos por (0.5 / 0.707) ~ 0.707
        scale = 0.5 / np.sqrt(2.0)

        if axis == "x":
            # forma (1, W, 1) -> se difunde a (H, W, 1)
            u = (np.arange(W, dtype=np.float32) * (2*np.pi / period)) + phase
            stripe = np.sin(u, dtype=np.float32)[None, :, None] * scale
        else:
            # forma (H, 1, 1) -> se difunde a (H, W, 1)
            v = (np.arange(H, dtype=np.float32) * (2*np.pi / period)) + phase
            stripe = np.sin(v, dtype=np.float32)[:, None, None] * scale

        # out = x + amp * stripe  (stripe se broadcast sin crear HxW explícito)
        out = x + amp * stripe
        # Asegura contigüidad/writable para siguientes pasos (ToTensorV2/own tensor):
        return np.ascontiguousarray(out, dtype=np.float32)

class RandomBandCut(A.ImageOnlyTransform):
    def __init__(self, p=0.10, max_thick_ratio=0.06):
        super().__init__( p=p)
        self.max_thick_ratio = max_thick_ratio

    def apply(self, img, **params):
        x = img.astype(np.float32)           # HxWx1
        h, w, _ = x.shape
        is_h = np.random.rand() < 0.5
        thick = int(max(1, (h if is_h else w) * np.random.uniform(0.01, self.max_thick_ratio)))
        if is_h:
            y0 = np.random.randint(0, max(1, h - thick))
            x[y0:y0+thick, :, :] = 0.0
        else:
            x0 = np.random.randint(0, max(1, w - thick))
            x[:, x0:x0+thick, :] = 0.0
        return x


class CenterDeBorder(A.ImageOnlyTransform):
    def __init__(self, max_crop=0.04, p=0.5):
        super().__init__( p=p)
        self.max_crop = max_crop

    def apply(self, img, **params):
        # img: HxWx1 float32
        x = img.astype(np.float32)
        h, w, _ = x.shape
        cy = int(h * self.max_crop * np.random.rand())
        cx = int(w * self.max_crop * np.random.rand())
        y0, y1 = cy, h - cy
        x0, x1 = cx, w - cx
        if (y1 - y0) < 8 or (x1 - x0) < 8:
            return x
        crop = x[y0:y1, x0:x1, :]                 # sigue Hc×Wc×1
        # resize con cv2 manteniendo canal
        crop2 = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        if crop2.ndim == 2:                        # a veces cv2 devuelve HW
            crop2 = crop2[:, :, None]
        return crop2.astype(np.float32)
