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
