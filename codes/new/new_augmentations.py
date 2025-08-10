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

# ---------- Augmentations específicas para zipper ----------

class RandomZipperStripe(A.ImageOnlyTransform):
    """
    Inyecta un patrón sinusoidal tipo 'zipper' débil en X o Y.
    Trabaja en HxWxC (float32). No cambia la media global fuerte.
    """
    def __init__(self, p=0.15, max_amp=0.20, min_period=6, max_period=22, axis="rand"):
        super().__init__(always_apply=False, p=p)
        self.max_amp = max_amp
        self.min_period = min_period
        self.max_period = max_period
        self.axis = axis  # "x", "y" o "rand"

    def apply(self, img, **params):
        x = img.astype(np.float32)
        h, w, c = x.shape
        amp = np.random.uniform(0.06, self.max_amp)
        period = np.random.randint(self.min_period, self.max_period + 1)
        phase = np.random.uniform(0, 2 * np.pi)

        axis = self.axis
        if axis == "rand":
            axis = "x" if np.random.rand() < 0.5 else "y"

        if axis == "x":
            grid = np.sin(2 * np.pi * (np.arange(w)[None, :] / period) + phase)
            stripe = np.repeat(grid[None, :, :], h, axis=0)  # HxW
        else:  # "y"
            grid = np.sin(2 * np.pi * (np.arange(h)[:, None] / period) + phase)
            stripe = np.repeat(grid[:, :, None], w, axis=1).squeeze(-1)

        stripe = stripe[:, :, None]  # HxWx1
        # normaliza a +/-0.5 para no saturar
        stripe = stripe - stripe.mean()
        stripe = stripe / (np.std(stripe) + 1e-6) * 0.5
        x = x + amp * stripe
        return x

class RandomBandCut(A.ImageOnlyTransform):
    """
    Enmascara una banda horizontal o vertical muy delgada (occlusion bands)
    para robustez a la localización exacta del zipper.
    """
    def __init__(self, p=0.1, max_thick_ratio=0.06):
        super().__init__(always_apply=False, p=p)
        self.max_thick_ratio = max_thick_ratio

    def apply(self, img, **params):
        x = img.astype(np.float32)
        h, w, c = x.shape
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
    """
    Recorta sutilmente los bordes (0–4%) y rescalea, para reducir sesgos de coil/bordes.
    """
    def __init__(self, max_crop=0.04, p=0.5):
        super().__init__(always_apply=False, p=p)
        self.max_crop = max_crop

    def apply(self, img, **params):
        x = img.astype(np.float32)
        h, w, c = x.shape
        cy = int(h * self.max_crop * np.random.rand())
        cx = int(w * self.max_crop * np.random.rand())
        y0, y1 = cy, h - cy
        x0, x1 = cx, w - cx
        if y1 - y0 < 8 or x1 - x0 < 8:
            return x
        crop = x[y0:y1, x0:x1, :]
        # resize back
        return np.array(Image.fromarray((crop.squeeze(-1)).astype(np.float32)).resize((w, h), Image.BILINEAR))[:, :, None]

# ----------------------------------------------------------
