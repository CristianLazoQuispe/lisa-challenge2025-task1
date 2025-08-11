"""
Datasets for the LISA 2025 challenge.
"""

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
import sys
import gc
import os
import cv2
import sys
import gc
import os

# Ruta absoluta del script actual
current_path = os.path.dirname(os.path.abspath(__file__))

# Agregar al sys.path si no está
if current_path not in sys.path:
    sys.path.append(current_path)

import new_augmentations


class MRIDataset2D(Dataset):
    """Dataset that loads individual 2D slices extracted from 3D NIfTI volumes.

    The input ``DataFrame`` should contain at least the columns:

    - ``'img_path'`` or ``'npy_path'``: path to the slice image or a NumPy
      array pickled via ``joblib``.  Exactly one of these columns must be
      present.
    - ``'path'`` or ``'filename'``: full path or filename of the original
      NIfTI volume (used to derive the volume name when aggregating).
    - Label columns as specified by the ``labels`` argument.
    - ``'patient_id'``: identifier for grouping slices belonging to the same
      patient; not strictly needed by the dataset but useful for splitting.

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe containing slice information and labels.
    is_train: bool, default=True
        Whether the dataset is used for training.  Determines whether to
        return labels.
    use_augmentation: bool, default=False
        Whether to apply data augmentations.  If ``False``, only base
        transformations (resize and normalise) are applied.
    is_numpy: bool, default=True
        If ``True``, the dataset expects a column ``'npy_path'`` with paths
        to pickled numpy arrays (pkl files).  If ``False``, it expects
        ``'img_path'`` pointing to image files (PNG, JPEG, etc.).
    labels: Iterable[str]
        Names of the label columns.  Used to extract targets from the
        dataframe.  The order of labels defines the order of outputs.
    image_size: int, default=224
        Side length of the square to which each slice is resized.

    Notes
    -----
    The dataset normalises each slice by subtracting its mean and dividing
    by its standard deviation plus a small epsilon.  If the standard
    deviation is extremely small (e.g., a completely uniform slice), the
    normalised image is set to zeros.
    """

    def __init__(self,
                 df,
                 is_train: bool = True,
                 use_augmentation: bool = False,
                 is_numpy: bool = True,
                 labels: Iterable[str] = (),
                 image_size: int = 224,
                 use_norma : bool = True,
                 norm_mode: str = "slice_z",
                 per_view_stats: Optional[dict] = None,
                 dataset_mean: Optional[float] = None,
                 dataset_std: Optional[float] = None) -> None:
        import pandas as pd  # imported here to avoid top‑level dependency if not needed
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.is_numpy = is_numpy
        self.labels = list(labels)
        self.image_size = image_size
        # Legacy flag: apply z-score per slice if norm_mode remains "slice_z".
        self.use_norma = use_norma
        # Normalisation mode: "slice_z" (per-slice z-score), "dataset_z_per_view" (fixed stats per view) or "none".
        self.norm_mode = norm_mode
        # Optional stats for dataset_z_per_view mode
        self.per_view_stats = per_view_stats or {}
        # Optional global mean/std (unused currently, reserved for future use)
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        # Map view codes to one‑hot vectors (axial, coronal, sagittal)
        self.view2onehot = {
            "axi": torch.tensor([1, 0, 0], dtype=torch.float32),
            "cor": torch.tensor([0, 1, 0], dtype=torch.float32),
            "sag": torch.tensor([0, 0, 1], dtype=torch.float32),
        }
        # Compose transforms
        self.transform = self._build_transform(image_size, use_augmentation)
        # Preload data if using numpy for efficiency
        self.data: List[np.ndarray | None] = [None] * len(self.df)
        if self.is_numpy:
            for idx in tqdm(range(len(self.df)), desc="Loading arrays", disable=len(self.df) < 100):
                path = self.df.iloc[idx].get('npy_path') or self.df.iloc[idx].get('img_path')
                # Replace .npy with .pkl if present (for backward compatibility)
                if path and path.endswith('.npy'):
                    path = path.replace('.npy', '.pkl')
                arr = joblib.load(path)
                self.data[idx] = arr

    """
                new_augmentations.CenterDeBorder(max_crop=0.1, p=0.2), #, min_keep=8
            # Slight random rotation around the slice plane
            A.HorizontalFlip(p=0.15),
            A.VerticalFlip(p=0.10),
            A.Rotate(limit=10, p=0.5),

            # Minor affine transformations (scale and translation)
            A.OneOf([
                # 🔎 Zoom + desplazamiento leve
                #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.1), rotate=(-15, 15),
                         shear={'x': (-12, 12), 'y': (-6, 6)}, p=1.0),
                # 🔳 Apagar zonas aleatorias (simula distorsión visual o falta de señal)
                A.CoarseDropout(num_holes_range=(1,2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, p=1.0),
                # 🖼️ Zoom tipo crop + resize (cambia FOV)
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.8, 1.2), p=1),
                #A.ShiftScaleRotate(shift_limit=0.28, scale_limit=0.15, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            ], p=0.3),                        
            # One of mild noise or blur
            A.OneOf([
                A.GaussNoise(std_range=(1e-5, 2e-1), mean_range=(0,1e-4), p=1.0),   # ruido muy leve
                A.GaussianBlur(sigma_limit= (0.5,5), blur_limit=(3,20), p=1.0),                # desenfoque apenas perceptible
                A.MedianBlur(blur_limit=(3,21), p=1.0),
            ], p=0.3),

            #
            A.OneOf([
                A.MultiplicativeNoise(multiplier=(0.95, 1.2), p=1.0),   # cambia el contraste levemente
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=1.0),
                A.CLAHE(clip_limit=2,p=1.0),
            ], p=0.3),
            A.OneOf([
                new_augmentations.RandomZipperStripe(p=1.0, max_amp=0.18, min_period=6, max_period=22, axis="rand"),
                new_augmentations.RandomBandCut(p=1.0),
                A.NoOp(p=1.0),  # placeholder si no lo importas aquí
            ], p=0.10)
    """
    @staticmethod
    def _build_transform(image_size: int, use_augmentation: bool) -> A.Compose:
        """Construct an Albumentations transformation pipeline.

        The augmentations are deliberately mild: small rotations and
        affine transforms plus optional Gaussian noise or blur.  Each
        transformation returns a PyTorch tensor.
        """
        base_transforms = [A.Resize(image_size, image_size), ToTensorV2()]
        if not use_augmentation:
            return A.Compose(base_transforms)
        aug_transforms = [
            new_augmentations.CenterDeBorder(p=0.5),
            # Slight random rotation around the slice plane
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),

            # Minor affine transformations (scale and translation)
            A.OneOf([
                # 🔎 Zoom + desplazamiento leve
                #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-15, 15), p=1.0),
                # 🔳 Apagar zonas aleatorias (simula distorsión visual o falta de señal)
                A.CoarseDropout(num_holes_range=(1,2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, p=1.0),
                # 🖼️ Zoom tipo crop + resize (cambia FOV)
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.8, 1.2), p=1),
            ], p=0.3),                        
            # One of mild noise or blur
            A.OneOf([
                A.GaussNoise(std_range=(1e-5, 2e-1), mean_range=(0,1e-4), p=1.0),   # ruido muy leve
                A.GaussianBlur(sigma_limit= (0.5,5), blur_limit=(3,20), p=1.0),                # desenfoque apenas perceptible
                A.MedianBlur(blur_limit=(3,21), p=1.0),
            ], p=0.3),

            A.OneOf([
                A.MultiplicativeNoise(multiplier=(0.95, 1.2), p=1.0),   # cambia el contraste levemente
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=1.0),
                A.CLAHE(clip_limit=2,p=1.0),
            ], p=0.3),

            new_augmentations.RandomZipperStripe(p=0.20, max_amp=0.18, min_period=6, max_period=22, axis="rand"),
            new_augmentations.RandomBandCut(p=0.12),

        ]
        return A.Compose([A.Resize(image_size, image_size)] + aug_transforms + [ToTensorV2()])

    def __len__(self) -> int:
        return len(self.df)

    def _apply_normalization(self, img: np.ndarray, view_code: str) -> np.ndarray:
        """
        Apply normalisation to the slice according to the selected mode.
        Supported modes:

        - ``none``: returns the image as float32 without changes.
        - ``slice_z``: subtract mean and divide by std of this slice.
        - ``dataset_z_per_view``: clip to [p1, p99] and z-score using fixed mean/std
          computed on the training set for each view code (e.g. "axi", "cor", "sag").

        Parameters
        ----------
        img: np.ndarray
            The 2D slice as a numpy array (float32).
        view_code: str
            The view code extracted from the filename.

        Returns
        -------
        np.ndarray
            Normalised slice as float32.
        """
        x = img.astype(np.float32)
        mode = self.norm_mode or "none"
        #print("MODE : ",mode)
        #print("self.use_norma : ",self.use_norma)
        #print("MODE : ",mode)
        # No normalisation
        if mode == "none":
            return x
        # Per-slice z-score
        if mode == "slice_z":
            mu = float(x.mean())
            sigma = float(x.std())
            if sigma > 1e-6:
                return (x - mu) / sigma
            # constant slice
            return np.zeros_like(x, dtype=np.float32)
        # Fixed stats per view
        if mode == "dataset_z_per_view":
            #print("per_view_stats:",self.per_view_stats)
            #print("view_code:",view_code)
            stats = self.per_view_stats.get(view_code)
            #print("stats:",stats)
            if stats is not None:
                p1 = stats.get("p1")
                p99 = stats.get("p99")
                mu = stats.get("mean")
                sigma = stats.get("std")
                # Clip if percentiles provided
                #print("Using dataset_z_per_view")
                if p1 is not None and p99 is not None:
                    x = np.clip(x, p1, p99)
                if sigma is not None and sigma > 0:
                    return (x - mu) / sigma
                return x - mu
            # Fallback: per-slice z-score if stats missing
            mu = float(x.mean())
            sigma = float(x.std())
            if sigma > 1e-6:
                return (x - mu) / sigma
            return np.zeros_like(x, dtype=np.float32)
        # Unknown mode -> fallback to per-slice z-score
        mu = float(x.mean())
        sigma = float(x.std())
        if sigma > 1e-6:
            return (x - mu) / sigma
        return np.zeros_like(x, dtype=np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Determine the view from the path: everything after '_LF_' until '.nii'
        full_path = row.get('npy_path')# or row.get('filename') or row.get('img_path') or row.get('npy_path')
        if full_path is None:
            raise ValueError("Row is missing a valid path or filename column")
        # Extract view code (axi/cor/sag) from something like '..._LF_axi.nii.gz'
        try:
            if "_lf_" in full_path:
                view_code = full_path.split('_lf_')[-1].split('_')[0]
            else:
                view_code = full_path.split('_LF_')[-1].split('_')[0]

        except Exception:
            view_code = 'axi'
        view_onehot = self.view2onehot.get(view_code, torch.tensor([1, 0, 0], dtype=torch.float32))
        #print("view_onehot:",view_onehot)
        # Load image as numpy array
        if self.is_numpy:
            arr = self.data[idx]
            if arr is None:
                path = row.get('npy_path') or row.get('img_path')
                if path.endswith('.npy'):
                    path = path.replace('.npy', '.pkl')
                arr = joblib.load(path)
                self.data[idx] = arr
            img = arr.astype(np.float32)
        else:
            # Open image file in grayscale
            img = Image.open(row['img_path']).convert('L')
            img = np.array(img, dtype=np.float32)
        # Apply selected normalisation
        if self.use_norma:
            img = self._apply_normalization(img, view_code)
        # Expand to (H, W, 1) for Albumentations
        img = np.expand_dims(img, axis=-1)
        # Apply transforms to convert to tensor
        if self.transform:
            img = self.transform(image=img)['image']  # returns tensor (1, H, W)

        img_np = img.numpy().squeeze()  # HxW float32, ya augmentado y recortado
        #x_,y_,r_ = new_augmentations._brain_centroid_radius(img_np)
        top, bottom, left, right = new_augmentations._brain_margins_connected(img_np)
        

        aux_tensor = torch.tensor([top, bottom, left, right], dtype=torch.float32)
        if self.is_train:
            label_values = row[self.labels].values.astype(np.int64)
            labels_tensor = torch.tensor(label_values, dtype=torch.long)
            return img, labels_tensor, full_path, view_onehot,aux_tensor
        else:
            return img, -1, full_path, view_onehot,aux_tensor
