"""
Datasets for the LISA 2025 challenge.

This module defines a simple `MRIDataset2D` class for loading 2D slices
from 3D volumetric MR images.  It uses Z‑score normalisation on a per‑slice
basis and exposes a one‑hot encoded view (axial, coronal, sagittal) that can
optionally be passed to models.  The dataset supports both numpy arrays
(pickled via joblib) and image files on disk.  Augmentations are kept
moderate to avoid altering the appearance of artefacts; by default only
random rotations, minor affine transforms and mild noise/blur are applied.

Note
----
The provided augmentation pipeline is inspired by recommendations from
publications on MRI quality assessment【42949289359295†L333-L340】.  You can
adjust or extend the transforms as needed.  The dataset does not compute
global dataset statistics; instead each slice is standardised independently
((x−mean)/std) to avoid scale differences between volumes.
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
                 image_size: int = 224) -> None:
        import pandas as pd  # imported here to avoid top‑level dependency if not needed
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.is_numpy = is_numpy
        self.labels = list(labels)
        self.image_size = image_size
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
            # Slight random rotation around the slice plane
            A.Rotate(limit=15, p=0.5),
            # Minor affine transformations (scale and translation)
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), p=0.5),
            # One of mild noise or blur
            A.OneOf([
                A.GaussNoise(var_limit=(1e-5, 0.005), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 1.0), p=1.0),
            ], p=0.3),
        ]
        return A.Compose([A.Resize(image_size, image_size)] + aug_transforms + [ToTensorV2()])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        # Determine the view from the path: everything after '_LF_' until '.nii'
        full_path = row.get('path') or row.get('filename') or row.get('img_path') or row.get('npy_path')
        if full_path is None:
            raise ValueError("Row is missing a valid path or filename column")
        # Extract view code (axi/cor/sag) from something like '..._LF_axi.nii.gz'
        try:
            view_code = full_path.split('_LF_')[-1].split('.')[0]
        except Exception:
            view_code = 'axi'
        view_onehot = self.view2onehot.get(view_code, torch.tensor([1, 0, 0], dtype=torch.float32))
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
        # Normalise slice: Z‑score
        mean_val = float(img.mean())
        std_val = float(img.std())
        if std_val > 1e-6:
            img = (img - mean_val) / std_val
        else:
            img = np.zeros_like(img, dtype=np.float32)
        # Expand to (H, W, 1) for Albumentations
        img = np.expand_dims(img, axis=-1)
        # Apply transforms to convert to tensor
        if self.transform:
            img = self.transform(image=img)['image']  # returns tensor (1, H, W)
        if self.is_train:
            label_values = row[self.labels].values.astype(np.int64)
            labels_tensor = torch.tensor(label_values, dtype=torch.long)
            return img, labels_tensor, full_path, view_onehot
        else:
            return img, -1, full_path, view_onehot
