"""
Dataset definitions for the LISA 2025 challenge.

This clean implementation focuses on robust preprocessing and moderation of
augmentations.  Notable changes relative to the original implementation:

* **Z‑score normalisation**: Instead of min‑max scaling each slice separately,
  images are normalised by subtracting their mean and dividing by their standard
  deviation.  If the standard deviation is zero (e.g. uniform background), the
  output is set to zero.  This stabilises intensity distributions and is
  consistent with recommendations from the LISA challenge organisers【42949289359295†L333-L339】.
* **Simplified augmentations**: Only rotation (±15°) and small affine
  transformations are applied during training.  Aggressive contrast changes and
  flips are omitted to avoid modifying the type of artefact.
* **Caching**: Loaded arrays are cached in memory on initialisation to avoid
  repeated disk access.

The dataset returns a 2D image tensor, the label vector, the path and a
one‑hot vector indicating the acquisition plane.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


def get_aug_transforms(image_size: int) -> list[A.BasicTransform]:
    """Return a list of augmentations appropriate for MRI quality assessment.

    The augmentations here are intentionally conservative: only small rotations
    and affine transformations are used.  This prevents distorting the
    artefacts we want the model to learn【42949289359295†L333-L339】.
    """
    return [
        A.Resize(image_size, image_size),
        A.Rotate(limit=15, p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.1), rotate=(-10, 10), p=0.5),
        ToTensorV2(),
    ]


def get_base_transforms(image_size: int) -> list[A.BasicTransform]:
    """Return the base (non‑augmenting) transforms for inference.
    """
    return [
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ]


class MRIDataset2D(Dataset):
    """Dataset for 2D slices of 3D NIfTI volumes.

    Args:
        df: DataFrame with at least ``img_path``, ``npy_path``, ``path`` and label
            columns.
        is_train: If ``True``, labels are returned; otherwise ``-1`` is used.
        use_augmentation: Whether to apply training augmentations.
        is_numpy: If ``True``, ``npy_path``/``pkl`` will be used; otherwise
            ``img_path``.
        labels: List of label names corresponding to the target vector.
        transform: Optional custom transform.
        image_size: Output spatial size (images are resized to ``image_size`` x ``image_size``).
    """

    def __init__(self,
                 df: pd.DataFrame,
                 is_train: bool = True,
                 use_augmentation: bool = False,
                 is_numpy: bool = False,
                 labels: list[str] | None = None,
                 transform: A.Compose | None = None,
                 image_size: int = 256,
                 ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.is_numpy = is_numpy
        self.labels = labels or []
        # Map from suffix to one‑hot view vector
        self.view2onehot = {
            "axi": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float),
            "cor": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float),
            "sag": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float),
        }
        if transform is None:
            if self.use_augmentation:
                self.transform = A.Compose(get_aug_transforms(image_size))
            else:
                self.transform = A.Compose(get_base_transforms(image_size))
        else:
            self.transform = A.Compose(transform)
        # Preload data arrays to avoid repeated IO
        self.data: list[np.ndarray] = [None for _ in range(len(self.df))]
        for idx in tqdm(range(len(self.df)), desc="Loading MRI data"):
            arr = joblib.load(self.df.iloc[idx]['npy_path'].replace(".npy", ".pkl"))
            self.data[idx] = arr

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | int, str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = row["path"]
        # Extract view suffix from filename: e.g. LISA_0001_LF_axi.nii.gz → "axi"
        view_axis = image_path.split("_LF_")[-1].split(".nii")[0]
        view_onehot = self.view2onehot.get(view_axis, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float))

        # Load image array
        if self.is_numpy:
            arr = self.data[idx]
            img = np.expand_dims(arr, axis=-1)  # (H, W, 1)
        else:
            pil_img = Image.open(row['img_path']).convert("L")
            img = np.array(pil_img, dtype=np.float32)
            img = np.expand_dims(img, axis=-1)

        # Z‑score normalisation per image
        mean_val = img.mean()
        std_val = img.std()
        if std_val > 0:
            img = (img - mean_val) / std_val
        else:
            img = np.zeros_like(img, dtype=np.float32)

        # Apply transforms
        if self.transform:
            img = self.transform(image=img)["image"]

        if self.is_train:
            label = torch.tensor(row[self.labels].values.astype(np.int64))
            return img, label, row['img_path'], view_onehot
        else:
            return img, -1, row['npy_path'], view_onehot