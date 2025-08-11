import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, ResizeWithPadOrCropd,
    ScaleIntensityd, EnsureTyped, Resized,Spacingd,
    RandFlipd, RandAffined, RandZoomd, Compose
)

# 🔧 Clase personalizada para reorientar el volumen según view_axis
class ReorientToViewAxisd:
    def __init__(self, keys, view_axis_getter):
        self.keys = keys
        self.view_axis_getter = view_axis_getter

    def __call__(self, data):
        d = dict(data)
        view_axis = self.view_axis_getter(d)

        for key in self.keys:
            img = d[key]
            if view_axis == "sag":
                pass
            elif view_axis == "cor":
                d[key] = np.transpose(img, (0, 2, 1, 3))  # (C, H, D, W)
            elif view_axis == "axi":
                d[key] = np.transpose(img, (0, 3, 1, 2))  # (C, W, D, H)
            else:
                raise ValueError(f"Invalid view_axis: {view_axis}")
        return d


class MRIDataset3D(Dataset):
    def __init__(self, df, is_train=False, use_augmentation=False, spatial_size=(40,120,120),
                 labels=["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.labels = labels
        def view_axis_getter(data_dict):
            return data_dict["view_axis"]
        
        self.view2onehot = {
            "axi": torch.tensor([1, 0, 0], dtype=torch.float),
            "cor": torch.tensor([0, 1, 0], dtype=torch.float),
            "sag": torch.tensor([0, 0, 1], dtype=torch.float),
        }

        base_transforms = [
            LoadImaged(keys=["image"], image_only=False),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            ReorientToViewAxisd(keys=["image"], view_axis_getter=view_axis_getter),
            Spacingd(keys=["image"], pixdim=(5, 1.5, 1.5), mode="bilinear"),
            Resized(keys=["image"],spatial_size=spatial_size, mode="trilinear"),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image"]),
        ]

        if use_augmentation:
            base_transforms += [
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.3),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.3),
                RandAffined(keys=["image"], prob=0.2,
                            rotate_range=(0.1, 0.1, 0.1),
                            scale_range=(0.05, 0.05, 0.05)),
                RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
            ]

        self.transform = Compose(base_transforms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["path"]
        view_axis  = image_path.split("_LF_")[-1].split(".nii")[0]
        sample = {
            "image": image_path,
            "view_axis": view_axis,
        }

        transformed = self.transform(sample)
        image = transformed["image"]
        view_onehot = self.view2onehot[view_axis]

        if self.is_train:
            label = torch.tensor(row[self.labels].values.astype(np.int64))
            return image, label,row["filename"],view_onehot
        else:
            return image,-1, row["filename"],view_onehot