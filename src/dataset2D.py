# ✅ IMPORTS
import os
import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
# 
def extract_group_id(filename):
    match = re.match(r"(LISA_\d+)_", filename)
    return match.group(1) if match else filename

"""
"""
def get_aug_transforms(image_size):
    return [
            A.Resize(image_size, image_size),
            # 🔁 Flip + rotación leve
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),


            A.OneOf([
                # 🔎 Zoom + desplazamiento leve
                #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-15, 15), p=1.0),
                # 🔳 Apagar zonas aleatorias (simula distorsión visual o falta de señal)
                A.CoarseDropout(num_holes_range=(1,2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill=0, p=1.0),
                # 🖼️ Zoom tipo crop + resize (cambia FOV)
                A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), ratio=(0.7, 1.3), p=1.0),
            ], p=0.33),

            A.OneOf([
                A.GaussNoise(std_range=(1e-5, 2e-1), mean_range=(0,1e-4), p=1.0),   # ruido muy leve
                A.GaussianBlur(sigma_limit= (0.5,5), blur_limit=(3,20), p=1.0),                # desenfoque apenas perceptible
                A.MedianBlur(blur_limit=(3,21), p=1.0),
            ], p=0.33),

            A.OneOf([
                A.MultiplicativeNoise(multiplier=(0.95, 1.2), p=1.0),   # cambia el contraste levemente
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1,p=1.0),
                A.CLAHE(clip_limit=2,p=1.0),
            ], p=0.33),

            # 🧮 Normalización y ToTensor
            #A.Normalize(mean=(0.5,), std=(0.5,)),

            ToTensorV2()]

def get_base_transforms(image_size):
    return [
                A.Resize(image_size, image_size),
                #A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2()
            ]


# ✅ DATASET
class MRIDataset2D(Dataset):
    def __init__(self, df, is_train=True,use_augmentation = False,is_numpy=False,labels=[],transform = None,image_size=256):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.is_numpy = is_numpy
        self.labels   = labels
        self.view2onehot = {
            "axi": torch.tensor([1, 0, 0], dtype=torch.float),
            "cor": torch.tensor([0, 1, 0], dtype=torch.float),
            "sag": torch.tensor([0, 0, 1], dtype=torch.float),
        }

        if transform is None:
            if self.use_augmentation:
                self.transform = A.Compose(get_aug_transforms(image_size))
            else:
                self.transform = A.Compose(get_base_transforms(image_size))
        else:
            self.transform = A.Compose(transform)

        self.data = [None for i in range(df.shape[0])]
        for idx in tqdm(range(df.shape[0])):

            arr = joblib.load(self.df.iloc[idx]['npy_path'].replace(".npy", ".pkl"))
            self.data[idx] = arr
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["path"]
        view_axis  = image_path.split("_LF_")[-1].split(".nii")[0]
        view_onehot = self.view2onehot[view_axis]

        if self.is_numpy:
            #arr = np.load(row['npy_path'])           # (H, W) o (D, H, W)
            arr = self.data[idx]
            img = np.expand_dims(arr, axis=-1)  # (H, W) → (H, W, 1)
            #img = (img-img.min())/(img.max()-img.min())

            """
            min_val = img.min()
            max_val = img.max()

            if max_val > min_val:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = np.zeros_like(img)  # o img.fill(0.0)
            """
            # Normalise slice: Z‑score
            mean_val = float(img.mean())
            std_val = float(img.std())
            #print(std_val)
            if std_val > 1e-6:
                img = (img - mean_val) / std_val
            else:
                img = np.zeros_like(img, dtype=np.float32)

            img = img.astype(np.float32)        # obligatorio para Albumentations

        else:
            img = Image.open(row['img_path']).convert("L")  # "L" es modo de 8 bits en escala de grises
            img = np.array(img, dtype=np.float32)  # ahora img tiene shape (H, W)
            mean_val = float(img.mean())
            std_val = float(img.std())
            if std_val > 1e-6:
                img = (img - mean_val) / std_val
            else:
                img = np.zeros_like(img, dtype=np.float32)
            #img = (img-img.min())/(img.max()-img.min())
            
            img = np.expand_dims(img, axis=-1)     # (H, W, 1) para Albumentations
            
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.is_train:
            label = torch.tensor(row[self.labels].values.astype(np.int64))  # shape (7,)
            return img, label, row['img_path'],view_onehot
        else:
            return img, -1, row['npy_path'],view_onehot
        

