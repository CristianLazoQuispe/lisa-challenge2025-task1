
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import random
import torch

LABELS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]


def robust_split_by_patient(df_train_original,df_train, args):
    # Split using original data
    df_all = df_train_original.copy()
    df_all["patient_id"] = df_all["filename"].str.extract(r"(LISA_\d+)")
    train_ids, test_ids = train_test_split(df_all["patient_id"].unique(), test_size=0.1, random_state=args.seed, shuffle=True)
    df_train_original  = df_all[df_all["patient_id"].isin(train_ids)].reset_index(drop=True)
    df_test_back       = df_all[df_all["patient_id"].isin(test_ids)].reset_index(drop=True)
    # 🧠 Asignar folds robustos
    df_train_original = assign_robust_folds(df_train_original, n_splits=args.n_splits, top_k=2, seed=42)

    for i, label in enumerate(args.label_cols):
        y = df_train_original[label].values
        print(f"Train    : {label} true dist: {np.bincount(y, minlength=3)}")
    for i, label in enumerate(args.label_cols):
        y = df_test_back[label].values
        print(f"Testback : {label} true dist: {np.bincount(y, minlength=3)}")
    # Assign folds to processed data
    df_all       = df_train.copy()
    df_train     = df_all[df_all["patient_id"].isin(train_ids)].reset_index(drop=True)
    df_test_back = df_all[df_all["patient_id"].isin(test_ids)].reset_index(drop=True)
    df_train = df_train.merge(df_train_original[["patient_id","fold"]],on="patient_id",how="left")

    return df_train,df_test_back

# Semilla global
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_sample_weights(df, label_cols):
    class_weights = {0: 1.0, 1: 10.0, 2: 15.0}
    weights = []
    for _, row in df.iterrows():
        w = sum([class_weights[int(row[col])] for col in label_cols])
        weights.append(w)
    return weights

def compute_weights_from_df(df, labels=LABELS, use_manual=False):
    if use_manual:
        weights_per_label = {
            "Noise":      torch.tensor([1.0, 10.0, 20.0]),
            "Zipper":     torch.tensor([1.0, 10.0, 20.0]),
            "Positioning":torch.tensor([1.0, 10.0, 20.0]),
            "Banding":    torch.tensor([1.0, 10.0, 20.0]),
            "Motion":     torch.tensor([1.0, 10.0, 20.0]),
            "Contrast":   torch.tensor([1.0, 10.0, 20.0]),
            "Distortion": torch.tensor([1.0, 10.0, 20.0]),
        }
        return weights_per_label

    weights = {}
    for label in labels:
        counts = df[label].value_counts(normalize=True).to_dict()
        w = torch.tensor([
            1.0 / counts.get(0, 1e-6),
            1.0 / counts.get(1, 1e-6),
            1.0 / counts.get(2, 1e-6)
        ])
        weights[label] = w / w.sum()
    return weights

def label_severity_counts(df, severity_class=2):
    """Cuenta cuántas veces aparece la clase '2' por etiqueta."""
    return {
        col: (df[col] == severity_class).sum()
        for col in df.columns if col not in ["ID", "group", "filename", "fold"]
    }


def label_severity_counts(df):
    return {
        label: (df[label] == 2).sum()
        for label in LABELS
    }

def assign_robust_folds(df, n_splits=5, top_k=2, seed=42):
    df = df.copy()

    # 1. Obtener las etiquetas más raras con clase 2
    severity_counts = label_severity_counts(df)
    sorted_labels = sorted(severity_counts, key=severity_counts.get)
    selected_labels = sorted_labels[:top_k]
    print(f"🎯 Labels usados para estratificación: {selected_labels}")

    # 2. Crear clave combinada para estratificación
    df["stratify_key"] = df[selected_labels].astype(str).agg("|".join, axis=1)

    # 3. Separar casos raros (combinaciones únicas)
    key_counts = df["stratify_key"].value_counts()
    rare_keys = key_counts[key_counts < 2].index.tolist()
    df_rare = df[df["stratify_key"].isin(rare_keys)].copy()
    df_common = df[~df["stratify_key"].isin(rare_keys)].copy()
    print(f"🟡 Casos raros: {len(df_rare)} — Casos comunes: {len(df_common)}")

    # 4. KFold estratificado sobre los comunes
    df_common["fold"] = -1
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(skf.split(df_common, df_common["stratify_key"])):
        df_common.loc[df_common.index[val_idx], "fold"] = fold

    # 5. Asignar fold aleatorio a los raros
    df_rare["fold"] = -1
    for i, idx in enumerate(df_rare.index):
        df_rare.loc[idx, "fold"] = i % n_splits

    # 6. Unir y devolver
    df_out = pd.concat([df_common, df_rare], axis=0).drop(columns=["stratify_key"])
    df_out = df_out.sort_index().reset_index(drop=True)
    return df_out


