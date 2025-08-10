
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
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
    #df_train_original = assign_robust_folds(df_train_original, n_splits=args.n_splits, top_k=2, seed=42)
    df_train_original = assign_patient_stratified_folds(df_train_original, n_splits=args.n_splits, top_k=2, seed=42)

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

    # Filtrar duplicados y quedarte con la primera aparición
    df_train_unique = df_train_original.drop_duplicates(subset=["patient_id", "fold"]).reset_index(drop=True)

    print("df_train antes del merge : ",df_train.shape)
    print("df_train_original antes del merge : ",df_train_original.shape)
    print("df_train_unique   antes del merge : ",df_train_unique.shape)
    #print(df_train_original[["patient_id","fold"]][df_train_original["patient_id"].isin(train_ids[:2])])
    df_train = df_train.merge(df_train_unique[["patient_id","fold"]],on="patient_id",how="left")
    print("df_train despues del merge : ",df_train.shape)
    print(list(df_train.columns))
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

import torch
import numpy as np

def _safe_counts(series, num_classes=3):
    """Devuelve conteos por clase [0..C-1] como np.array, con eps para evitar ceros."""
    eps = 1e-6
    counts = np.zeros(num_classes, dtype=np.float64)
    vc = series.value_counts().to_dict()
    for c in range(num_classes):
        counts[c] = max(vc.get(c, 0), eps)
    return counts

def _weights_effective(n_counts, beta=0.99):
    """Class-Balanced weights (Cui et al.) con normalización media=1."""
    n = np.asarray(n_counts, dtype=np.float64)
    eff_num = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = 1.0 / eff_num
    w /= w.mean()
    return w

def _weights_invfreq(n_counts, alpha=0.5, cap=8.0):
    """Inversa de frecuencia^alpha con normalización media=1 y límite superior 'cap'."""
    n = np.asarray(n_counts, dtype=np.float64)
    N, K = n.sum(), len(n)
    w = (N / (K * n)) ** alpha
    w /= w.mean()
    w = np.clip(w, 1.0 / cap, cap)
    return w

def compute_weights_from_df(
    df,
    labels,
    method: str = "effective",   # "effective" o "invfreq"
    beta: float = 0.99,          # para "effective"
    alpha: float = 0.5,          # para "invfreq"
    cap: float = 8.0,            # para "invfreq"
    device=None,
    dtype=torch.float32,
):
    """
    Devuelve {label: tensor([w0,w1,w2])}, normalizados (media≈1).
    Pensado para usarse con Focal (γ≈1.5–2.0) + label_smoothing bajo (≈0.05).
    """
    if method == "manual":
        weights_per_label = {
            "Noise":      torch.tensor([1.0, 2.0, 5.0]),
            "Zipper":     torch.tensor([1.0, 2.0, 5.0]),
            "Positioning":torch.tensor([1.0, 2.0, 5.0]),
            "Banding":    torch.tensor([1.0, 2.0, 5.0]),
            "Motion":     torch.tensor([1.0, 2.0, 5.0]),
            "Contrast":   torch.tensor([1.0, 2.0, 5.0]),
            "Distortion": torch.tensor([1.0, 2.0, 5.0]),
        }
        return weights_per_label
    
    weights_per_label = {}
    for label in labels:
        n_counts = _safe_counts(df[label])

        if method == "effective":
            w = _weights_effective(n_counts, beta=beta)
        elif method == "invfreq":
            w = _weights_invfreq(n_counts, alpha=alpha, cap=cap)
        else:
            raise ValueError(f"method debe ser 'effective' o 'invfreq', no '{method}'.")

        t = torch.tensor(w, dtype=dtype)
        if device is not None:
            t = t.to(device)
        weights_per_label[label] = t
    return weights_per_label


def compute_weights_from_df_old(df, labels=LABELS, use_manual=False):
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



def assign_patient_stratified_folds(df, n_splits=5, top_k=2, seed=42):
    df = df.copy()

    # 1. Elegir etiquetas más raras con severidad 2 (como en tu código)
    severity_counts = label_severity_counts(df)
    sorted_labels = sorted(severity_counts, key=severity_counts.get)
    selected_labels = sorted_labels[:top_k]
    print(f"🎯 Labels usados para estratificación: {selected_labels}")

    # 2. Crear clave combinada de estratificación (usando esas etiquetas)
    df["stratify_key"] = df[selected_labels].astype(str).agg("|".join, axis=1)

    # 3. Usar StratifiedGroupKFold para estratificar por paciente
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, y=df["stratify_key"], groups=df["patient_id"])
    ):
        df.loc[val_idx, "fold"] = fold

    # 4. Limpiar columnas auxiliares
    df = df.drop(columns=["stratify_key"])

    return df

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


