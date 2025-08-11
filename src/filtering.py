import os, re, numpy as np, pandas as pd
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Regex para extraer el índice del slice desde el path
SLICE_RE = re.compile(r".*_(\d+)\.npy$")

def _slice_idx_from_path(p):
    m = SLICE_RE.match(p)
    return int(m.group(1)) if m else 0

def load_npy_norm(path):
    arr = np.load(path).astype(np.float32)
    a, b = arr.min(), arr.max()
    if b > a:
        arr = (arr - a) / (b - a)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr

def filter_similar_group_ssim(df_group, view_col="view", ssim_thresh=0.98, keep_first_last=True):
    """
    Filtra slices muy parecidos dentro de un grupo (paciente+vista) usando SSIM.
    """
    keep_idx = []
    for view, dfv in df_group.groupby(view_col):
        dfv = dfv.sort_values("npy_path", key=lambda s: s.map(_slice_idx_from_path))
        last_img = None

        for i, row in dfv.iterrows():
            img = load_npy_norm(row["npy_path"])
            if img.ndim == 3:  # Si viene como (D,H,W), tomar el central
                img = img[img.shape[0] // 2]

            if last_img is None:
                keep_idx.append(i)
                last_img = img
                continue

            score = ssim(last_img, img, data_range=1.0)
            if score < ssim_thresh:  # suficientemente distinto
                keep_idx.append(i)
                last_img = img

        if keep_first_last and len(dfv) > 1:
            first_i = dfv.index[0]
            last_i = dfv.index[-1]
            if first_i not in keep_idx:
                keep_idx.insert(0, first_i)
            if last_i not in keep_idx:
                keep_idx.append(last_i)

    return df_group.loc[sorted(set(keep_idx))]

def filter_dataset_by_similarity_ssim(df, ssim_thresh=0.98, group_cols=("patient_id", "view")):
    kept = []
    for _, g in tqdm(df.groupby(list(group_cols)), desc="Filtrando pacientes", total=df[group_cols[0]].nunique()):
        kept.append(filter_similar_group_ssim(g, ssim_thresh=ssim_thresh))
    return pd.concat(kept, axis=0).sort_index().reset_index(drop=True)

# ==== USO ====
# df_filtrado = filter_dataset_by_similarity_ssim(df_train, ssim_thresh=0.985)
# print(f"Original: {df_train.shape} → Filtrado: {df_filtrado.shape}")
