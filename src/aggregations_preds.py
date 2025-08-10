import numpy as np
import torch
from collections import defaultdict

def aggregate_predictions_by_img_path(val_probs, val_trues, val_paths, thr2=0.10, thr1=0.30):
    """
    Agrupa y promedia predicciones y etiquetas verdaderas por imagen base (ej. LISA_0001_LF_axi.nii.gz).

    Regla para predicción por volumen (por etiqueta):
      - Si >= thr2 de los slices predicen clase 2 → 2
      - Si no, si >= thr1 predicen clase 1 → 1
      - Si no → 0

    Returns:
        y_pred (np.ndarray): (N, 7) categorías 0/1/2
        y_probs (np.ndarray): (N, 7, 3) promedio simple de probs por volumen (sin cambiar)
        y_true (np.ndarray): (N, 7) etiquetas verdaderas por volumen (redondeo de la media)
        filenames (list[str]): nombres de volumen
    """
    probs_all = torch.cat(val_probs, dim=0).cpu().numpy()  # (N_slices, 7, 3)
    trues_all = torch.cat(val_trues, dim=0).cpu().numpy()  # (N_slices, 7)
    paths_all = ["_".join(p.split("/")[-1].split(".")[0].split("_")[:-1]) + ".nii.gz" for p in val_paths]

    grouped_probs = defaultdict(list)
    grouped_trues = defaultdict(list)

    for path, probs, true in zip(paths_all, probs_all, trues_all):
        grouped_probs[path].append(probs)  # (7,3)
        grouped_trues[path].append(true)   # (7,)

    final_prods, final_preds, final_trues, final_filenames = [], [], [], []
    for key in grouped_probs:
        slices_probs = np.stack(grouped_probs[key], axis=0)   # (S, 7, 3)
        mean_probs   = slices_probs.mean(axis=0)              # (7, 3) ← lo mantenemos igual

        #"""
        # >>>>>>>>>>>> SOLO CAMBIA ESTA PARTE <<<<<<<<<<<<
        slice_preds = np.argmax(slices_probs, axis=2)         # (S, 7)
        frac_2 = (slice_preds == 2).mean(axis=0)              # (7,)
        frac_1 = (slice_preds == 1).mean(axis=0)              # (7,)

        pred = np.where(frac_2 >= thr2, 2,
                        np.where(frac_1 >= thr1, 1, 0))       # (7,)
        #print(pred)
        #print(frac_2,frac_1,pred)
        # >>>>>>>>>>>> FIN CAMBIO <<<<<<<<<<<<<<<<<<<<<<<<
        """
        # >>>>>>>>>>>> SOLO CAMBIA ESTA PARTE <<<<<<<<<<<<
        slice_preds = np.argmax(slices_probs, axis=2)         # (S, 7)
        frac_2 = (slice_preds == 2).mean(axis=0)              # (7,)
        frac_1 = (slice_preds == 1).mean(axis=0)              # (7,)

        pred = np.where(frac_2 >= thr2, 2,
                        np.where(frac_1 >= thr1, 1, 0))       # (7,)

        # media de probs usando TOP-% según clase final
        S, L, _ = slices_probs.shape
        mean_probs = np.zeros((L, 3), dtype=float)

        for j in range(L):
            cls = pred[j]  # 0/1/2
            # tamaño del top-k según la clase elegida
            if cls == 2:
                k = max(1, int(np.ceil(thr2 * S)))
                order = np.argsort(slices_probs[:, j, 2])[-k:]
            elif cls == 1:
                k = max(1, int(np.ceil(thr1 * S)))
                order = np.argsort(slices_probs[:, j, 1])[-k:]
            else:
                # Para clase 0, usa top (1 - max(thr2, thr1)) del score de clase 0
                k0 = 1.0 - max(thr2, thr1)
                k = max(1, int(np.ceil(k0 * S)))
                order = np.argsort(slices_probs[:, j, 0])[-k:]

            mean_probs[j] = slices_probs[order, j, :].mean(axis=0)
        # >>>>>>>>>>>> FIN CAMBIO <<<<<<<<<<<<<<<<<<<<<<<<
        #"""

        mean_true = np.round(np.mean(np.stack(grouped_trues[key], axis=0), axis=0)).astype(int)  # (7,)

        final_preds.append(pred)
        final_prods.append(mean_probs)
        final_trues.append(mean_true)
        final_filenames.append(key)

    return np.stack(final_preds), np.stack(final_prods), np.stack(final_trues), final_filenames


import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

def ensemble_mode_and_mean_probsnew(csv_paths, label_cols, save_preds_path="ensemble_preds.csv",thr2=None,thr1=None):
    """
    - Toma argmax (0/1/2) por etiqueta en cada CSV.
    - Agrupa por filename -> promedia y redondea -> pred final.
    - Promedia probabilidades por filename y las guarda en *_prods.csv.
    """
    assert len(csv_paths) > 0, "Debes proveer al menos un archivo"
    os.makedirs(os.path.dirname(save_preds_path), exist_ok=True)

    dfs = [pd.read_csv(p) for p in csv_paths]
    base_cols = [f"{lab}_{j}" for lab in label_cols for j in range(3)]

    # Validaciones
    for df in dfs:
        assert "filename" in df.columns, "Cada CSV debe tener columna 'filename'"
        assert all(c in df.columns for c in base_cols), "Faltan columnas de probabilidad"

    # --- Concatenar todo ---
    
    df_all = pd.concat(dfs, ignore_index=True)
    #"""
    df_all = dfs[0].copy()
    for col in base_cols:
        ga = 0
        for df in dfs:
            ga+=df[col]
        df_all[col]= ga/len(dfs)
    #"""    
    # --- Media de probabilidades por filename ---
    df_probs = (
        df_all.groupby("filename")[base_cols]
        .mean()
        .reset_index()
    )
    save_probs_path = save_preds_path.replace(".csv", "_prods.csv")
    df_probs.to_csv(save_probs_path, index=False)
    print(f"✅ Probabilidades promedio guardadas en: {save_probs_path}")

    # --- Argmax por fila y media por filename ---
    preds_per_row = []
    for lab in label_cols:
        class_cols = [f"{lab}_{j}" for j in range(3)]
        preds_per_row.append(df_all[class_cols].values.argmax(axis=1))
    preds_per_row = np.stack(preds_per_row, axis=1)  # (N, L)

    df_preds_all = pd.DataFrame(preds_per_row, columns=label_cols)
    df_preds_all["filename"] = df_all["filename"]

    # Agrupar por filename -> media -> redondear
    print(df_preds_all)
    df_preds = (
        df_preds_all.groupby("filename")[label_cols]
        .mean()
        .round(0)
        .astype(int)
        .reset_index()
    )

    # Añadir ID
    df_preds["ID"] = df_preds["filename"].apply(lambda x: x.split("_LF_")[0])
    df_preds[["ID", *label_cols]].to_csv(save_preds_path, index=False)
    print(f"✅ Predicciones finales guardadas en: {save_preds_path}")

    return df_probs, df_preds


import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

def ensemble_mode_and_mean_probsga(csv_paths, label_cols, save_preds_path="ensemble_preds.csv", thr2=0.1, thr1=0.3):
    """
    - Promedia probabilidades por filename.
    - Aplica reglas:
        Si prob clase 2 >= thr2 → 2
        Si prob clase 1 >= thr1 → 1
        Si no → argmax
    - Guarda promedios y predicciones.
    """
    assert len(csv_paths) > 0, "Debes proveer al menos un archivo"
    os.makedirs(os.path.dirname(save_preds_path), exist_ok=True)

    dfs = [pd.read_csv(p) for p in csv_paths]
    base_cols = [f"{lab}_{j}" for lab in label_cols for j in range(3)]

    # Validar que todas las columnas están
    for df in dfs:
        assert "filename" in df.columns, "Cada CSV debe tener columna 'filename'"
        assert all(c in df.columns for c in base_cols), "Faltan columnas de probabilidad"

    # --- Concatenar todo ---
    df_all = dfs[0].copy()
    for col in base_cols:
        ga = 0
        for df in dfs:
            ga+=df[col]
        df_all[col]= ga/len(dfs)

    # --- Media de probabilidades por filename ---
    df_probs = (
        df_all.groupby("filename")[base_cols]
        .mean()
        .reset_index()
    )
    save_probs_path = save_preds_path.replace(".csv", "_prods.csv")
    df_probs.to_csv(save_probs_path, index=False)
    print(f"✅ Probabilidades promedio guardadas en: {save_probs_path}")

    # --- Aplicar reglas de predicción ---
    preds_final = []
    for _, row in df_probs.iterrows():
        pred_row = {}
        for lab in label_cols:
            p0, p1, p2 = row[f"{lab}_0"], row[f"{lab}_1"], row[f"{lab}_2"]
            if p2 >= thr2:
                pred_row[lab] = 2
            elif p1 >= thr1:
                pred_row[lab] = 1
            else:
                pred_row[lab] = np.argmax([p0, p1, p2])
        preds_final.append(pred_row)

    df_preds = pd.DataFrame(preds_final)
    df_preds.insert(0, "filename", df_probs["filename"])
    df_preds["ID"] = df_preds["filename"].apply(lambda x: x.split("_LF_")[0])

    # Guardar
    df_preds[["ID", *label_cols]].to_csv(save_preds_path, index=False)
    print(f"✅ Predicciones finales guardadas en: {save_preds_path}")

    return df_probs, df_preds


def ensemble_mode_and_mean_probs(csv_paths, label_cols, save_preds_path="ensemble_preds.csv",thr2=None,thr1=None):
    """
    Hace ensemble tomando:
      - Moda de predicciones (0/1/2) por etiqueta.
      - Media de probabilidades (si están disponibles).

    Devuelve:
        df_probs (DataFrame): probabilidades promedio
        df_preds (DataFrame): predicciones finales por moda
    """
    assert len(csv_paths) > 0, "Debes proveer al menos un archivo"
    os.makedirs(os.path.dirname(save_preds_path), exist_ok=True)

    dfs = [pd.read_csv(p) for p in csv_paths]

    filenames = dfs[0]["filename"].values
    for df in dfs:
        assert "filename" in df.columns, "Cada CSV debe tener columna 'filename'"
        assert (df["filename"].values == filenames).all(), "Archivos con distintos filenames u orden"

    # Detectores
    def have_probs(df):
        return all(all(f"{lab}_{j}" in df.columns for j in range(3)) for lab in label_cols)
    def have_classes(df):
        return all(lab in df.columns for lab in label_cols)

    probs_list = []
    pred_dfs = []

    for df in dfs:
        if have_probs(df):
            # Guardar probabilidades
            base_cols = [f"{lab}_{j}" for lab in label_cols for j in range(3)]
            probs_list.append(df[base_cols].values)

            # Predicciones por argmax
            pred_df = pd.DataFrame({"filename": df["filename"].values})
            for lab in label_cols:
                class_cols = [f"{lab}_{j}" for j in range(3)]
                pred_df[lab] = df[class_cols].values.argmax(axis=1).astype(int)
            pred_dfs.append(pred_df)



        elif have_classes(df):
            # No hay probs, rellenamos con one-hot
            one_hot_probs = []
            for lab in label_cols:
                one_hot = np.eye(3)[df[lab].astype(int).values]  # (N,3)
                one_hot_probs.append(one_hot)
            probs_list.append(np.concatenate(one_hot_probs, axis=1))
            pred_dfs.append(df[["filename", *label_cols]].copy())
        else:
            raise ValueError("CSV inválido: debe contener probs o clases.")

    # Media de probabilidades
    avg_probs = np.mean(probs_list, axis=0)  # (N, L*3)
    base_cols = [f"{lab}_{j}" for lab in label_cols for j in range(3)]
    df_probs = pd.DataFrame(avg_probs, columns=base_cols)
    df_probs.insert(0, "filename", filenames)
    save_probs_path = save_preds_path.replace(".csv", "_prods.csv")
    df_probs.to_csv(save_probs_path, index=False)
    print(f"✅ Probabilidades promedio guardadas en: {save_probs_path}")

    # Moda de predicciones
    K, N, L = len(pred_dfs), len(filenames), len(label_cols)
    stacked = np.stack([pdf[label_cols].values for pdf in pred_dfs], axis=0)  # (K,N,L)
    preds_mode = []
    for l in range(L):
        col_df = pd.DataFrame(stacked[:, :, l].T)  # (N,K)
        preds_mode.append(col_df.mode(axis=1)[0].astype(int).values)
    preds_mode = np.stack(preds_mode, axis=1)

    df_preds = pd.DataFrame({"filename": filenames})
    for i, lab in enumerate(label_cols):
        df_preds[lab] = preds_mode[:, i]
    df_preds["ID"] = df_preds["filename"].apply(lambda x: x.split("_LF_")[0])
    df_preds[['ID', *label_cols]].to_csv(save_preds_path, index=False)
    print(f"✅ Predicciones por moda guardadas en: {save_preds_path}")

    return df_probs, df_preds

