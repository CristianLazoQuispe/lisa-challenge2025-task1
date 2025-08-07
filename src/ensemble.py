import pandas as pd
import numpy as np
import os

def ensemble_probs_from_files(csv_paths, label_cols, save_preds_path="ensemble_preds.csv"):
    os.makedirs(os.path.dirname(save_preds_path),exist_ok=True)
    save_probs_path  = save_preds_path.replace(".csv","_prods.csv")
    assert len(csv_paths) > 0, "Debes proveer al menos un archivo"

    dfs = [pd.read_csv(p) for p in csv_paths]
    base_cols = [f"{label}_{i}" for label in label_cols for i in range(3)]

    # Validación
    filenames = dfs[0]["filename"].values
    for df in dfs:
        assert all(col in df.columns for col in base_cols), "Faltan columnas de probabilidad"
        assert (df["filename"].values == filenames).all(), "Los archivos no tienen los mismos filenames en orden"

    # Promedio de probabilidades
    probs_list = [df[base_cols].values for df in dfs]
    avg_probs = np.mean(probs_list, axis=0)

    # DataFrame de probabilidades
    df_probs = pd.DataFrame(avg_probs, columns=base_cols)
    df_probs.insert(0, "filename", filenames)
    df_probs.to_csv(save_probs_path, index=False)
    df_probs["ID"] = df_probs["filename"].apply(lambda x: x.split("_LF_")[0])  # e.g., LISA_0001
    print(f"✅ Probabilidades guardadas en: {save_probs_path}")

    # DataFrame de predicciones finales (argmax)
    df_preds = pd.DataFrame({"filename": filenames})
    df_preds["ID"] = df_preds["filename"].apply(lambda x: x.split("_LF_")[0])  # e.g., LISA_0001
    for i, label in enumerate(label_cols):
        class_cols = [f"{label}_{j}" for j in range(3)]
        df_preds[label] = df_probs[class_cols].values.argmax(axis=1)

    del df_preds["filename"]
    df_preds.to_csv(save_preds_path, index=False)
    print(f"✅ Predicciones finales guardadas en: {save_preds_path}")

    return df_probs, df_preds


