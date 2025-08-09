"""
Utilidades de soporte para el desafío LISA 2025.

Este módulo contiene funciones para dividir los datos de forma robusta por
paciente, estratificando según las etiquetas raras de severidad 2, así
como funciones para calcular pesos de clase y fijar la semilla global.  Estas
rutinas están inspiradas en el código original proporcionado por el usuario
pero dependen únicamente de bibliotecas estándar.

Variables de entorno como ``WANDB_API_KEY``, ``PROJECT_WANDB`` y ``ENTITY``
deben definirse en un fichero ``.env`` o en el entorno antes de ejecutar
este código para la integración con Weights & Biases.
"""

from __future__ import annotations

import random
from typing import Iterable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold


LABELS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]


def set_seed(seed: int = 42) -> None:
    """Establece la semilla global para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_severity_counts(df: pd.DataFrame, labels: Iterable[str] = LABELS, severity_class: int = 2) -> Dict[str, int]:
    """Cuenta cuántas veces aparece la clase ``severity_class`` para cada etiqueta."""
    return {label: int((df[label] == severity_class).sum()) for label in labels}


def assign_patient_stratified_folds(df: pd.DataFrame, n_splits: int = 5, top_k: int = 2, seed: int = 42) -> pd.DataFrame:
    """Asigna folds estratificados por paciente.

    Selecciona las ``top_k`` etiquetas con menor número de clase 2 para
    construir una clave de estratificación y utiliza ``StratifiedGroupKFold``
    para dividir por paciente manteniendo la distribución de esas etiquetas.

    Args:
        df: DataFrame con columnas ``patient_id`` y las etiquetas.
        n_splits: Número de folds.
        top_k: Número de etiquetas raras a usar para la estratificación.
        seed: Semilla aleatoria.

    Returns:
        DataFrame con una columna ``fold`` que indica el número de fold para
        cada fila.
    """
    df = df.copy()
    # Etiquetas más raras según severidad 2
    severity_counts = label_severity_counts(df)
    sorted_labels = sorted(severity_counts, key=severity_counts.get)
    selected_labels = sorted_labels[:top_k]
    # Clave de estratificación combinando las etiquetas seleccionadas
    df["stratify_key"] = df[selected_labels].astype(str).agg("|".join, axis=1)
    # KFold estratificado por paciente
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(
        sgkf.split(df, y=df["stratify_key"], groups=df["patient_id"])
    ):
        df.loc[val_idx, "fold"] = fold
    df = df.drop(columns=["stratify_key"])
    return df


def assign_volume_stratified_folds(df: pd.DataFrame,
                                   volume_id_col: str = 'patient_id',
                                   label_cols: Iterable[str] = LABELS,
                                   n_splits: int = 5,
                                   top_k: int = 2,
                                   seed: int = 42) -> pd.DataFrame:
    """Asigna folds estratificados por volumen en un DataFrame de cortes 2D.

    Este método está pensado para el caso en que cada volumen 3D está
    representado por múltiples cortes 2D.  Se asegura de que todos los
    cortes de un mismo volumen caigan en el mismo ``fold``, y utiliza una
    estrategia de estratificación basada en la distribución de etiquetas
    a nivel de volumen (tomando la severidad máxima de cada etiqueta a lo
    largo de sus cortes).

    Args:
        df: DataFrame con columnas de etiquetas (``label_cols``) y la
            columna ``volume_id_col`` que identifica a cada volumen (por
            ejemplo, ``patient_id`` o ``filename`` sin sufijo de corte).
        volume_id_col: Nombre de la columna que agrupa los cortes por
            volumen.  Todos los registros con el mismo valor en esta
            columna se asignarán al mismo fold.
        label_cols: Lista de nombres de las columnas de etiquetas.
        n_splits: Número de folds de cross‑validation.
        top_k: Número de etiquetas raras (según severidad 2) a usar para
            la clave de estratificación.
        seed: Semilla aleatoria para reproducibilidad.

    Returns:
        El DataFrame original con una nueva columna ``fold`` asignando
        cada corte a un fold.  Todos los cortes de un volumen comparten
        el mismo número de fold.

    Nota:
        La función calcula primero la severidad máxima para cada etiqueta
        en todos los cortes de cada volumen, creando así un vector de
        etiquetas agregadas.  Luego cuenta cuántos volúmenes tienen
        severidad 2 en cada etiqueta para identificar las etiquetas
        minoritarias.  Se construye una clave de estratificación a partir
        de estas etiquetas y se aplica ``StratifiedGroupKFold`` con el
        identificador de volumen como ``groups``.  Finalmente, la
        asignación de folds se propaga a todas las filas de ``df``.
    """
    if volume_id_col not in df.columns:
        raise ValueError(f"El DataFrame no contiene la columna de volumen '{volume_id_col}'")
    # Agrupar por volumen y calcular la severidad máxima para cada etiqueta
    # Esto condensa las múltiples filas por volumen en una representación
    # única que refleja la peor (máxima) severidad observada en cualquier
    # corte para cada etiqueta.
    agg_labels = (
        df.groupby(volume_id_col)[list(label_cols)]
          .max()
          .reset_index()
    )
    # Contar severidad 2 por etiqueta a nivel de volumen para identificar
    # las etiquetas más raras
    severity_counts = label_severity_counts(agg_labels[label_cols])
    # Ordenar etiquetas por recuento ascendente y elegir top_k
    sorted_labels = sorted(severity_counts, key=severity_counts.get)
    selected_labels = sorted_labels[:top_k]
    # Construir clave de estratificación combinando las etiquetas
    agg_labels['stratify_key'] = agg_labels[selected_labels].astype(str).agg('|'.join, axis=1)
    # Inicializar KFold estratificado por volumen
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Crear nueva columna fold en el DataFrame agregado
    agg_labels['fold'] = -1
    for fold, (_, val_idx) in enumerate(
        sgkf.split(agg_labels, y=agg_labels['stratify_key'], groups=agg_labels[volume_id_col])
    ):
        agg_labels.loc[val_idx, 'fold'] = fold
    # Propagar la asignación de folds al DataFrame original
    df = df.copy()
    df = df.merge(agg_labels[[volume_id_col, 'fold']], on=volume_id_col, how='left')
    return df


def compute_sample_weights(df: pd.DataFrame, label_cols: Iterable[str]) -> list[float]:
    """Computa pesos por muestra basados en la frecuencia de combinaciones de etiquetas.

    Cada muestra recibe un peso igual al inverso de la frecuencia de su
    combinación de etiquetas.  Esto tiende a equilibrar el número de
    muestras por combinación (puede ajustarse según sea necesario).
    """
    counts = df[label_cols].apply(lambda x: tuple(x), axis=1)
    freq = counts.value_counts().to_dict()
    return [1.0 / freq[tuple(row[label_cols])] for _, row in df.iterrows()]


def compute_weights_from_df(
    df: pd.DataFrame,
    labels: Iterable[str],
    method: str = "effective",
    beta: float = 0.99,
    alpha: float = 0.5,
    cap: float = 8.0,
    device=None,
    dtype=torch.float32,
) -> Dict[str, torch.Tensor]:
    """Calcula pesos de clase por etiqueta.

    Los pesos se normalizan para tener media ≈ 1.  Esto puede usarse con
    CrossEntropy o Focal Loss.  ``method`` puede ser ``"effective"``
    (Class‑Balanced) o ``"invfreq"`` (inversa de frecuencia^alpha).
    """
    def _safe_counts(series, num_classes=3):
        eps = 1e-6
        counts = np.zeros(num_classes, dtype=np.float64)
        vc = series.value_counts().to_dict()
        for c in range(num_classes):
            counts[c] = max(vc.get(c, 0), eps)
        return counts
    def _weights_effective(n_counts, beta=beta):
        n = np.asarray(n_counts, dtype=np.float64)
        eff_num = (1.0 - np.power(beta, n)) / (1.0 - beta)
        w = 1.0 / eff_num
        w /= w.mean()
        return w
    def _weights_invfreq(n_counts, alpha=alpha, cap=cap):
        n = np.asarray(n_counts, dtype=np.float64)
        N, K = n.sum(), len(n)
        w = (N / (K * n)) ** alpha
        w /= w.mean()
        w = np.clip(w, 1.0 / cap, cap)
        return w
    weights_per_label: Dict[str, torch.Tensor] = {}
    for label in labels:
        n_counts = _safe_counts(df[label])
        if method == "effective":
            w = _weights_effective(n_counts)
        elif method == "invfreq":
            w = _weights_invfreq(n_counts)
        else:
            raise ValueError(f"Método desconocido: {method}")
        t = torch.tensor(w, dtype=dtype)
        if device is not None:
            t = t.to(device)
        weights_per_label[label] = t
    return weights_per_label