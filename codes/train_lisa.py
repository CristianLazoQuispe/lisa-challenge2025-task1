"""
Script para entrenar y evaluar modelos para el LISA Challenge 2025 con
distintas estrategias de agregación de cortes.  Este script asume que los
archivos 3D se han preprocesado en imágenes 2D y que existe un DataFrame
con las rutas a cada corte y sus etiquetas.  Utiliza un modelo 2D basado
en timm (`Model2DTimm`), entrena con validación cruzada estratificada por
paciente y compara cuatro métodos de agregación para obtener la predicción
del volumen completo:

1. ``mean``: media de probabilidades y argmax;
2. ``vote``: votación mayoritaria de las clases de cada corte;
3. ``max``: escoge la clase con probabilidad máxima entre todos los cortes;
4. ``weighted``: media ponderada según la posición del corte.

Para cada fold registra las métricas en Weights & Biases (WandB).  Al final
selecciona la estrategia con mejor F1 macro promedio y guarda los pesos
del modelo y los resultados.

Uso:
    python train_lisa.py --train_csv path/to/train.csv --out_dir ./results

Los parámetros principales (modelo, número de epochs, etc.) se definen
mediante ``argparse``.  Para ejecutar inferencia en un conjunto de
prueba después del entrenamiento, se puede ejecutar el mismo script con
``--do_inference`` y ``--test_csv``.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import sys
import gc
import os

sys.path.append("../")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from src.utils import assign_patient_stratified_folds, compute_weights_from_df, set_seed
from clean_dataset2D import MRIDataset2D
from clean_models2D import Model2DTimm
from clean_trainer import update_optimizer
import wandb


def aggregate_mean(probs: np.ndarray, trues: np.ndarray, paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Agrupación por media de probabilidades (argmax sobre la media)."""
    grouped_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_trues: Dict[str, List[np.ndarray]] = defaultdict(list)
    # Derivar nombre de volumen LISA_xxx_LF_xyz.nii.gz a partir del nombre de la imagen
    for p, pr, tr in zip(paths, probs, trues):
        vol = "_".join(os.path.basename(p).split(".")[0].split("_")[:-1]) + ".nii.gz"
        grouped_probs[vol].append(pr)
        grouped_trues[vol].append(tr)
    final_preds = []
    final_probs = []
    final_trues = []
    final_names = []
    for name in grouped_probs:
        mean_prob = np.mean(grouped_probs[name], axis=0)
        pred = np.argmax(mean_prob, axis=1)
        true = np.round(np.mean(grouped_trues[name], axis=0)).astype(int)
        final_preds.append(pred)
        final_probs.append(mean_prob)
        final_trues.append(true)
        final_names.append(name)
    return (np.stack(final_preds), np.stack(final_probs), np.stack(final_trues), final_names)


def aggregate_vote(probs: np.ndarray, trues: np.ndarray, paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Agrupación por votación mayoritaria de clases (mode)."""
    grouped_votes: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_trues: Dict[str, List[np.ndarray]] = defaultdict(list)
    for p, pr, tr in zip(paths, probs, trues):
        vol = "_".join(os.path.basename(p).split(".")[0].split("_")[:-1]) + ".nii.gz"
        grouped_votes[vol].append(np.argmax(pr, axis=1))
        grouped_trues[vol].append(tr)
    final_preds = []
    final_trues = []
    final_names = []
    for name in grouped_votes:
        # mode along slices for each label
        votes = np.stack(grouped_votes[name])  # (num_slices, num_labels)
        mode_preds = []
        for j in range(votes.shape[1]):
            counts = Counter(votes[:, j])
            mode_preds.append(max(counts, key=counts.get))
        true = np.round(np.mean(grouped_trues[name], axis=0)).astype(int)
        final_preds.append(np.array(mode_preds))
        final_trues.append(true)
        final_names.append(name)
    return (np.stack(final_preds), None, np.stack(final_trues), final_names)


def aggregate_max(probs: np.ndarray, trues: np.ndarray, paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Agrupación por máximo: selecciona la clase con probabilidad máxima entre todas las láminas."""
    grouped_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_trues: Dict[str, List[np.ndarray]] = defaultdict(list)
    for p, pr, tr in zip(paths, probs, trues):
        vol = "_".join(os.path.basename(p).split(".")[0].split("_")[:-1]) + ".nii.gz"
        grouped_probs[vol].append(pr)
        grouped_trues[vol].append(tr)
    final_preds = []
    final_trues = []
    final_names = []
    for name in grouped_probs:
        arr = np.stack(grouped_probs[name], axis=0)  # (num_slices, L, C)
        # Tomamos el índice de la probabilidad máxima a lo largo de las slices
        max_idx = np.argmax(arr, axis=0)  # (L, C)
        # max_idx contiene el índice de la slice con la mayor probabilidad por etiqueta y clase
        # Seleccionamos la clase con probabilidad máxima por etiqueta
        preds = []
        for j in range(arr.shape[1]):
            # arr[:, j] -> (num_slices, C)
            slice_idx, class_idx = np.unravel_index(arr[:, j, :].argmax(), arr[:, j, :].shape)
            preds.append(class_idx)
        true = np.round(np.mean(grouped_trues[name], axis=0)).astype(int)
        final_preds.append(np.array(preds))
        final_trues.append(true)
        final_names.append(name)
    return (np.stack(final_preds), None, np.stack(final_trues), final_names)


def aggregate_weighted(probs: np.ndarray, trues: np.ndarray, paths: List[str], n_slices: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Agrupación por media ponderada según la posición del corte.

    Se da más peso a las láminas centrales.  Los pesos se calculan como
    ``1 - abs(i - (n_slices-1)/2) / ((n_slices-1)/2)``, de modo que la lámina
    central tenga peso 1 y las extremas peso 0.
    """
    # Precalcular pesos
    indices = np.arange(n_slices)
    center = (n_slices - 1) / 2.0
    weights = 1.0 - np.abs(indices - center) / center
    # Normalizar
    weights = weights / weights.sum()
    grouped_probs: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_trues: Dict[str, List[np.ndarray]] = defaultdict(list)
    grouped_indices: Dict[str, List[int]] = defaultdict(list)
    for p, pr, tr in zip(paths, probs, trues):
        vol = "_".join(os.path.basename(p).split(".")[0].split("_")[:-1]) + ".nii.gz"
        # Extraer índice de corte del nombre (asume patrón *_<idx>.png).  Si no se puede extraer,
        # se asigna un índice incremental.
        try:
            idx = int(os.path.splitext(p)[0].split("_")[-1])
        except ValueError:
            idx = len(grouped_probs[vol])
        grouped_probs[vol].append(pr)
        grouped_trues[vol].append(tr)
        grouped_indices[vol].append(idx)
    final_preds = []
    final_probs = []
    final_trues = []
    final_names = []
    for name in grouped_probs:
        # Ordenar según índice
        order = np.argsort(grouped_indices[name])
        arr = np.stack([grouped_probs[name][i] for i in order], axis=0)  # (num_slices, L, C)
        wts = np.array([weights[min(i, len(weights)-1)] for i in sorted(grouped_indices[name])])
        wts = wts / wts.sum()
        # Ponderar y sumar
        weighted_prob = np.tensordot(wts, arr, axes=(0, 0))  # (L, C)
        preds = np.argmax(weighted_prob, axis=1)
        true = np.round(np.mean(grouped_trues[name], axis=0)).astype(int)
        final_preds.append(preds)
        final_probs.append(weighted_prob)
        final_trues.append(true)
        final_names.append(name)
    return (np.stack(final_preds), np.stack(final_probs), np.stack(final_trues), final_names)


def evaluate_aggregator(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute F1 macro, F1 micro and accuracy per volume."""
    res: Dict[str, float] = {}
    res["f1_macro"] = f1_score(y_true.flatten(), y_pred.flatten(), average="macro", zero_division=0)
    res["f1_micro"] = f1_score(y_true.flatten(), y_pred.flatten(), average="micro", zero_division=0)
    res["acc"] = accuracy_score(y_true.flatten(), y_pred.flatten())
    return res


def train_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, label_cols: Iterable[str], args, run) -> Dict[str, float]:
    """Entrena un fold, evalúa distintas estrategias de agregación y devuelve resultados.

    Se construye un DataLoader de entrenamiento e inferencia con las clases
    definidas en ``clean_dataset2D`` y ``clean_models2D``.  Tras entrenar,
    se realizan predicciones de validación por corte, se agregan con los
    cuatro métodos y se calculan las métricas.  Las métricas se registran en
    WandB mediante ``run.log``.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Construir datasets
    train_ds = MRIDataset2D(train_df, is_train=True, use_augmentation=True, is_numpy=True, labels=label_cols, image_size=args.image_size)
    val_ds = MRIDataset2D(val_df, is_train=True, use_augmentation=False, is_numpy=True, labels=label_cols, image_size=args.image_size)
    # DataLoaders
    if args.use_sampling:
        weights = [1.0 for _ in range(len(train_ds))]
        sampler = None
    else:
        sampler = None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    model = Model2DTimm(
        base_model=args.base_model,
        in_channels=args.in_channels,
        num_labels=len(label_cols),
        num_classes=3,
        pretrained=True,
        head_type=args.head_type,
        head_dropout=args.head_dropout,
        drop_path_rate=args.drop_path_rate,
        use_view=args.use_view,
    )
    model = model.to(device)
    # Optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.95, patience=5)
    # Loss per label (CrossEntropy)
    criterions = [torch.nn.CrossEntropyLoss() for _ in label_cols]

    best_f1 = 0.0
    patience_counter = args.patience
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y, _, view in train_loader:
            x, y, view = x.to(device), y.to(device), view.to(device)
            with torch.cuda.amp.autocast():
                logits = model(x, view)
                loss = sum(
                    criterions[i](logits[:, i], y[:, i]) for i in range(len(label_cols))
                ) / len(label_cols)
            scaler.scale(loss).backward()
            update_optimizer(model, optimiser, scaler)
            epoch_loss += loss.item()
        # Validación por cortes
        model.eval()
        val_probs = []
        val_trues = []
        val_paths = []
        with torch.no_grad():
            for x, y, path, view in val_loader:
                x, y, view = x.to(device), y.to(device), view.to(device)
                logits = model(x, view)
                # Probs por etiqueta
                batch_probs = torch.stack([
                    torch.softmax(logits[:, i], dim=-1) for i in range(len(label_cols))
                ], dim=1)
                val_probs.append(batch_probs.cpu())
                val_trues.append(y.cpu())
                val_paths.extend(path)
        probs_all = torch.cat(val_probs, dim=0).numpy()
        trues_all = torch.cat(val_trues, dim=0).numpy()
        # Evaluar agregaciones
        results = {}
        for agg_name in ["mean", "vote", "max", "weighted"]:
            if agg_name == "mean":
                y_pred, y_prob, y_true, _ = aggregate_mean(probs_all, trues_all, val_paths)
            elif agg_name == "vote":
                y_pred, _, y_true, _ = aggregate_vote(probs_all, trues_all, val_paths)
            elif agg_name == "max":
                y_pred, _, y_true, _ = aggregate_max(probs_all, trues_all, val_paths)
            else:  # weighted
                y_pred, y_prob, y_true, _ = aggregate_weighted(probs_all, trues_all, val_paths, args.n_slices)
            metrics = evaluate_aggregator(y_pred, y_true)
            results[f"val_{agg_name}_f1_macro"] = metrics["f1_macro"]
            results[f"val_{agg_name}_f1_micro"] = metrics["f1_micro"]
            results[f"val_{agg_name}_acc"] = metrics["acc"]
        # Seleccionar mejor agregación según F1 macro
        best_agg = max(["mean", "vote", "max", "weighted"], key=lambda a: results[f"val_{a}_f1_macro"])
        if results[f"val_{best_agg}_f1_macro"] > best_f1:
            best_f1 = results[f"val_{best_agg}_f1_macro"]
            patience_counter = args.patience
            # Guardar mejores pesos y agregación
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            best_aggregation = best_agg
        else:
            patience_counter -= 1
            if patience_counter == 0:
                break
        # Logging en WandB
        log_data = {
            "epoch": epoch,
            "train_loss": epoch_loss / len(train_loader),
        }
        log_data.update(results)
        run.log(log_data)
    # Cargar mejores pesos
    model.load_state_dict(best_state_dict)
    return {
        "best_f1_macro": best_f1,
        "best_aggregation": best_aggregation,
        "model": model,
    }


def run_training(args):
    """Controla la validación cruzada, el registro en WandB y la inferencia final."""
    set_seed(args.seed)
    df = pd.read_csv(args.train_csv)
    df["patient_id"] = df["filename"].str.extract(r"(LISA_\d+)")
    # Asignar folds estratificados por paciente
    df, _ = assign_patient_stratified_folds(df, n_splits=args.n_splits, top_k=2, seed=args.seed), None
    # Columnas de etiquetas
    label_cols = args.label_cols.split(",")
    # Iniciar run de WandB
    run = wandb.init(project=args.wandb_project,entity=args.entity, name=args.experiment_name, config=vars(args))
    fold_results = []
    for fold in range(args.n_splits):
        train_df = df[df.fold != fold].reset_index(drop=True)
        val_df = df[df.fold == fold].reset_index(drop=True)
        res = train_fold(train_df, val_df, label_cols, args, run)
        fold_results.append(res)
        run.log({f"fold_{fold}_best_f1": res["best_f1_macro"]})
        # Guardar modelo
        model_path = os.path.join(args.out_dir, f"model_fold{fold}.pt")
        torch.save(res["model"].state_dict(), model_path)
    # Seleccionar agregación global más repetida
    best_aggs = [r["best_aggregation"] for r in fold_results]
    global_best = Counter(best_aggs).most_common(1)[0][0]
    run.log({"global_best_aggregation": global_best})
    run.finish()
    print(f"Mejor agregación global: {global_best}")


def run_inference(args):
    """Realiza inferencia en un conjunto de prueba utilizando los modelos entrenados."""
    df_test = pd.read_csv(args.test_csv)
    df_test["patient_id"] = df_test["filename"].str.extract(r"(LISA_\d+)")
    label_cols = args.label_cols.split(",")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = []
    for fold in range(args.n_splits):
        model = Model2DTimm(
            base_model=args.base_model,
            in_channels=args.in_channels,
            num_labels=len(label_cols),
            num_classes=3,
            pretrained=False,
            head_type=args.head_type,
            head_dropout=args.head_dropout,
            drop_path_rate=args.drop_path_rate,
            use_view=args.use_view,
        )
        model_path = os.path.join(args.out_dir, f"model_fold{fold}.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    # Dataset y loader
    test_ds = MRIDataset2D(df_test, is_train=False, use_augmentation=False, is_numpy=True,
                           labels=label_cols, image_size=args.image_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Agregación
    all_probs = []
    all_trues = []
    all_paths = []
    with torch.no_grad():
        for x, _, path, view in test_loader:
            x, view = x.to(device), view.to(device)
            # Ensemble de modelos: media de logits
            ensemble_logits = None
            for m in models:
                logits = m(x, view)
                if ensemble_logits is None:
                    ensemble_logits = logits / len(models)
                else:
                    ensemble_logits += logits / len(models)
            batch_probs = torch.stack([
                torch.softmax(ensemble_logits[:, i], dim=-1) for i in range(len(label_cols))
            ], dim=1)
            all_probs.append(batch_probs.cpu())
            all_trues.append(np.zeros((batch_probs.shape[0], len(label_cols))))
            all_paths.extend(path)
    probs_all = torch.cat(all_probs, dim=0).numpy()
    trues_all = np.vstack(all_trues).astype(int)
    # Elegir la mejor agregación (igual que en training)
    # Por simplicidad se usa mean; el usuario puede ajustar a global_best
    if args.aggregation == "mean":
        y_pred, y_prob, y_true, filenames = aggregate_mean(probs_all, trues_all, all_paths)
    elif args.aggregation == "vote":
        y_pred, _, y_true, filenames = aggregate_vote(probs_all, trues_all, all_paths)
    elif args.aggregation == "max":
        y_pred, _, y_true, filenames = aggregate_max(probs_all, trues_all, all_paths)
    else:
        y_pred, y_prob, y_true, filenames = aggregate_weighted(probs_all, trues_all, all_paths, args.n_slices)
    # Crear DataFrame de salida
    records = []
    for fname, pred in zip(filenames, y_pred):
        row = {"filename": fname}
        for i, lbl in enumerate(label_cols):
            row[lbl] = int(pred[i])
        records.append(row)
    df_pred = pd.DataFrame(records)
    df_pred.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
    print(f"Archivo de predicciones guardado en {args.out_dir}/submission.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--train_csv", type=str, default="train.csv", help="Ruta al CSV de entrenamiento preprocesado")
    #parser.add_argument("--test_csv", type=str, default="test.csv", help="Ruta al CSV de prueba")
    parser.add_argument("--train_csv", type=str, default="../results/preprocessed_data/df_train_imgs.csv")
    parser.add_argument("--test_csv", type=str, default="../results/preprocessed_data/df_test_imgs.csv")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directorio de salida para modelos y predicciones")
    parser.add_argument("--experiment_name", type=str, default="lisa_experiment", help="Nombre del experimento para WandB")
    parser.add_argument("--wandb_project", type=str, default="lisa2025", help="Nombre del proyecto WandB")
    parser.add_argument("--wandb_project", type=str, default="ml_projects", help="Nombre del proyecto WandB")
    parser.add_argument("--n_splits", type=int, default=5, help="Número de folds para validación cruzada")
    parser.add_argument("--image_size", type=int, default=256, help="Tamaño de las imágenes de entrada")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de batch")
    parser.add_argument("--num_workers", type=int, default=4, help="Número de workers de DataLoader")
    parser.add_argument("--epochs", type=int, default=30, help="Épocas de entrenamiento")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia para early stopping")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--in_channels", type=int, default=1, help="Número de canales de entrada (1 para MRI en escala de grises)")
    parser.add_argument("--base_model", type=str, default="maxvit_tiny_tf_512.in1k", help="Backbone de timm")
    parser.add_argument("--head_type", type=str, default="label_tokens", choices=["simple", "label_tokens"], help="Tipo de cabeza de clasificación")
    parser.add_argument("--head_dropout", type=float, default=0.2, help="Dropout en la cabeza")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="Drop path rate para el backbone")
    parser.add_argument("--use_view", action="store_true", help="Concatena la vista (axial/coronal/sagital) a las features del backbone")
    parser.add_argument("--n_slices", type=int, default=40, help="Número total de cortes por volumen (para agregación ponderada)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo de cómputo")
    parser.add_argument("--label_cols", type=str, default="Noise,Zipper,Positioning,Banding,Motion,Contrast,Distortion", help="Etiquetas separadas por comas")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--do_inference", action="store_true", help="Ejecuta inferencia en lugar de entrenamiento")
    parser.add_argument("--aggregation", type=str, default="mean", choices=["mean","vote","max","weighted"], help="Método de agregación para inferencia")
    parser.add_argument("--use_sampling", action="store_true", help="Usa muestreo ponderado en el DataLoader (no implementado en este ejemplo)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.do_inference:
        run_inference(args)
    else:
        run_training(args)


    """
    python train_lisa.py \
  --device cuda:3 \
  --out_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/lisa_clean \
  --experiment_name lisa_model \
  --wandb_project lisa2025 \
  --n_splits 3 \
  --image_size 256 \
  --batch_size 16 \
  --epochs 30 \
  --lr 1e-4 \
  --head_type label_tokens \
  --use_view \
  --drop_path_rate 0.1 \
  --in_channels 1 \
  --seed 42

    """
    pass
