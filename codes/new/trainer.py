"""
Training loop and evaluation utilities for the LISA 2025 challenge.

This module implements cross‑validation training with stratified splits,
per‑label class weighting, multiple aggregation strategies for combining
slice predictions into volume predictions, and comprehensive logging to
Weights & Biases (WandB).  It relies on the dataset and model modules
defined in this repository and uses the utility functions from
``new_repo.utils`` for fold assignment and weight computation.

Key features
------------

* **Cross‑validation training**: Uses stratified group K‑fold splitting
  by patient to ensure that volumes from the same patient do not appear
  in both training and validation sets.  The user specifies ``n_splits``.

* **Class balancing**: Weights are computed per label using either
  class‑balanced effective numbers or inverse frequency.  These weights
  can be used with CrossEntropy or Focal loss to mitigate class
  imbalance.

* **Aggregation strategies**: Because volumes are represented by many
  slices, the raw predictions must be aggregated into a single score per
  volume.  Four strategies are provided: ``mean`` (average
  probabilities), ``vote`` (majority class per label), ``max`` (class
  corresponding to the maximum probability across slices) and ``weighted``
  (weighted average of probabilities, currently uniform weights).  All
  strategies are evaluated on validation and test sets, and their F1
  scores and class distributions are logged to WandB.

* **WandB integration**: Runs are configured via environment variables
  ``WANDB_API_KEY``, ``PROJECT_WANDB`` and ``ENTITY``.  Metrics for
  training loss, validation F1 and counts of predicted classes are
  logged.  Aggregation results are recorded under keys like
  ``val_mean_f1_macro``.  Users can monitor the training live.

Usage
-----

The primary entry point is ``train_and_evaluate``, which accepts data
frames for training and an optional validation dataset (``test_back_df``)
with known labels for additional evaluation.  After training, the
function returns a dictionary with aggregated results and saves model
weights to disk.  See the main script ``train.py`` for an example
command‑line interface.
"""

from __future__ import annotations

import os
import json
import joblib
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset import MRIDataset2D
from model import Model2DTimm
from utils import (assign_patient_stratified_folds, assign_volume_stratified_folds,
                    compute_weights_from_df, compute_sample_weights, set_seed)


def softmax_logits(y_hat: torch.Tensor) -> torch.Tensor:
    """Compute softmax over the class dimension for each label.

    ``y_hat`` has shape (B, L, C).  Returns a tensor of the same shape
    containing probabilities.
    """
    # Stack softmax per label
    return torch.stack([
        torch.softmax(y_hat[:, i], dim=-1) for i in range(y_hat.shape[1])
    ], dim=1)


def collect_slice_predictions(loader: DataLoader, model: nn.Module, device: torch.device) -> List[Dict]:
    """Run inference on a dataloader and collect per‑slice predictions.

    Returns a list of dictionaries, each containing the following keys:
    ``'path'`` (slice path), ``'probs'`` (numpy array of shape (L, C)),
    ``'preds'`` (numpy array of shape (L,)), and ``'true'`` (numpy array
    of shape (L,) or ``None`` if labels are not available).
    """
    results: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for x, y, path, view,aux_target in tqdm(loader, desc="Inference", leave=False):
            x = x.to(device, non_blocking=True)
            view = view.to(device, non_blocking=True)
            y_hat,aux_pred = model(x, view) if 'view' in model.forward.__code__.co_varnames else model(x)
            probs = softmax_logits(y_hat).cpu().numpy()  # (B,L,C)
            preds = np.argmax(probs, axis=-1)            # (B,L)
            y_np: Optional[np.ndarray] = None
            # Determine if true labels are available.  In the test set the
            # dataset returns ``y = -1`` (scalar) or a 1‑D tensor of -1s; in
            # that case we omit the labels.  When training/validation the
            # labels have shape (B, L).
            if isinstance(y, torch.Tensor):
                # If y is 2‑D (batch, labels) then labels exist
                if y.ndim > 1:
                    y_np = y.cpu().numpy()
                # If y is 1‑D (batch,), check if it's not all -1
                elif y.ndim == 1 and y.numel() > 0 and (y[0].item() != -1):
                    y_np = y.unsqueeze(1).cpu().numpy()
            for i in range(len(path)):
                results.append({
                    'path': path[i],
                    'probs': probs[i],
                    'preds': preds[i],
                    'true': y_np[i] if y_np is not None else None,
                })
    return results

def compute_stats_per_view(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute clipping percentiles and mean/std for each view code based on
    the slices in ``df``.  Returns a dictionary mapping view code to
    ``{'p1', 'p99', 'mean', 'std'}`` used for dataset_z_per_view normalisation.
    """
    import numpy as np
    import os
    stats: Dict[str, Dict[str, float]] = {}
    buckets: Dict[str, List[np.ndarray]] = {}
    for idx in range(len(df)):
        row = df.iloc[idx]
        path = row.get('npy_path')# or row.get('img_path') or row.get('path') or row.get('filename')
        if not path:
            continue
        # Determine view code from file name
        view_code = 'axi'
        try:
            base_name = os.path.basename(path)
            if '_LF_' in base_name:
                view_code = base_name.split('_LF_')[-1].split('_')[0]
        except Exception:
            view_code = 'axi'
        # Load array (support .npy or .pkl)
        load_path = path
        if load_path.endswith('.npy'):
            load_path = load_path.replace('.npy', '.pkl')
        arr = joblib.load(load_path).astype(np.float32)
        buckets.setdefault(view_code, []).append(arr.ravel())
    for view, arrs in buckets.items():
        if not arrs:
            continue
        x = np.concatenate(arrs)
        p1 = np.percentile(x, 1)
        p99 = np.percentile(x, 99)
        x_clip = np.clip(x, p1, p99)
        mu = x_clip.mean()
        sigma = x_clip.std() + 1e-6
        stats[view] = {
            'p1': float(p1),
            'p99': float(p99),
            'mean': float(mu),
            'std': float(sigma),
        }
    print("COMPUTED! dataset_z_per_view")
    return stats


def aggregate_slices(results: List[Dict], num_labels: int, num_classes: int,
                     aggregator: str = 'mean', weights: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """Aggregate slice predictions into volume predictions using a given strategy.

    Parameters
    ----------
    results: list of dicts
        Output of ``collect_slice_predictions``.
    num_labels: int
        Number of labels per slice.
    num_classes: int
        Number of discrete classes (e.g. 3).
    aggregator: str, default='mean'
        Aggregation strategy: one of ``'mean'``, ``'vote'``, ``'max'`` or
        ``'weighted'``.  See below for definitions.
    weights: Optional[np.ndarray]
        Optional weights for the weighted average.  Must have shape
        (n_slices,).  If ``None`` and ``aggregator`` is ``'weighted'``, all
        slices are weighted equally.

    Returns
    -------
    y_pred: np.ndarray of shape (N_volumes, num_labels)
        Predicted class per label for each volume.
    y_prob: np.ndarray of shape (N_volumes, num_labels, num_classes)
        Aggregated probabilities per label and class.
    y_true: Optional[np.ndarray]
        True labels per volume (if available).  ``None`` if no ground
        truths were provided in ``results``.
    vol_names: list of str
        Names of the volumes (derived from slice paths).

    Aggregation strategies
    ----------------------

    * ``mean``: probabilities are averaged across slices.
    * ``vote``: for each label, take the class with the highest count
      among slice‑level predictions.
    * ``max``: for each label, take the class corresponding to the
      maximum probability observed among all slices (max‑pooling).
    * ``weighted``: probabilities are combined via a weighted average.
      If ``weights`` is ``None``, equal weights are used.
    """
    # Group by volume name: strip slice index and suffix
    volume_groups: Dict[str, List[Dict]] = {}
    for item in results:
        # Derive volume name: remove last underscore and digits from base name
        # Example: path '.../LISA_0001_LF_axi_023.png' -> 'LISA_0001_LF_axi.nii.gz'
        base = os.path.basename(item['path'])
        # Remove extension
        name_no_ext = os.path.splitext(base)[0]
        parts = name_no_ext.split('_')
        if parts[-1].isdigit():
            parts = parts[:-1]
        vol_name = '_'.join(parts) + '.nii.gz'
        volume_groups.setdefault(vol_name, []).append(item)
    # Aggregate predictions per volume
    vol_names = list(volume_groups.keys())
    n_vols = len(vol_names)
    y_pred = np.zeros((n_vols, num_labels), dtype=int)
    y_prob = np.zeros((n_vols, num_labels, num_classes), dtype=float)
    y_true = None
    # Determine if we have ground truths
    has_true = any(item['true'] is not None for item in results)
    if has_true:
        y_true = np.zeros((n_vols, num_labels), dtype=int)
    for i, name in enumerate(vol_names):
        group = volume_groups[name]
        # Stack per‑slice predictions
        probs = np.stack([item['probs'] for item in group], axis=0)    # (n_slices, L, C)
        preds = np.stack([item['preds'] for item in group], axis=0)    # (n_slices, L)
        # Combine according to aggregator
        if aggregator == 'mean':
            agg_probs = np.mean(probs, axis=0)  # (L,C)
            agg_pred = np.argmax(agg_probs, axis=1)  # (L,)
        elif aggregator == 'vote':
            agg_probs = np.mean(probs, axis=0)  # still compute mean for completeness
            # majority vote per label
            agg_pred = []
            for lbl in range(num_labels):
                counts = np.bincount(preds[:, lbl], minlength=num_classes)
                agg_pred.append(np.argmax(counts))
            agg_pred = np.array(agg_pred, dtype=int)
        elif aggregator == 'max':
            agg_probs = np.max(probs, axis=0)
            agg_pred = np.argmax(agg_probs, axis=1)
        elif aggregator == 'weighted':
            # Weighted average across slices
            n_slices = probs.shape[0]
            if weights is None:
                w = np.ones(n_slices, dtype=float) / n_slices
            else:
                w = np.asarray(weights, dtype=float)
                if w.shape[0] != n_slices:
                    # broadcast or normalise weights per volume
                    w = np.ones(n_slices, dtype=float) / n_slices
                w = w / w.sum()
            # Weighted average per label/class
            agg_probs = np.tensordot(w, probs, axes=(0, 0))  # (L,C)
            agg_pred = np.argmax(agg_probs, axis=1)
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
        y_prob[i] = agg_probs
        y_pred[i] = agg_pred
        if has_true:
            # True labels: majority or mean rounding
            trues = np.stack([item['true'] for item in group], axis=0)  # (n_slices, L)
            # Majority vote for ground truth (should be consistent across slices)
            # but in case of disagreement take the mean and round
            avg_true = np.mean(trues, axis=0)
            true_label = np.round(avg_true).astype(int)
            y_true[i] = true_label
    return y_pred, y_prob, y_true, vol_names

class DynamicFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(DynamicFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        p_t = torch.gather(prob, 1, targets.unsqueeze(1)).squeeze()
        alpha_t = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = alpha_t * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def evaluate_aggregators(results: List[Dict], label_cols: Iterable[str], num_classes: int = 3,
                         aggregators: Iterable[str] = ('mean', 'vote', 'max', 'weighted')) -> Dict[str, Dict]:
    """Evaluate multiple aggregation strategies on collected slice predictions.

    Parameters
    ----------
    results: list of dicts
        Output from ``collect_slice_predictions``.
    label_cols: Iterable[str]
        Names of the label columns.
    num_classes: int
        Number of classes per label.
    aggregators: iterable of str
        List of aggregator names to evaluate.

    Returns
    -------
    metrics: dict
        Dictionary keyed by aggregator name.  Each value is another
        dictionary with keys:

        - ``'f1_macro'``: macro F1 score across all labels.
        - ``'f1_micro'``: micro F1 score across all labels.
        - ``'acc'``: accuracy across all labels.
        - ``'per_label_f1'``: dict mapping each label to its macro F1.
        - ``'pred_counts'``: dict mapping each label to a dict of counts
          for classes 0, 1, 2.
    """
    label_cols = list(label_cols)
    num_labels = len(label_cols)
    metrics: Dict[str, Dict] = {}
    for agg in aggregators:
        y_pred, y_prob, y_true, names = aggregate_slices(
            results, num_labels=num_labels, num_classes=num_classes, aggregator=agg
        )
        # y_true may be None if test set has no labels
        if y_true is None:
            # Only return predictions and counts
            pred_counts = {}
            for i, col in enumerate(label_cols):
                counts = np.bincount(y_pred[:, i], minlength=num_classes)
                pred_counts[col] = {cls: int(counts[cls]) for cls in range(num_classes)}
            metrics[agg] = {
                'pred_counts': pred_counts,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'names': names,
            }
            continue
        # Flatten to compute global metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        # Compute classification metrics
        f1_macro = float(f1_score(y_true_flat, y_pred_flat, average='macro'))
        f1_micro = float(f1_score(y_true_flat, y_pred_flat, average='micro'))
        acc      = float((y_true_flat == y_pred_flat).mean())
        # Per label F1 and cross‑entropy loss
        per_label_f1: Dict[str, float] = {}
        per_label_loss: Dict[str, float] = {}
        pred_counts: Dict[str, Dict[int, int]] = {}
        # Compute negative log likelihood per label
        epsilon = 1e-8
        for i, col in enumerate(label_cols):
            yt = y_true[:, i]
            yp = y_pred[:, i]
            # F1 macro per label
            per_label_f1[col] = float(f1_score(yt, yp, average='macro'))
            counts = np.bincount(yp, minlength=num_classes)
            pred_counts[col] = {cls: int(counts[cls]) for cls in range(num_classes)}
            # Cross entropy loss per label
            # y_prob: (N_vol, L, C) aggregated probabilities
            probs_label = y_prob[:, i, :]  # (N_vol, C)
            # Convert true labels to one‑hot indices
            # Compute negative log probability of the true class
            # Add small epsilon to avoid log(0)
            nll = -np.log(probs_label[np.arange(probs_label.shape[0]), yt] + epsilon)
            per_label_loss[col] = float(np.mean(nll))
        # Macro loss across labels
        loss_macro = float(np.mean(list(per_label_loss.values())))
        metrics[agg] = {
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'acc': acc,
            'per_label_f1': per_label_f1,
            'per_label_loss': per_label_loss,
            'loss_macro': loss_macro,
            'pred_counts': pred_counts,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'y_true': y_true,
            'names': names,
        }
    return metrics


def epoch_subsample_frac_unique_patients(df, frac=0.1, seed=None):
    """
    Selecciona un subconjunto de pacientes (frac del total) y 
    devuelve exactamente un registro aleatorio por cada paciente seleccionado.
    """
    rng = np.random.default_rng(seed)

    # Lista de patient_id únicos y muestreo del frac
    unique_patients = df["patient_id"].unique()
    n_select = max(1, int(len(unique_patients) * frac))
    selected_patients = rng.choice(unique_patients, size=n_select, replace=False)

    # Filtrar DataFrame
    df_selected = df[df["patient_id"].isin(selected_patients)]

    # Elegir un registro aleatorio por paciente (sin warning)
    df_sampled = (
        df_selected.groupby("patient_id", group_keys=False, sort=False)
                   .apply(lambda g: g.sample(n=5, random_state=seed, replace=True), include_groups=False)
                   .reset_index(drop=True)
    )

    return df_sampled


class DynamicFocalLoss2(nn.Module):
    def __init__(self, alpha=(0.25, 0.75, 1.0), gamma=2.0, reduction='mean'):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: (B,3), targets: (B,)
        ce = F.cross_entropy(inputs, targets, reduction='none')
        prob = F.softmax(inputs, dim=1)
        p_t = prob[torch.arange(prob.size(0), device=prob.device), targets]
        alpha_t = self.alpha[targets]               # <-- por clase
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

import torch

def _yolo_to_corners(box: torch.Tensor) -> torch.Tensor:
    """
    box: (..., 4) con (cx, cy, w, h) en [0,1]
    return: (..., 4) con (x1, y1, x2, y2) en [0,1]
    """
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _box_area_xyxy(box_xyxy: torch.Tensor) -> torch.Tensor:
    w = (box_xyxy[..., 2] - box_xyxy[..., 0]).clamp(min=0)
    h = (box_xyxy[..., 3] - box_xyxy[..., 1]).clamp(min=0)
    return w * h

@torch.no_grad()
def iou_yolo(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU entre cajas YOLO normalizadas.
    box1, box2: tensores con shape (..., 4) en formato (cx, cy, w, h), valores en [0,1].
                 Se permite broadcasting en las dimensiones iniciales.
    return: IoU con shape broadcasted (...,)

    Ej.: box1.shape = (B,4), box2.shape = (B,4)  -> IoU shape (B,)
         box1.shape = (B,4), box2.shape = (1,4)  -> IoU shape (B,)
    """
    b1 = _yolo_to_corners(box1)
    b2 = _yolo_to_corners(box2)

    # intersección
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])

    inter = _box_area_xyxy(torch.stack([x1, y1, x2, y2], dim=-1))
    area1 = _box_area_xyxy(b1)
    area2 = _box_area_xyxy(b2)
    union = area1 + area2 - inter
    return inter / (union + eps)

@torch.no_grad()
def giou_yolo(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Generalized IoU para cajas en formato YOLO normalizado.
    return: GIoU con shape broadcasted (...,)
    """
    b1 = _yolo_to_corners(box1)
    b2 = _yolo_to_corners(box2)

    # IoU normal
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = _box_area_xyxy(torch.stack([x1, y1, x2, y2], dim=-1))
    area1 = _box_area_xyxy(b1)
    area2 = _box_area_xyxy(b2)
    union = area1 + area2 - inter
    iou = inter / (union + eps)

    # caja envolvente mínima
    cx1 = torch.min(b1[..., 0], b2[..., 0])
    cy1 = torch.min(b1[..., 1], b2[..., 1])
    cx2 = torch.max(b1[..., 2], b2[..., 2])
    cy2 = torch.max(b1[..., 3], b2[..., 3])
    c_area = _box_area_xyxy(torch.stack([cx1, cy1, cx2, cy2], dim=-1))

    giou = iou - (c_area - union) / (c_area + eps)
    return giou


def train_and_evaluate(train_df: pd.DataFrame,
                       test_back_df: Optional[pd.DataFrame],
                       label_cols: Iterable[str],
                       args: Dict,
                       aggregators: Iterable[str] = ('mean', 'vote', 'max', 'weighted')) -> Dict:
    """Train models with cross‑validation and evaluate aggregators.

    Parameters
    ----------
    train_df: DataFrame
        Training data.  Must include columns for labels, ``patient_id``,
        ``img_path``/``npy_path`` and ``path``/``filename``.
    test_back_df: DataFrame or ``None``
        Additional evaluation set with ground truth labels.  If provided,
        aggregator metrics will be computed on this set as well.  This
        corresponds to the ``df_test_back`` in the original code.
    label_cols: iterable of str
        Names of the label columns.
    args: dict
        Hyperparameters and options.  Expected keys include ``n_splits``,
        ``batch_size``, ``epochs``, ``patience``, ``lr``, ``weight_decay``,
        ``image_size``, ``in_channels``, ``base_model``, ``pretrained``,
        ``head_type``, ``head_config``, ``use_view``, ``slice_frac``,
        ``seed``, ``use_sampling``, ``device``, ``weight_method``,
        ``weight_beta``, ``weight_alpha``, ``weight_cap``.
    aggregators: iterable of str
        Names of aggregation strategies to evaluate.

    Returns
    -------
    results: dict
        Contains overall cross‑validation metrics and per‑fold details,
        along with aggregator results for validation and test back sets.
    """
    label_cols = list(label_cols)
    n_splits = args.get('n_splits', 5)
    device   = torch.device(args.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    # Assign folds
    # If the training DataFrame already contains a 'fold' column, use it directly.
    # Otherwise, decide how to split: by volume or by patient depending on args.
    if 'fold' in train_df.columns:
        df = train_df.copy()
        # Ensure fold values are integers and within range [0, n_splits-1]
        if not pd.api.types.is_integer_dtype(df['fold']):
            df['fold'] = df['fold'].astype(int)
    else:
        split_by_volume = args.get('split_by_volume', False)
        volume_id_col = args.get('volume_id', 'patient_id')
        if split_by_volume:
            # Use volume-level stratified splitting
            df = assign_volume_stratified_folds(train_df, volume_id_col=volume_id_col,
                                                label_cols=label_cols,
                                                n_splits=n_splits,
                                                top_k=args.top_k,
                                                seed=args.get('seed', 42))
        else:
            # Default: stratify by patient using rare labels
            df = assign_patient_stratified_folds(train_df, n_splits=n_splits, seed=args.get('seed', 42))
    # Compute class weights per label
    weights = compute_weights_from_df(
        df, labels=label_cols,
        method=args.get('weight_method', 'effective'),
        beta=args.get('weight_beta', 0.99),
        alpha=args.get('weight_alpha', 0.5),
        cap=args.get('weight_cap', 8.0),
        device=device,
        dtype=torch.float32,
    )
    # Create loss functions per label (weighted CrossEntropy)
    criterions: List[nn.Module] = []
    for lbl in label_cols:
        w = weights[lbl]
        print(args.get('dynamic_w', "0.25, 0.75, 1.0"))
        criterions.append(DynamicFocalLoss2(
        alpha=torch.tensor([float(value) for value in args.get('dynamic_w', "0.25, 0.75, 1.0").split(",")], dtype=torch.float, device=device),
        gamma=2.0)
        )
      #nn.CrossEntropyLoss(weight=w))
        #criterions.append(DynamicFocalLoss(alpha=0.25, gamma=2.0)) #nn.CrossEntropyLoss(weight=w))
    # Logging with WandB
    # Determine if WandB logging is enabled via environment variables.  We will
    # create a new WandB run for each fold rather than one global run.  This
    # prevents mixing of step counters across folds.
    use_wandb = bool(os.getenv('WANDB_API_KEY')) and bool(os.getenv('PROJECT_WANDB'))
    # A WandB run will be initialised within each fold if enabled
    run = None
    # Store metrics
    fold_results = []
    # Paths to save models
    out_dir = args.get('save_dir', './models')
    os.makedirs(out_dir, exist_ok=True)
    # Loop over folds
    for fold in range(n_splits):
        # Split data
        train_fold = df[df['fold'] != fold].reset_index(drop=True)
        val_fold   = df[df['fold'] == fold].reset_index(drop=True)
        # Determine normalisation mode and compute per-view stats for this fold if needed
        norm_mode = args.get('norm_mode', 'slice_z')
        per_view_stats = None
        if norm_mode == 'dataset_z_per_view':
            print("Compute dataset_z_per_view")
            per_view_stats = compute_stats_per_view(train_fold)
            # Save stats to disk for inference
            stats_file = os.path.join(out_dir, f"per_view_stats_fold{fold}.json")
            with open(stats_file, 'w') as f:
                json.dump(per_view_stats, f)
        # Create datasets with explicit normalisation
        train_ds = MRIDataset2D(train_fold, is_train=True, use_augmentation=True,
                                is_numpy=True, labels=label_cols, image_size=args.get('image_size', 224),
                                norm_mode=norm_mode, per_view_stats=per_view_stats)
        val_ds   = MRIDataset2D(val_fold,   is_train=True, use_augmentation=False,
                                is_numpy=True, labels=label_cols, image_size=args.get('image_size', 224),
                                norm_mode=norm_mode, per_view_stats=per_view_stats)
        # DataLoaders
        if args.get('use_sampling', True):
            sample_weights = compute_sample_weights(train_fold, label_cols)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=args.get('batch_size', 16), sampler=sampler,
                                      num_workers=args.get('num_workers', 4), pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.get('batch_size', 16), shuffle=True,
                                      num_workers=args.get('num_workers', 4), pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.get('batch_size', 16), shuffle=False,
                                num_workers=args.get('num_workers', 4), pin_memory=True)
        # Instantiate model
        model = Model2DTimm(
            base_model=args.get('base_model', 'maxvit_tiny_tf_512.in1k'),
            in_channels=args.get('in_channels', 1),
            num_labels=len(label_cols),
            num_classes=3,
            pretrained=args.get('pretrained', True),
            head_type=args.get('head_type', 'label_tokens'),
            head_config=args.get('head_config', {}),
            use_view=args.get('use_view', False),
        ).to(device)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.get('lr', 1e-4), weight_decay=args.get('weight_decay', 1e-4))
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(),"lr":args.get('lr', 1e-4)},
            {"params": model.view_emb.parameters(),"lr":args.get('lr', 1e-4)},
            {"params": model.head.parameters(),"lr":args.get('lr', 1e-4)},
            {"params": model.reg_head.parameters(),"lr":5*args.get('lr', 1e-4)}],
            weight_decay=args.get('weight_decay', 1e-4)
        )
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.95, patience=args.get('scheduler_patience', 5)
        )
        # Initialise a WandB run for this fold if logging is enabled
        if use_wandb:
            import wandb
            exp_name = args.get('experiment_name', 'lisa_experiment')
            run = wandb.init(
                project=os.getenv('PROJECT_WANDB'),
                entity=os.getenv('ENTITY'),
                name=f"{exp_name}_fold{fold}",
                group=exp_name,
                config=args,
                reinit=True,
            )
        # Training loop
        best_f1 = 0.0
        patience_counter = args.get('patience', 5)
        best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        for epoch in range(args.get('epochs', 30)):
            # Optional slice fraction for each epoch
            if args.get('slice_frac', 1.0) < 1.0:
                frac = args['slice_frac']
                """
                train_fold_ep = (
                    train_fold.groupby('patient_id', group_keys=False)
                        .sample(frac=frac, random_state=epoch)
                        .reset_index(drop=True)
                )"""
                train_fold_ep = epoch_subsample_frac_unique_patients(train_fold, frac=0.1, seed=None)

                train_ds_ep = MRIDataset2D(train_fold_ep, is_train=True, use_augmentation=True,
                                           is_numpy=True, labels=label_cols, image_size=args.get('image_size', 224),
                                           norm_mode=norm_mode, per_view_stats=per_view_stats)
                if args.get('use_sampling', True):
                    sample_weights = compute_sample_weights(train_fold_ep, label_cols)
                    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                    train_loader = DataLoader(train_ds_ep, batch_size=args.get('batch_size', 16), sampler=sampler,
                                              num_workers=args.get('num_workers', 4), pin_memory=True)
                else:
                    train_loader = DataLoader(train_ds_ep, batch_size=args.get('batch_size', 16), shuffle=True,
                                              num_workers=args.get('num_workers', 4), pin_memory=True)
            # Train one epoch
            model.train()
            running_loss = 0.0
            nbatches = 0
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}", leave=False, dynamic_ncols=True)
            for step, (x, y, _, view,aux_target) in enumerate(pbar, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                view = view.to(device, non_blocking=True)
                aux_target = aux_target.to(device, non_blocking=True)  # (B,3) en [0,1]

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    y_hat,aux_pred  = model(x, view) if args.get('use_view', False) else model(x)
                    cls_loss = sum(criterions[i](y_hat[:, i], y[:, i]) for i in range(len(label_cols))) / len(label_cols)
                    
                    #aux_loss = F.smooth_l1_loss(aux_pred, aux_target)
                    #loss = cls_loss + args.get("lambda_aux", 0.2) * aux_loss
                    aux_l1  = torch.nn.functional.smooth_l1_loss(aux_pred, aux_target)
                    aux_giou = (1.0 - giou_yolo(aux_pred, aux_target)).mean()
                    aux_loss = 0.5 * aux_l1 + 0.5 * aux_giou
                    loss = cls_loss + args.get("lambda_aux", 0.5) * aux_loss   # súbelo a 0.3–1.0 para probar

                grad_scaler.scale(loss).backward()
                # Update optimizer
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                running_loss += float(loss.item())
                nbatches += 1
                # Do not log per batch.  Logging happens at the end of the epoch with a global step.
            epoch_loss = running_loss / max(1, nbatches)
            # ------- Validation at epoch end -------
            # 1) Compute slice‑level validation loss to compare with training loss.
            model.eval()
            val_loss_total = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for x_val, y_val, _, view_val,aux_target_val in val_loader:
                    x_val   = x_val.to(device, non_blocking=True)
                    y_val   = y_val.to(device, non_blocking=True)
                    view_val= view_val.to(device, non_blocking=True)
                    aux_target_val = aux_target_val.to(device, non_blocking=True)  # (B,3) en [0,1]
                    with torch.cuda.amp.autocast():
                        y_hat_val,aux_pred_val = model(x_val, view_val) if args.get('use_view', False) else model(x_val)
                        cls_loss_val = sum(
                            criterions[i](y_hat_val[:, i], y_val[:, i]) for i in range(len(label_cols))
                        ) / len(label_cols)
                        #aux_loss_val = F.smooth_l1_loss(aux_pred_val, aux_target_val)
                        #loss_val = cls_loss_val + args.get("lambda_aux", 0.2) * aux_loss_val

                        aux_l1_val  = torch.nn.functional.smooth_l1_loss(aux_pred_val, aux_target_val)
                        aux_giou_val = (1.0 - giou_yolo(aux_pred_val, aux_target_val)).mean()
                        aux_loss_val = 0.5 * aux_l1_val + 0.5 * aux_giou_val
                        loss_val = cls_loss_val + args.get("lambda_aux", 0.5) * aux_loss_val   # súbelo a 0.3–1.0 para probar


                    val_loss_total += float(loss_val.item())
                    n_val_batches += 1
            val_loss_slice_level = val_loss_total / max(1, n_val_batches)
            # 2) Collect per‑slice predictions on val set for aggregator evaluation
            val_results = collect_slice_predictions(val_loader, model, device)
            val_metrics = evaluate_aggregators(val_results, label_cols, num_classes=3, aggregators=aggregators)
            # 3) Choose mean aggregator to compute scheduler metric
            val_f1 = val_metrics['mean']['f1_macro'] if 'mean' in val_metrics and 'f1_macro' in val_metrics['mean'] else 0.0
            scheduler.step(val_f1)
            # ------- Logging per epoch -------
            # Print a summary of validation metrics for each aggregator to the console
            print(f"Fold {fold} Epoch {epoch} (train_loss={epoch_loss:.4f}, val_loss={val_loss_slice_level:.4f}):", end=" ")
            summary_strs = []
            for agg_name, m in val_metrics.items():
                if 'f1_macro' in m:
                    summary_strs.append(f"{agg_name}: F1_macro={m['f1_macro']:.4f}, acc={m['acc']:.4f}")
            print(" | ".join(summary_strs))
            # Accumulate metrics into a dictionary and log once per epoch using local step
            if use_wandb and run is not None:
                step_epoch = epoch  # local step for this fold's run
                log_data: Dict[str, float | int] = {
                    'epoch': epoch,
                    'train/loss': epoch_loss,
                    'val/loss': val_loss_slice_level,
                }
                for agg_name, m in val_metrics.items():
                    if 'f1_macro' in m:
                        log_data[f'val_global/val_{agg_name}_f1_macro'] = m['f1_macro']
                        log_data[f'val_global/val_{agg_name}_f1_micro'] = m['f1_micro']
                        log_data[f'val_global/val_{agg_name}_acc'] = m['acc']
                        log_data[f'val_global/val_{agg_name}_loss_macro'] = m.get('loss_macro', 0.0)
                        for lbl in label_cols:
                            log_data[f'val_{agg_name}_f1/{lbl}'] = m['per_label_f1'].get(lbl, 0.0)
                            log_data[f'val_{agg_name}_loss/{lbl}'] = m['per_label_loss'].get(lbl, 0.0)
                            for cls in range(3):
                                cnt = m['pred_counts'][lbl].get(cls, 0)
                                log_data[f'val_{agg_name}_counts/{lbl}_{cls}'] = cnt
                run.log(log_data, step=step_epoch)
            # Track best model based on mean aggregator F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = args.get('patience', 5)
                best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter -= 1
                if patience_counter == 0:
                    # Early stopping
                    break
        # Save best model for fold
        model.load_state_dict(best_state_dict)
        model_path = os.path.join(out_dir, f"model_fold{fold}.pt")
        torch.save(best_state_dict, model_path)
        # Evaluate aggregators on val set again using best model
        best_val_results = collect_slice_predictions(val_loader, model, device)
        best_val_metrics = evaluate_aggregators(best_val_results, label_cols, num_classes=3, aggregators=aggregators)
        # Evaluate aggregators on test_back if provided
        test_back_metrics = None
        if test_back_df is not None and not test_back_df.empty:
            test_back_ds = MRIDataset2D(test_back_df, is_train=True, use_augmentation=False,
                                       is_numpy=True, labels=label_cols, image_size=args.get('image_size', 224),
                                       norm_mode=norm_mode, per_view_stats=per_view_stats)
            test_back_loader = DataLoader(test_back_ds, batch_size=args.get('batch_size', 16), shuffle=False,
                                          num_workers=args.get('num_workers', 4), pin_memory=True)
            test_back_results = collect_slice_predictions(test_back_loader, model, device)
            test_back_metrics = evaluate_aggregators(test_back_results, label_cols, num_classes=3, aggregators=aggregators)
        # Print final validation metrics and optional test_back metrics for this fold
        print(f"Fold {fold} final validation metrics:")
        for agg_name, m in best_val_metrics.items():
            if 'f1_macro' in m:
                print(f"  {agg_name}: F1_macro={m['f1_macro']:.4f}, acc={m['acc']:.4f}")
        if test_back_metrics:
            print(f"Fold {fold} test_back metrics:")
            for agg_name, m in test_back_metrics.items():
                if 'f1_macro' in m:
                    print(f"  {agg_name}: F1_macro={m['f1_macro']:.4f}, acc={m['acc']:.4f}")
        # Log aggregator metrics for fold and close this fold's WandB run
        fold_results.append({
            'fold': fold,
            'best_f1': best_f1,
            'val_metrics': best_val_metrics,
            'test_back_metrics': test_back_metrics,
        })

        # ----- Save per-aggregator predictions and probabilities -----
        # For each aggregator, if predictions and probabilities are available,
        # write them to CSV files for further analysis.  We save one file per
        # aggregator and dataset (val, test_back) per fold.  Each row contains
        # the filename, the predicted class per label, and the probability of
        # each class per label.
        def _save_preds(metrics: Dict[str, Dict], dataset_name: str) -> None:
            for agg_name, m in metrics.items():
                # y_prob and y_pred are only present when y_true is available (val/test_back)
                if 'y_prob' in m and 'y_pred' in m and 'names' in m:
                    y_pred_arr = m['y_pred']  # shape (N, L)
                    y_prob_arr = m['y_prob']  # shape (N, L, C)
                    names_list = m['names']
                    # Construct DataFrame
                    df_pred = pd.DataFrame({'filename': names_list})
                    for i, lbl in enumerate(label_cols):
                        df_pred[lbl] = y_pred_arr[:, i]
                        for cls_idx in range(y_prob_arr.shape[-1]):
                            df_pred[f'{lbl}_{cls_idx}'] = y_prob_arr[:, i, cls_idx]
                    # Ensure directory exists
                    agg_dir = os.path.join(out_dir, f'preds_fold{fold}')
                    os.makedirs(agg_dir, exist_ok=True)
                    # File path
                    file_name = f'{dataset_name}_{agg_name}_fold{fold}_probs.csv'
                    file_path = os.path.join(agg_dir, file_name)
                    df_pred.to_csv(file_path, index=False)
                    print(f"Saved {dataset_name} predictions for aggregator '{agg_name}' to {file_path}")

                    file_name = f'{dataset_name}_{agg_name}_fold{fold}_preds.csv'
                    file_path = os.path.join(agg_dir, file_name)
                    df_pred[["filename"]+label_cols].to_csv(file_path, index=False)
                    print(f"Saved {dataset_name} predictions for aggregator '{agg_name}' to {file_path}")

        # Save validation predictions
        _save_preds(best_val_metrics, 'val')
        # Save test_back predictions
        if test_back_metrics:
            _save_preds(test_back_metrics, 'test_back')
        if use_wandb and run is not None:
            # Prepare summary metrics for this fold (best F1 and optional test_back)
            summary_data: Dict[str, float | int] = {
                'best_f1': best_f1,
            }
            if test_back_metrics:
                for agg_name, m in test_back_metrics.items():
                    if 'f1_macro' in m:
                        summary_data[f'test_back_global/test_back_{agg_name}_f1_macro'] = m['f1_macro']
                        summary_data[f'test_back_global/test_back_{agg_name}_f1_micro'] = m['f1_micro']
                        summary_data[f'test_back_global/test_back_{agg_name}_acc'] = m['acc']
                        summary_data[f'test_back_global/test_back_{agg_name}_loss_macro'] = m.get('loss_macro', 0.0)
                        for lbl in label_cols:
                            summary_data[f'test_back_{agg_name}_f1/{lbl}'] = m['per_label_f1'][lbl]
                            summary_data[f'test_back_{agg_name}_loss/{lbl}'] = m['per_label_loss'][lbl]
                            for cls in range(3):
                                cnt = m['pred_counts'][lbl].get(cls, 0)
                                summary_data[f'test_back_{agg_name}_counts/{lbl}_{cls}'] = cnt
            # Log at the final epoch step for this fold's run
            step_summary = args.get('epochs', 30)
            run.log(summary_data, step=step_summary)
            # Finish this fold's WandB run
            run.finish()
    # Ensemble across folds on validation and test_back sets
    # For each aggregator, average probabilities across folds and recompute predictions
    ensemble_results = {}
    if n_splits > 1:
        # Collect per fold predictions for val and test_back
        val_preds_by_fold = {agg: [] for agg in aggregators}
        test_preds_by_fold = {agg: [] for agg in aggregators}
        val_names = None
        test_names = None
        for fr in fold_results:
            # Validation
            for agg in aggregators:
                m = fr['val_metrics'][agg]
                val_preds_by_fold[agg].append(m['y_prob'])  # (N, L, C)
                val_names = m['names']
            # Test back
            if test_back_df is not None and fr['test_back_metrics']:
                for agg in aggregators:
                    m = fr['test_back_metrics'][agg]
                    test_preds_by_fold[agg].append(m['y_prob'])
                    test_names = m['names']
        # Compute ensemble metrics by aggregating metrics across folds rather than stacking
        val_ensemble_metrics: Dict[str, Dict] = {}
        test_ensemble_metrics: Dict[str, Dict] = {}
        for agg in aggregators:
            # Validation ensemble: average metrics across folds
            metrics_list = [fr['val_metrics'][agg] for fr in fold_results if agg in fr['val_metrics']]
            if metrics_list:
                f1_macro_avg = float(np.mean([m['f1_macro'] for m in metrics_list]))
                f1_micro_avg = float(np.mean([m['f1_micro'] for m in metrics_list]))
                acc_avg      = float(np.mean([m['acc'] for m in metrics_list]))
                per_label_f1_avg: Dict[str, float] = {}
                per_label_loss_avg: Dict[str, float] = {}
                pred_counts_sum: Dict[str, Dict[int, int]] = {lbl: {0:0,1:0,2:0} for lbl in label_cols}
                for lbl in label_cols:
                    per_label_f1_avg[lbl] = float(np.mean([m['per_label_f1'][lbl] for m in metrics_list]))
                    per_label_loss_avg[lbl] = float(np.mean([m['per_label_loss'][lbl] for m in metrics_list]))
                    # Sum counts across folds
                    for m in metrics_list:
                        for cls in range(3):
                            pred_counts_sum[lbl][cls] += m['pred_counts'][lbl].get(cls, 0)
                loss_macro_avg = float(np.mean(list(per_label_loss_avg.values())))
                val_ensemble_metrics[agg] = {
                    'f1_macro': f1_macro_avg,
                    'f1_micro': f1_micro_avg,
                    'acc': acc_avg,
                    'per_label_f1': per_label_f1_avg,
                    'per_label_loss': per_label_loss_avg,
                    'loss_macro': loss_macro_avg,
                    'pred_counts': pred_counts_sum,
                }
            # Test_back ensemble: average metrics across folds
            if test_back_df is not None:
                metrics_test_list = []
                for fr in fold_results:
                    if fr['test_back_metrics'] and agg in fr['test_back_metrics']:
                        metrics_test_list.append(fr['test_back_metrics'][agg])
                if metrics_test_list:
                    f1_macro_avg = float(np.mean([m['f1_macro'] for m in metrics_test_list]))
                    f1_micro_avg = float(np.mean([m['f1_micro'] for m in metrics_test_list]))
                    acc_avg      = float(np.mean([m['acc'] for m in metrics_test_list]))
                    per_label_f1_avg: Dict[str, float] = {}
                    per_label_loss_avg: Dict[str, float] = {}
                    pred_counts_sum: Dict[str, Dict[int, int]] = {lbl: {0:0,1:0,2:0} for lbl in label_cols}
                    for lbl in label_cols:
                        per_label_f1_avg[lbl] = float(np.mean([m['per_label_f1'][lbl] for m in metrics_test_list]))
                        per_label_loss_avg[lbl] = float(np.mean([m['per_label_loss'][lbl] for m in metrics_test_list]))
                        for m in metrics_test_list:
                            for cls in range(3):
                                pred_counts_sum[lbl][cls] += m['pred_counts'][lbl].get(cls, 0)
                    loss_macro_avg = float(np.mean(list(per_label_loss_avg.values())))
                    test_ensemble_metrics[agg] = {
                        'f1_macro': f1_macro_avg,
                        'f1_micro': f1_micro_avg,
                        'acc': acc_avg,
                        'per_label_f1': per_label_f1_avg,
                        'per_label_loss': per_label_loss_avg,
                        'loss_macro': loss_macro_avg,
                        'pred_counts': pred_counts_sum,
                    }
        ensemble_results['val'] = val_ensemble_metrics
        ensemble_results['test_back'] = test_ensemble_metrics
        # Print ensemble metrics to the console
        if val_ensemble_metrics:
            print("Ensemble validation results:")
            for agg_name, m in val_ensemble_metrics.items():
                print(f"  {agg_name}: F1_macro={m['f1_macro']:.4f}, acc={m['acc']:.4f}")
        if test_ensemble_metrics:
            print("Ensemble test_back results:")
            for agg_name, m in test_ensemble_metrics.items():
                print(f"  {agg_name}: F1_macro={m['f1_macro']:.4f}, acc={m['acc']:.4f}")
        # If WandB is enabled, log ensemble metrics in a separate run
        if use_wandb and (val_ensemble_metrics or test_ensemble_metrics):
            import wandb
            exp_name = args.get('experiment_name', 'lisa_experiment')
            en_run = wandb.init(
                project=os.getenv('PROJECT_WANDB'),
                entity=os.getenv('ENTITY'),
                name=f"{exp_name}_ensemble",
                group=exp_name,
                config=args,
                reinit=True,
            )
            log_data: Dict[str, float | int] = {}
            for agg_name, m in val_ensemble_metrics.items():
                log_data[f'ensemble_val_{agg_name}_f1_macro'] = m['f1_macro']
                log_data[f'ensemble_val_{agg_name}_f1_micro'] = m['f1_micro']
                log_data[f'ensemble_val_{agg_name}_acc'] = m['acc']
                log_data[f'ensemble_val_{agg_name}_loss_macro'] = m.get('loss_macro', 0.0)
                for lbl in label_cols:
                    log_data[f'ensemble_val_{agg_name}_f1/{lbl}'] = m['per_label_f1'][lbl]
                    log_data[f'ensemble_val_{agg_name}_loss/{lbl}'] = m['per_label_loss'][lbl]
                    for cls in range(3):
                        log_data[f'ensemble_val_{agg_name}_counts/{lbl}_{cls}'] = m['pred_counts'][lbl][cls]
            for agg_name, m in test_ensemble_metrics.items():
                log_data[f'ensemble_test_back_global/ensemble_test_back_{agg_name}_f1_macro'] = m['f1_macro']
                log_data[f'ensemble_test_back_global/ensemble_test_back_{agg_name}_f1_micro'] = m['f1_micro']
                log_data[f'ensemble_test_back_global/ensemble_test_back_{agg_name}_acc'] = m['acc']
                log_data[f'ensemble_test_back_global/ensemble_test_back_{agg_name}_loss_macro'] = m.get('loss_macro', 0.0)
                for lbl in label_cols:
                    log_data[f'ensemble_test_back_{agg_name}_f1/{lbl}'] = m['per_label_f1'][lbl]
                    log_data[f'ensemble_test_back_{agg_name}_loss/{lbl}'] = m['per_label_loss'][lbl]
                    for cls in range(3):
                        log_data[f'ensemble_test_back_{agg_name}_counts/{lbl}_{cls}'] = m['pred_counts'][lbl][cls]
            # Log at step 0 for the ensemble run
            en_run.log(log_data, step=0)
            en_run.finish()
    # Return results
    return {
        'fold_results': fold_results,
        'ensemble_results': ensemble_results,
    }
