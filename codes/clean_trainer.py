"""
Training utilities for the LISA 2025 challenge.

This module contains a simplified and cleaned version of the original training
loop.  The main changes relative to the original implementation are:

* Removed the duplicate ``optimizer.step()`` call after ``update_optimizer``.
  The helper function ``update_optimizer`` already unscales gradients,
  performs a step and zeroes the gradients.  Calling ``optimizer.step()``
  again immediately afterward would lead to an additional (often zero) step.
* The ``Model2DTimm`` forward signature now accepts an optional ``view``
  tensor.  All model calls have been updated to pass the one‑hot view
  vector when available.
* Slightly simplified logging; users can integrate their own monitoring or
  WandB logic on top of these functions.

The module exposes a single function ``train_single_fold`` for clarity.  Users
should handle cross‑validation outside this module.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from clean_models2D import Model2DTimm
from clean_dataset2D import MRIDataset2D


class SlidingWindowAverage:
    """Track the average of the last ``window_size`` values."""

    def __init__(self, window_size: int = 10) -> None:
        self.values = deque(maxlen=window_size)

    def update(self, val: float) -> None:
        self.values.append(val)

    def get(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))

    def is_full(self) -> bool:
        return len(self.values) == self.values.maxlen


def update_optimizer(model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     grad_scaler: torch.cuda.amp.GradScaler | None,
                     clip_grad_max_norm: float = 0.5,
                     lr_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None) -> None:
    """Perform a gradient update with optional gradient clipping and AMP.

    This helper unscales gradients (when using AMP), clips them, performs the
    optimisation step, resets gradients and optionally updates a scheduler.
    """
    if grad_scaler:
        grad_scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)
    if grad_scaler:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if lr_scheduler:
        lr_scheduler.step()


def inference_dataset(loader: DataLoader,
                      model: nn.Module,
                      device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run inference on a dataset and aggregate predictions per volume.

    Args:
        loader: DataLoader providing batches of ``(img, label, path, view)``.
        model: Model with signature ``model(x, view)``.
        device: Device on which to run the model.

    Returns:
        y_pred: Array of predicted classes of shape ``(N, L)``.
        y_prob: Array of class probabilities of shape ``(N, L, C)``.
        y_true: Array of true labels of shape ``(N, L)``.
        filenames: List of unique volume filenames.
    """
    val_probs, val_trues, val_paths = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, path, view in loader:
            x = x.to(device)
            view = view.to(device)
            y_hat = model(x, view)
            # softmax per label
            batch_probs = torch.stack([
                torch.softmax(y_hat[:, i], dim=-1) for i in range(y_hat.shape[1])
            ], dim=1)
            val_probs.append(batch_probs.cpu())
            val_trues.append(y.cpu())
            val_paths.extend(path)
    # concatenate
    probs_all = torch.cat(val_probs, dim=0).numpy()  # (M, L, C)
    trues_all = torch.cat(val_trues, dim=0).numpy()  # (M, L)
    # derive volume filename by stripping slice index
    paths_all = ["_".join(p.split("/")[-1].split(".")[0].split("_")[:-1]) + ".nii.gz" for p in val_paths]
    from collections import defaultdict
    grouped_probs: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_trues: dict[str, list[np.ndarray]] = defaultdict(list)
    for p, pr, tr in zip(paths_all, probs_all, trues_all):
        grouped_probs[p].append(pr)
        grouped_trues[p].append(tr)
    final_preds = []
    final_probs = []
    final_trues = []
    final_names = []
    for name in grouped_probs:
        mean_prob = np.mean(grouped_probs[name], axis=0)  # (L, C)
        pred = np.argmax(mean_prob, axis=1)
        mean_true = np.mean(grouped_trues[name], axis=0)
        true = np.round(mean_true).astype(int)
        final_preds.append(pred)
        final_probs.append(mean_prob)
        final_trues.append(true)
        final_names.append(name)
    return (np.stack(final_preds), np.stack(final_probs),
            np.stack(final_trues), final_names)


def train_single_fold(train_df,
                      val_df,
                      label_cols: Iterable[str],
                      model: Model2DTimm,
                      device: torch.device,
                      batch_size: int = 16,
                      lr: float = 1e-4,
                      weight_decay: float = 1e-4,
                      epochs: int = 30,
                      patience: int = 5,
                      use_sampling: bool = True,
                      num_workers: int = 4,
                      slice_frac: float = 1.0,
                      criterions: Iterable[nn.Module] | None = None,
                      ) -> Tuple[nn.Module, float]:
    """Train a single fold and return the best model and best F1 score.

    Args:
        train_df: DataFrame for training.
        val_df: DataFrame for validation.
        label_cols: List of label column names.
        model: Model to train.
        device: Device for computation.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: Weight decay.
        epochs: Maximum number of epochs.
        patience: Number of epochs without improvement before stopping.
        use_sampling: Whether to use a ``WeightedRandomSampler`` for class balance.
        num_workers: Number of DataLoader workers.
        slice_frac: Fraction of training data to sample each epoch.
        criterions: List of per‑label loss functions.  If ``None``, CrossEntropy
            loss is used for each label.

    Returns:
        A tuple containing the best model (with loaded weights) and the best
        F1 macro score achieved on the validation set.
    """
    # Prepare datasets
    train_ds_full = MRIDataset2D(df=train_df, is_train=True, use_augmentation=True,
                                 is_numpy=True, labels=label_cols, image_size=256)
    val_ds = MRIDataset2D(df=val_df, is_train=True, use_augmentation=False,
                          is_numpy=True, labels=label_cols, image_size=256)
    if criterions is None:
        # Equal weight cross‑entropy per label
        criterions = [nn.CrossEntropyLoss() for _ in label_cols]

    # Set up optimiser and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=5)

    # DataLoader for validation
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=False)

    best_f1 = 0.0
    patience_counter = patience
    for epoch in range(epochs):
        # Subsample training data if requested
        if slice_frac < 1.0:
            # Sample within each patient
            train_df_ep = (
                train_df.groupby("patient_id", group_keys=False)
                .sample(frac=slice_frac, random_state=epoch)
                .reset_index(drop=True)
            )
        else:
            train_df_ep = train_df
        train_ds = MRIDataset2D(df=train_df_ep, is_train=True, use_augmentation=True,
                                is_numpy=True, labels=label_cols, image_size=256)
        if use_sampling:
            # Compute sample weights based on label distribution
            # Each sample gets a weight equal to the inverse frequency of its least
            # represented class.  This is a simple heuristic; more elaborate
            # weighting schemes can be used.
            counts = train_df_ep[label_cols].apply(lambda x: tuple(x), axis=1)
            freq = counts.value_counts().to_dict()
            weights = [1.0 / freq[tuple(train_df_ep.loc[i, label_cols])] for i in range(len(train_df_ep))]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                      num_workers=num_workers, persistent_workers=False)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, persistent_workers=False)

        model.train()
        total_loss = 0.0
        for x, y, _, view in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y, view = x.to(device), y.to(device), view.to(device)
            with torch.cuda.amp.autocast():
                y_hat = model(x, view)
                loss = sum(
                    criterions[i](y_hat[:, i], y[:, i]) for i in range(len(label_cols))
                ) / len(label_cols)
            grad_scaler.scale(loss).backward()
            update_optimizer(model, optimizer, grad_scaler)
            total_loss += loss.item()

        # Validation
        y_pred, y_prob, y_true, _ = inference_dataset(val_loader, model, device)
        val_f1 = f1_score(y_true.flatten(), y_pred.flatten(), average="macro")
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = patience
            # Save best model weights in memory
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter -= 1
            if patience_counter == 0:
                break
    # Load best model state dict
    model.load_state_dict(best_state_dict)
    return model, best_f1