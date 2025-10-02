"""
This script orchestrates data loading, preprocessing, cross‑validation
training, aggregator evaluation and optional test inference.  It reads
configuration parameters from the command line and environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import MRIDataset2D
from src.model import Model2DTimm
from src.trainer import (aggregate_slices, collect_slice_predictions,
                         train_and_evaluate)
from src.utils import LABELS, set_seed

# Here you can set WandB environment variables if needed
os.environ["WANDB_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_CACHE_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_ARTIFACT_DIR"] = "/data/cristian/paper_2025/wandb_dir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate LISA models with cross validation.")
    parser.add_argument(
        '--top_k',
        type=int,
        default=2,
        help='Número de etiquetas raras para la estratificación')
    parser.add_argument(
        "--train_csv",
        type=str,
        default="./results/preprocessed_data/df_test_imgs.csv")
    parser.add_argument(
        "--test_csv",
        type=str,
        default="../../results/preprocessed_data/df_test_imgs.csv")
    parser.add_argument(
        '--test_back_csv',
        type=str,
        default=None,
        help='Path to test_back CSV with labels')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models',
        help='Directory to save models and outputs')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./models',
        help='Directory to save models and outputs')
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='new_model_bbox_giou_brain0.1_l0.1',
        help='Name of the experiment for WandB')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross‑validation splits')
    parser.add_argument(
        '--threshold_brain_presence',
        type=float,
        default=0.1,
        help='threshold for filtering slices')
    parser.add_argument(
        '--dynamic_w',
        type=str,
        default="0.25,0.5,1.0",
        help='threshold for filtering slices')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Maximum number of epochs per fold')
    parser.add_argument(
        '--patience',
        type=int,
        default=200,
        help='Number of epochs without improvement before stopping')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument(
        '--lambda_aux',
        type=float,
        default=0.1,
        help='lambda_aux for centroid of brain')

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay')
    parser.add_argument(
        '--base_model',
        type=str,
        default='maxvit_nano_rw_256.sw_in1k',
        help='timm model name')
    parser.add_argument(
        '--head_type',
        type=str,
        default='label_tokens',
        choices=[
            'simple',
            'label_tokens'],
        help='Type of classification head')
    parser.add_argument('--no_use_view', dest='use_view', action='store_false',
                        help='Disable view information (default: enabled)')
    parser.set_defaults(use_view=True)

    parser.add_argument(
        '--no_do_inference',
        dest='do_inference',
        action='store_false',
        help='Disable view information (default: enabled)')
    parser.set_defaults(do_inference=True)

    parser.add_argument(
        '--pretrained',
        type=int,
        default=0,
        help='Use pretrained backbone (1=yes, 0=no)')
    parser.add_argument(
        '--image_size',
        type=int,
        default=256,
        help='Image size (square) for resizing slices')
    parser.add_argument(
        '--in_channels',
        type=int,
        default=1,
        help='Number of input channels')
    parser.add_argument(
        '--slice_frac',
        type=float,
        default=0.99,
        help='Fraction of training slices used per epoch')
    parser.add_argument(
        '--use_sampling',
        type=int,
        default=1,
        help='Use WeightedRandomSampler (1=yes, 0=no)')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader worker processes')
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help='CUDA/CPU device (e.g. cuda:0)')
    parser.add_argument(
        '--weight_method',
        type=str,
        default='effective',
        choices=[
            'effective',
            'invfreq'],
        help='Method for class weights')
    parser.add_argument(
        '--weight_beta',
        type=float,
        default=0.99,
        help='Beta for effective number class weights')
    parser.add_argument(
        '--weight_alpha',
        type=float,
        default=0.5,
        help='Alpha for inverse frequency class weights')
    parser.add_argument(
        '--weight_cap',
        type=float,
        default=8.0,
        help='Max cap for inverse frequency class weights')
    parser.add_argument(
        '--head_config',
        type=str,
        default=None,
        help='JSON string for head configuration')
    parser.add_argument(
        '--aggregators',
        nargs='+',
        default=['mean'],
        help='Aggregation strategies to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--scheduler_patience',
        type=int,
        default=5,
        help='Patience for LR scheduler')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=50,
        help='Steps between logging to WandB')
    parser.add_argument(
        '--do_inference',
        action='store_true',
        help='Perform inference on test set after training')
    parser.add_argument(
        '--split_by_volume',
        action='store_true',
        help='Use volume-level stratified folds instead of patient-level')
    parser.add_argument(
        '--volume_id',
        type=str,
        default='patient_id',
        help='Column name to identify volumes when splitting by volume')
    parser.add_argument(
        '--norm_mode',
        type=str,
        default='dataset_z_per_view',
        choices=[
            'none',
            'slice_z',
            'dataset_z_per_view'],
        help="""Normalisation mode for input images. "slice_z" applies z-score per slice (legacy behaviour)',
                              "dataset_z_per_view" applies clipping and z-score using fixed statistics per view computed on the training set, and "none" skips normalisation.""")
    return parser.parse_args()


def extract_patient_id(filename: str) -> str:
    """Extract the patient identifier from a filename.

    Expected patterns:
    - 'LISA_0001' or 'LISA_VALIDATION_0001'
    We take the first two segments (e.g. 'LISA_0001').
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    if len(parts) >= 2:
        base = '_'.join(parts[:-1])
        print("base 1:", base)
        return base
    print("base 2:", base)
    return base


def preprocess_df(df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    """Prepare the dataframe by ensuring required columns and casting labels.

    Adds a ``patient_id`` column derived from ``filename`` and ensures
    labels are integers.
    """
    df = df.copy()
    # Derive patient_id
    if 'patient_id' not in df.columns:
        df['patient_id'] = df['filename'].apply(extract_patient_id)
    # Cast labels to int
    for c in label_cols:
        if df[c].dtype != int:
            df[c] = df[c].astype(int)
    return df


def main() -> None:
    # Load environment variables
    # load_dotenv()
    args = parse_args()
    # Convert head_config JSON string to dict
    head_config = None
    if args.head_config:
        head_config = json.loads(args.head_config)
    # Compose args dict for trainer
    args_dict = vars(args).copy()
    args_dict['head_config'] = head_config
    # Set seed
    set_seed(args.seed)

    # Call training
    if not (args.do_inference and args.test_csv):
        from sklearn.model_selection import train_test_split

        # Load train data
        train_df = pd.read_csv(args.train_csv)
        train_df = preprocess_df(train_df, LABELS)

        print(f"Original: {train_df.shape}")
        train_df = train_df[train_df["ratio"] >=
                            args.threshold_brain_presence].reset_index()

        print(f"Original: {train_df.shape}")
        # train_df = filter_dataset_by_similarity_ssim(train_df, ssim_thresh=0.6)
        print(f"→ Filtrado: {train_df.shape}")

        # Load test_back if provided
        test_back_df = None
        if args.test_back_csv:
            test_back_df = pd.read_csv(args.test_back_csv)
            test_back_df = preprocess_df(test_back_df, LABELS)
        else:
            print("Doing TEST BACK")
            df_all = train_df.copy()
            df_all["patient_id"] = df_all["filename"].str.extract(
                r"(LISA_\d+)")
            train_ids, test_ids = train_test_split(
                df_all["patient_id"].unique(), test_size=0.1, random_state=42, shuffle=True)
            train_df = df_all[df_all["patient_id"].isin(
                train_ids)].reset_index(drop=True)
            test_back_df = df_all[df_all["patient_id"].isin(
                test_ids)].reset_index(drop=True)
            test_back_df = preprocess_df(test_back_df, LABELS)

        results = train_and_evaluate(
            train_df=train_df,
            test_back_df=test_back_df,
            label_cols=LABELS,
            args=args_dict,
            aggregators=args.aggregators
        )
    # Optionally run inference on final test set
    if args.do_inference and args.test_csv:
        """
        Perform inference on the test set using trained models and aggregators.
        """
        # Load test dataframe
        test_df = pd.read_csv(args.test_csv)
        print(f"df_test Original: {test_df.shape}")
        # We expect no labels; still need patient_id
        print("test_df['filename']:")
        # 📊 Unique patient_id antes del filtrado
        print("[INFO] Antes del filtrado → unique filename:",
              test_df['filename'].nunique() if 'filename' in test_df else "N/A")
        if 'filename' in list(test_df.columns):
            print("[INFO] Frecuencia antes del filtrado:")
            print(test_df['filename'].value_counts())

        # Filtrado por threshold
        test_df = test_df[test_df["ratio"] >=
                          args.threshold_brain_presence].reset_index(drop=True)
        print(f"df_test Filtrado (por ratio): {test_df.shape}")

        # 📊 Unique patient_id después del filtrado
        if 'filename' in list(test_df.columns):
            print(
                "[INFO] Después del filtrado → unique filename:",
                test_df['filename'].nunique())
            print("[INFO] Frecuencia después del filtrado:")
            print(test_df['filename'].value_counts())

        # Determine device
        device = torch.device(
            args.device) if args.device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # Import aggregator utilities
        # For each fold/model, collect slice-level predictions on a dataset
        # normalised with the fold's per-view stats
        slice_results_per_model: List[List[dict]] = []
        vol_names: Optional[List[str]] = None
        for fold in range(args.n_splits):
            # Load model for this fold
            model_path = os.path.join(args.model_dir, f"model_fold{fold}.pt")
            print("Reading model:", model_path)
            model = Model2DTimm(
                base_model=args.base_model,
                in_channels=args.in_channels,
                num_labels=len(LABELS),
                num_classes=3,
                pretrained=bool(args.pretrained),
                head_type=args.head_type,
                head_config=head_config,
                use_view=args.use_view,
            ).to(device)
            print("args.use_view:", args.use_view)
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            # Load per-view stats for this fold if using dataset_z_per_view
            per_view_stats_fold = None
            if args.norm_mode == 'dataset_z_per_view':
                print("Using dataset_z_per_view")
                stats_file = os.path.join(
                    args.model_dir, f"per_view_stats_fold{fold}.json")
                if os.path.exists(stats_file):
                    with open(stats_file, 'r') as f:
                        per_view_stats_fold = json.load(f)
            # Create dataset for this fold with appropriate normalisation
            test_ds_fold = MRIDataset2D(
                test_df, is_train=False, use_augmentation=False,
                is_numpy=True, labels=LABELS, image_size=args.image_size,
                norm_mode=args.norm_mode, per_view_stats=per_view_stats_fold
            )
            test_loader_fold = DataLoader(
                test_ds_fold, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            # Collect slice predictions for this fold
            fold_results = collect_slice_predictions(
                test_loader_fold, model, device)
            slice_results_per_model.append(fold_results)
            # Determine volume names using mean aggregator once
            if vol_names is None:
                _, _, _, names = aggregate_slices(
                    fold_results, num_labels=len(LABELS), num_classes=3, aggregator='mean')
                vol_names = names
        # For each aggregator, compute aggregated probabilities across models
        # and average
        for agg in args.aggregators:
            all_probs_per_model = []
            for fold_results in slice_results_per_model:
                _, y_prob, _, _ = aggregate_slices(
                    fold_results, num_labels=len(LABELS), num_classes=3, aggregator=agg)
                all_probs_per_model.append(y_prob)
            # Average probabilities across models (folds)
            mean_probs = np.mean(
                np.stack(
                    all_probs_per_model,
                    axis=0),
                axis=0)  # (N,L,C)
            y_pred_final = np.argmax(mean_probs, axis=-1)  # (N,L)
            # Construct submission DataFrame
            submission = pd.DataFrame({'filename': vol_names})
            for i, lbl in enumerate(LABELS):
                submission[lbl] = y_pred_final[:, i]
                for cls in range(mean_probs.shape[-1]):
                    submission[f'{lbl}_{cls}'] = mean_probs[:, i, cls]
            # Write predictions and probabilities to CSV
            suffix = agg
            sub_path = os.path.join(
                args.save_dir, f'submission_{suffix}_probs.csv')
            submission.to_csv(sub_path, index=False)
            print(f"Saved submission for aggregator '{agg}' to {sub_path}")
            sub_path = os.path.join(
                args.save_dir, f'submission_{suffix}_preds.csv')
            submission[["filename"] + LABELS].to_csv(sub_path, index=False)
            print(f"Saved submission for aggregator '{agg}' to {sub_path}")


if __name__ == '__main__':
    main()

    """
    /new_model_bbox_giou_brain0.1_l0.1

    python train.py --save_dir "./results/new_model_bbox_giou_brain0.1_l0.1" --train_csv "./results/preprocessed_data/df_test_imgs.csv" --test_csv "./results/preprocessed_data/df_test_imgs.csv" --aggregators mean vote max weighted

    python train.py --save_dir "./results]" --train_csv "./results/preprocessed_data/df_test_imgs.csv" --test_csv "./results/preprocessed_data/df_test_imgs.csv" --aggregators mean vote max weighted


    python train.py --save_dir E:\\Datathon\\LISA\\lisa-challenge2025-task1\results\new_model_bbox_giou_brain0.1_l0.1 --train_csv "E:\\Datathon\\LISA\results\\preprocessed_data\\df_test_imgs.csv" --test_csv "E:\\Datathon\\LISA\results\\preprocessed_data\\df_test_imgs.csv"
    """
