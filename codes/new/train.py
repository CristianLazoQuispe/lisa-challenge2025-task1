"""
Entry point for training and evaluating models on the LISA 2025 challenge.

This script orchestrates data loading, preprocessing, cross‑validation
training, aggregator evaluation and optional test inference.  It reads
configuration parameters from the command line and environment variables.

Environment configuration
------------------------

The following environment variables are used to configure WandB:

* ``WANDB_API_KEY``: your API key for Weights & Biases.
* ``PROJECT_WANDB``: the project name on WandB.
* ``ENTITY``: your WandB entity or username.

These can be stored in a ``.env`` file in the working directory and
loaded automatically at runtime.  See the included ``train.py`` for
details.

Usage example:

    python train.py \
        --train_csv /path/to/train.csv \
        --test_back_csv /path/to/test_back.csv \
        --save_dir ./models \
        --n_splits 5 \
        --epochs 30 \
        --batch_size 16 \
        --lr 1e-4 \
        --base_model maxvit_tiny_tf_512.in1k \
        --head_type label_tokens \
        --use_view \
        --aggregators mean vote max

If ``test_csv`` (final test set without labels) is provided via
``--test_csv``, the script will perform inference on that dataset using
the specified aggregator (default: ``mean``) and write a submission CSV
with probabilities and predicted labels.
"""

from __future__ import annotations

import os
import argparse
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

from utils import set_seed, LABELS
from trainer import train_and_evaluate
from filtering import filter_dataset_by_similarity_ssim
from sklearn.model_selection import train_test_split
import os
import wandb


os.environ["WANDB_DIR"] = "/data/cristian/paper_2025/wandb_dir"  # Aquí se guardarán los archivos temporales y logs
os.environ["WANDB_CACHE_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_ARTIFACT_DIR"] = "/data/cristian/paper_2025/wandb_dir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate LISA models with cross validation.")
    parser.add_argument('--top_k', type=int, default=2, help='Número de etiquetas raras para la estratificación')
    parser.add_argument("--train_csv", type=str, default="../../results/preprocessed_data/df_train_imgs.csv")
    parser.add_argument("--test_csv", type=str, default="../../results/preprocessed_data/df_test_imgs.csv")
    parser.add_argument('--test_back_csv', type=str, default=None, help='Path to test_back CSV with labels')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models and outputs')
    parser.add_argument('--experiment_name', type=str, default='lisa_experiment', help='Name of the experiment for WandB')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross‑validation splits')
    parser.add_argument('--threshold_brain_presence', type=float, default=0.3, help='threshold for filtering slices')
    parser.add_argument('--dynamic_w', type=str, default="0.25, 0.75, 1.0", help='threshold for filtering slices') 
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum number of epochs per fold')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs without improvement before stopping')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_aux', type=float, default=0.2, help='lambda_aux for centroid of brain')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--base_model', type=str, default='maxvit_tiny_tf_512.in1k', help='timm model name')
    parser.add_argument('--head_type', type=str, default='label_tokens', choices=['simple', 'label_tokens'], help='Type of classification head')
    parser.add_argument('--use_view', action='store_true', help='Whether to incorporate view information')
    parser.add_argument('--pretrained', type=int, default=1, help='Use pretrained backbone (1=yes, 0=no)')
    parser.add_argument('--image_size', type=int, default=224, help='Image size (square) for resizing slices')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--slice_frac', type=float, default=1.0, help='Fraction of training slices used per epoch')
    parser.add_argument('--use_sampling', type=int, default=1, help='Use WeightedRandomSampler (1=yes, 0=no)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes')
    parser.add_argument('--device', type=str, default=None, help='CUDA/CPU device (e.g. cuda:0)')
    parser.add_argument('--weight_method', type=str, default='effective', choices=['effective', 'invfreq'], help='Method for class weights')
    parser.add_argument('--weight_beta', type=float, default=0.99, help='Beta for effective number class weights')
    parser.add_argument('--weight_alpha', type=float, default=0.5, help='Alpha for inverse frequency class weights')
    parser.add_argument('--weight_cap', type=float, default=8.0, help='Max cap for inverse frequency class weights')
    parser.add_argument('--head_config', type=str, default=None, help='JSON string for head configuration')
    parser.add_argument('--aggregators', nargs='+', default=['mean'], help='Aggregation strategies to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for LR scheduler')
    parser.add_argument('--log_interval', type=int, default=50, help='Steps between logging to WandB')
    parser.add_argument('--do_inference', action='store_true', help='Perform inference on test set after training')
    parser.add_argument('--split_by_volume', action='store_true', help='Use volume-level stratified folds instead of patient-level')
    parser.add_argument('--volume_id', type=str, default='patient_id', help='Column name to identify volumes when splitting by volume')
    parser.add_argument('--norm_mode', type=str, default='slice_z', choices=['none', 'slice_z', 'dataset_z_per_view'],
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
        return '_'.join(parts[:2])
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
    load_dotenv()
    args = parse_args()
    # Convert head_config JSON string to dict
    head_config = None
    if args.head_config:
        import json
        head_config = json.loads(args.head_config)
    # Compose args dict for trainer
    args_dict = vars(args).copy()
    args_dict['head_config'] = head_config
    # Set seed
    set_seed(args.seed)
    # Load train data
    train_df = pd.read_csv(args.train_csv)
    train_df = preprocess_df(train_df, LABELS)

    print(f"Original: {train_df.shape}")
    train_df = train_df[train_df["ratio"]>=args.threshold_brain_presence].reset_index()
    
    print(f"Original: {train_df.shape}")
    #train_df = filter_dataset_by_similarity_ssim(train_df, ssim_thresh=0.6)
    print(f"→ Filtrado: {train_df.shape}")

    # Load test_back if provided
    test_back_df = None
    if args.test_back_csv:
        test_back_df = pd.read_csv(args.test_back_csv)
        test_back_df = preprocess_df(test_back_df, LABELS)
    else:
        print("Doing TEST BACK")
        df_all = train_df.copy()
        df_all["patient_id"] = df_all["filename"].str.extract(r"(LISA_\d+)")
        train_ids, test_ids = train_test_split(df_all["patient_id"].unique(), test_size=0.1, random_state=42, shuffle=True)
        train_df     = df_all[df_all["patient_id"].isin(train_ids)].reset_index(drop=True)
        test_back_df = df_all[df_all["patient_id"].isin(test_ids)].reset_index(drop=True)
        test_back_df = preprocess_df(test_back_df, LABELS)

    # Call training
    #if not(args.do_inference and args.test_csv):
    results = train_and_evaluate(
        train_df=train_df,
        test_back_df=test_back_df,
        label_cols=LABELS,
        args=args_dict,
        aggregators=args.aggregators
    )
    # Optionally run inference on final test set
    if args.do_inference and args.test_csv:
        """Realiza inferencia en el conjunto de test sin etiquetas y guarda
        resultados por agregador.  Por cada agregador indicado en
        ``args.aggregators``, promedia las probabilidades de todos los
        modelos de los folds y calcula la clase final por etiqueta.  Se
        escriben CSVs separados para cada agregador con el sufijo
        ``submission_<aggregator>.csv`` en el directorio de ``save_dir``.

        El CSV contiene la columna ``filename`` (identificador del
        volumen), la predicción de cada etiqueta y la probabilidad de
        cada clase (``<label>_0``, ``<label>_1``, ``<label>_2``).
        """
        # Import inference utilities lazily to avoid circular import
        from dataset import MRIDataset2D
        from model import Model2DTimm
        from torch.utils.data import DataLoader
        import numpy as np
        import torch
        # Load test dataframe
        test_df = pd.read_csv(args.test_csv)
        print(f"df_test Original: {test_df.shape}")
        test_df  = test_df[test_df["ratio"]>=args.threshold_brain_presence].reset_index()
        print(f"df_test Original: {test_df.shape}")
        #test_df = filter_dataset_by_similarity_ssim(test_df, ssim_thresh=0.6)
        print(f"df_test→ Filtrado: {test_df.shape}")
        # We expect no labels; still need patient_id
        test_df['patient_id'] = test_df['filename'].apply(extract_patient_id)
        # Determine device
        device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Import aggregator utilities
        from trainer import collect_slice_predictions, aggregate_slices
        import json
        # For each fold/model, collect slice-level predictions on a dataset normalised with the fold's per-view stats
        slice_results_per_model: List[List[dict]] = []
        vol_names: Optional[List[str]] = None
        for fold in range(args.n_splits):
            # Load model for this fold
            model_path = os.path.join(args.save_dir, f"model_fold{fold}.pt")
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
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            # Load per-view stats for this fold if using dataset_z_per_view
            per_view_stats_fold = None
            if args.norm_mode == 'dataset_z_per_view':
                print("Using dataset_z_per_view")
                stats_file = os.path.join(args.save_dir, f"per_view_stats_fold{fold}.json")
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
            fold_results = collect_slice_predictions(test_loader_fold, model, device)
            slice_results_per_model.append(fold_results)
            # Determine volume names using mean aggregator once
            if vol_names is None:
                _, _, _, names = aggregate_slices(fold_results, num_labels=len(LABELS), num_classes=3, aggregator='mean')
                vol_names = names
        # For each aggregator, compute aggregated probabilities across models and average
        for agg in args.aggregators:
            all_probs_per_model = []
            for fold_results in slice_results_per_model:
                _, y_prob, _, _ = aggregate_slices(fold_results, num_labels=len(LABELS), num_classes=3, aggregator=agg)
                all_probs_per_model.append(y_prob)
            # Average probabilities across models (folds)
            mean_probs = np.mean(np.stack(all_probs_per_model, axis=0), axis=0)  # (N,L,C)
            y_pred_final = np.argmax(mean_probs, axis=-1)  # (N,L)
            # Construct submission DataFrame
            submission = pd.DataFrame({'filename': vol_names})
            for i, lbl in enumerate(LABELS):
                submission[lbl] = y_pred_final[:, i]
                for cls in range(mean_probs.shape[-1]):
                    submission[f'{lbl}_{cls}'] = mean_probs[:, i, cls]
            # Write predictions and probabilities to CSV
            suffix = agg
            sub_path = os.path.join(args.save_dir, f'submission_{suffix}_probs.csv')
            submission.to_csv(sub_path, index=False)
            print(f"Saved submission for aggregator '{agg}' to {sub_path}")
            sub_path = os.path.join(args.save_dir, f'submission_{suffix}_preds.csv')
            submission[["filename"]+LABELS].to_csv(sub_path, index=False)
            print(f"Saved submission for aggregator '{agg}' to {sub_path}")


if __name__ == '__main__':
    main()

    """
    
    python train.py \
    --save_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/lisa_clean_simple \
    --experiment_name lisa_clean_simple \
    --device cuda:5 \
    --epochs 500 \
    --patience 20 \
    --image_size 256 \
    --batch_size 32 \
    --lr 1e-5 \
    --base_model maxvit_nano_rw_256.sw_in1k \
    --head_type simple \
    --use_view \
    --aggregators mean vote max weighted \
    --volume_id patient_id \
    --n_splits 5 \
    --top_k 0

    1 0.25 0.5 0.75
    2 0.25 0.5 1.0
    3 0.25,0.25,1.0
    4 0.25,0.5,0.5
    5 0.1,0.25,0.75
    6 0.25,0.75,1.0

    python train.py \
    --save_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/lisa_clean_label_tokens_testback6_z \
    --experiment_name lisa_clean_label_tokens_testback6_z \
    --norm_mode dataset_z_per_view \
    --device cuda:5 \
    --slice_frac  0.99 \
    --dynamic_w 0.25,0.75,1.0 \
    --epochs 5000 \
    --patience 100 \
    --image_size 256 \
    --batch_size 32 \
    --lr 1e-5 \
    --base_model maxvit_nano_rw_256.sw_in1k \
    --head_type label_tokens \
    --use_view \
    --aggregators mean vote max weighted \
    --volume_id patient_id \
    --n_splits 5 \
    --top_k 2 \
    --do_inference
    
    
    python train.py     --save_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/lisa_clean_label_tokens_testback2_z_newaug2_pos+bbox_brain.0.02     --experiment_name lisa_clean_label_tokens_testback2_z_newaug2_pos+bbox_brain.0.02     --norm_mode dataset_z_per_view     --device cuda:1     --slice_frac  0.99     --dynamic_w 0.25,0.5,1.0     --epochs 5000     --patience 100     --image_size 256     --batch_size 32     --lr 1e-5     --base_model maxvit_nano_rw_256.sw_in1k     --head_type label_tokens     --use_view     --aggregators mean vote max weighted     --volume_id patient_id     --n_splits 5     --top_k 2 --threshold_brain_presence 0.02  --lambda_aux 0.2  --do_inference
    """