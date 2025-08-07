import os
import argparse
import pandas as pd
import torch
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from assign_robust_folds import assign_robust_folds
from train_with_robust_csv import train_with_cv
from inference_model import inference_model
from ensemble import ensemble_probs_from_files
from dataset import MRI2DOrdinalDataset,MRI3DOrdinalDataset
from torchvision import transforms
from models import SwinOrdinalClassifier
from sklearn.model_selection import train_test_split
import wandb
from my_wandb import init_wandb
import numpy as np

import os
import sys
parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_path)

from src.models.video3d18 import Videor3d18Classifier


import time
import gc
import traceback
import atexit
import torch
import numpy as np
import random

# Torch Settings
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = True

# Global variables
is_finished = False

def finish_process():
    global is_finished
    """
    function to finish wandb if there is an error in the code or force stop
    """
    if is_finished:
        return None
    print("Finishing process")
    try:
        time.sleep(2)
        print("Closing wandb.. ")
        wandb.finish()
        print("Waiting for closing wandb.. ")
        time.sleep(10)
        print("Wandb closed")
    except:
        print(traceback.format_exc())
    print("Cleaning memory.. ")
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleaned")
    is_finished = True

atexit.register(finish_process)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--experiment_name", type=str, default="2d_prediction")
    parser.add_argument("--tag_wandb", type=str, default="2d")
    parser.add_argument("--type_modeling", type=str, default="2d or 3d")    
    parser.add_argument("--train_csv", type=str, default="../../results/preprocessed_data/df_train_imagesNEW4x4_nativev2.csv")
    parser.add_argument("--test_csv", type=str, default="../../results/preprocessed_data/df_test_imagesNEW4x4_nativev2.csv")
    parser.add_argument("--save_dir", type=str, default="./results/")
    parser.add_argument("--base_model", type=str, default="vit_tiny_patch16_224.augreg_in21k")
    
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--is_numpy", type=int, default=0)
    
    parser.add_argument("--label_cols", type=str, default="Noise,Zipper,Positioning,Banding,Motion,Contrast,Distortion")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.3)
    parser.add_argument("--mixup_prob", type=float, default=0.5)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_sampling", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:5" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--loss_name", type=str, default="focal_loss",help="focal_loss,cross_simple,cross_weighted")
    parser.add_argument("--use_manual_weights", type=int, default=1)

    return parser.parse_args()



if __name__ == "__main__":
    atexit.register(finish_process)
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 📄 Leer CSVs
    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)

    # 🎯 Clases
    args.label_cols = args.label_cols.split(",")


    # Back testing
    df_all = df_train.copy()
    df_all["patient_id"] = df_all["filename"].str.extract(r"(LISA_\d+)")
    train_ids, test_ids = train_test_split(df_all["patient_id"].unique(), test_size=0.1, random_state=args.seed, shuffle=True)
    df_train      = df_all[df_all["patient_id"].isin(train_ids)].reset_index(drop=True)
    df_test_back  = df_all[df_all["patient_id"].isin(test_ids)].reset_index(drop=True)
    
    # 🧠 Asignar folds robustos
    df_train = assign_robust_folds(df_train, n_splits=args.n_splits, top_k=2, seed=42)

    for i, label in enumerate(args.label_cols):
        y = df_train[label].values
        print(f"Train    : {label} true dist: {np.bincount(y, minlength=3)}")
    for i, label in enumerate(args.label_cols):
        y = df_test_back[label].values
        print(f"Testback : {label} true dist: {np.bincount(y, minlength=3)}")

    results = {}
    if True:
        # 🚀 Entrenar y evaluar
        print("df_test_back:",df_test_back.shape)
        results = train_with_cv(df_train, df_test_back, args.label_cols, args,save_dir = args.save_dir)

        print(f"\n✅ Done. CV F1 Macro: {results['test_oof_f1_macro']:.4f}")
        print(f"🧪 OOF saved at: {results['oof_path']}")

    if args.type_modeling == "2d":
        test_ds = MRI2DOrdinalDataset(df_test, is_train=False, transform=val_tf,is_numpy=args.is_numpy,labels=args.label_cols)
    else:
        test_ds = MRI3DOrdinalDataset(df_test, is_train=False, transform=val_tf)
        
    if args.type_modeling == "2d":
        model = SwinOrdinalClassifier(base_model=args.base_model,in_channels=args.in_channels,num_labels=len(args.label_cols),
                                      num_classes=3,dropout_p=args.dropout_p).to(args.device)
    else:
        model = Videor3d18Classifier(num_labels=len(args.label_cols), num_classes=3,dropout_p=args.dropout_p,
                                      pretrained=True, freeze_backbone=False).to(args.device) 

    csvs = []
    for fold in range(args.n_splits):
        model.load_state_dict(torch.load(f"{args.save_dir}/model_fold{fold}.pt"))
        df_probs = inference_model(model, test_ds, label_cols=args.label_cols, device=args.device)
        df_probs.to_csv(f"{args.save_dir}/test_probs_fold{fold}.csv", index=False)

        # Ensemble
        csvs.append(f"{args.save_dir}/test_probs_fold{fold}.csv")
    df_probs, df_preds = ensemble_probs_from_files(csvs, args.label_cols, save_preds_path=f"{args.save_dir}/ensemble_submission.csv")

    run = init_wandb(args,f"{args.experiment_name}")
    log_values = results
    log_values['submission_final_prods'] =  wandb.Table(dataframe=df_probs)
    log_values['submission_final_preds'] =  wandb.Table(dataframe=df_preds)
    wandb.log(log_values,step=0)

    finish_process()

