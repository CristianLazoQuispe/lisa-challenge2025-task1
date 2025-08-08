import pandas as pd
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import traceback
import argparse
import atexit
import torch
import wandb
import time
import sys
import gc
import os

sys.path.append("../")

from src.dataset2D import MRIDataset2D
from src.dataset3D import MRIDataset3D
from src.models2D import Model2DTimm
from src.models3D import Model3DResnet
from src.mywandb import init_wandb
from src import utils
from src import trainer
from src import filtering
from src import aggregations_preds
from src.inference_model import inference_model
from src.ensemble import ensemble_probs_from_files
from torchvision import transforms


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
    parser.add_argument("--train_original_csv", type=str, default="../results/preprocessed_data/df_train.csv")
    parser.add_argument("--train_csv", type=str, default="../results/preprocessed_data/df_train_imgs.csv")
    parser.add_argument("--test_csv", type=str, default="../results/preprocessed_data/df_test_imgs.csv")
    parser.add_argument("--save_dir", type=str, default="./results/")
    parser.add_argument("--base_model", type=str, default="vit_tiny_patch16_224.augreg_in21k")
    
    parser.add_argument("--threshold_brain_presence", type=float, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--slice_frac", type=float, default=0.3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    
    parser.add_argument("--is_numpy", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    
    
    parser.add_argument("--label_cols", type=str, default="Noise,Zipper,Positioning,Banding,Motion,Contrast,Distortion")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout_p", type=float, default=0.3)

    parser.add_argument("--weight_method", type=str, default="effective",
                        choices=["effective", "invfreq","manual"],help="Método para calcular class weights: 'effective' (CB-Focal) o 'invfreq'")
    parser.add_argument("--weight_beta", type=float, default=0.99, help="Parámetro beta para método 'effective'")
    parser.add_argument("--weight_alpha", type=float, default=0.5, help="Parámetro alpha para método 'invfreq'")
    parser.add_argument("--weight_cap", type=float, default=8.0,help="Cap máximo para pesos en método 'invfreq'")

    parser.add_argument("--mixup_prob", type=float, default=1.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--cutmix_prob", type=float, default=0.3)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)

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
    df_train_original = pd.read_csv(args.train_original_csv)

    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)
    df_train["patient_id"] = df_train["filename"].str.extract(r"(LISA_\d+)")
    df_test["patient_id"]  = df_test["filename"].str.extract(r"(LISA_VALIDATION_\d+)")
    
    df_train = df_train[df_train["ratio"]>=args.threshold_brain_presence].reset_index()
    df_test  = df_test[df_test["ratio"]>=args.threshold_brain_presence].reset_index()
    
    print(f"Original: {df_train.shape}")
    df_train = filtering.filter_dataset_by_similarity_ssim(df_train, ssim_thresh=0.85)
    print(f"→ Filtrado: {df_train.shape}")

    #raise

    # 🎯 Clases
    args.label_cols = args.label_cols.split(",")

    df_train,df_test_back = utils.robust_split_by_patient(df_train_original,df_train,args)

    results = {}
    if False:
        # 🚀 Entrenar y evaluar
        print("df_test_back:",df_test_back.shape)
        results = trainer.train_with_cv(df_train, df_test_back, args.label_cols, args,save_dir = args.save_dir)

        print(f"\n✅ Done. CV F1 Macro: {results['test_oof_f1_macro']:.4f}")
        print(f"🧪 OOF saved at: {results['oof_path']}")

    if args.type_modeling == "2d":
        test_back_ds = MRIDataset2D(df=df_test_back,is_train=False,use_augmentation=False,is_numpy=bool(args.is_numpy),labels=args.label_cols,image_size=args.image_size)
        test_ds = MRIDataset2D(df=df_test,is_train=False,use_augmentation=False,is_numpy=bool(args.is_numpy),labels=args.label_cols,image_size=args.image_size)
    else:
        test_ds = MRIDataset3D(df=df_test,is_train=False,use_augmentation=False,labels=args.label_cols)
        
    if args.type_modeling == "2d":
        model = Model2DTimm(base_model=args.base_model,in_channels=args.in_channels,num_labels=len(args.label_cols),
                                      num_classes=3,dropout_p=args.dropout_p).to(args.device)
    else:
        model = Model3DResnet(num_labels=len(args.label_cols), num_classes=3,dropout_p=args.dropout_p,
                                      pretrained=True, freeze_backbone=False).to(args.device) 

    method = "probs"
    maxi_f1_macro = 0
    better_thr = None
    for method in ["probs","label"]:
        if method == "probs":
            list_thr = [[0.10,0.30]]
        
        else:
            list_thr = [[0.01,0.50],[0.10,0.50],[0.2,0.50],[0.3,0.50],[0.4,0.50],[0.5,0.50],[0.5,0.60],[0.5,0.70]] #[0.4,0.50]
            list_thr = [[0.01,0.2],[0.10,0.2],[0.2,0.2],[0.3,0.2],[0.4,0.2],[0.5,0.2],[0.5,0.7],
                        [0.01,0.3],[0.10,0.3],[0.2,0.3],[0.3,0.3],[0.4,0.3],[0.5,0.3],[0.5,0.8],
                        [0.01,0.4],[0.10,0.4],[0.2,0.4],[0.3,0.4],[0.4,0.4],[0.5,0.4],[0.5,0.9],
                        [0.01,0.6],[0.10,0.6],[0.2,0.6],[0.3,0.6],[0.4,0.6],[0.5,0.6],[0.5,1.0]] #[0.4,0.3]
            list_thr = [[0.4,0.3]]
        for thr2,thr1 in list_thr:
            print("*"*10)
            print(method, " thr2,thr1", thr2,thr1)
            csvs = []
            for fold in range(args.n_splits):
                model.load_state_dict(torch.load(f"{args.save_dir}/model_fold{fold}.pt"))
                df_probs = inference_model(model, test_back_ds, label_cols=args.label_cols, device=args.device,args=args,method=method,thr2=thr2, thr1=thr1)
                df_probs.to_csv(f"{args.save_dir}/test_back_probs_fold{fold}.csv", index=False)
                print("df_probs:",df_probs.shape)
                # Ensemble
                csvs.append(f"{args.save_dir}/test_back_probs_fold{fold}.csv")
            if method == "probs":
                df_probs, df_preds = ensemble_probs_from_files(csvs, args.label_cols, save_preds_path=f"{args.save_dir}/ensemble_test_back_submission.csv")
            else:
                df_probs, df_preds = aggregations_preds.ensemble_mode_and_mean_probs(csvs,  args.label_cols, 
                                                                                save_preds_path=f"{args.save_dir}/ensemble_test_back_submission.csv",thr2=thr2, thr1=thr1)

            # Asegurar que el orden de columnas sea igual
            df_test_back_unique = df_test_back.drop_duplicates(subset=["filename"]).reset_index(drop=True)
            y_true = df_test_back_unique[args.label_cols].values
            y_pred = df_preds[args.label_cols].values
            print(y_true.shape)
            print(y_pred.shape)
            f1_per_label_dict = {}
            for i, col in enumerate(args.label_cols):
                f1_per_label_dict[col] = f1_score(y_true[:, i], y_pred[:, i], average='macro')

            f1_macro = sum(f1_per_label_dict.values()) / len(f1_per_label_dict)
            f1_micro = f1_score(y_true.ravel(), y_pred.ravel(), average='micro')

            print("F1 por etiqueta:", f1_per_label_dict)
            print("F1 macro:", f1_macro)
            print("F1 micro:", f1_micro)
            if f1_macro>maxi_f1_macro:
                maxi_f1_macro = f1_macro
                better_thr = [thr2,thr1, f1_macro,f1_micro,f1_per_label_dict]

    print("better_thr:",better_thr)
    csvs = []
    for fold in range(args.n_splits):
        model.load_state_dict(torch.load(f"{args.save_dir}/model_fold{fold}.pt"))
        df_probs = inference_model(model, test_ds, label_cols=args.label_cols, device=args.device,args=args,method=method,thr2=thr2, thr1=thr1)
        df_probs.to_csv(f"{args.save_dir}/test_probs_fold{fold}.csv", index=False)

        # Ensemble
        csvs.append(f"{args.save_dir}/test_probs_fold{fold}.csv")
    if method == "probs":
        df_probs, df_preds = ensemble_probs_from_files(csvs, args.label_cols, save_preds_path=f"{args.save_dir}/ensemble_submission.csv")
    else:
        df_probs, df_preds = aggregations_preds.ensemble_mode_and_mean_probs(csvs,  args.label_cols, 
                                                                        save_preds_path=f"{args.save_dir}/ensemble_submission.csv",thr2=thr2, thr1=thr1)

    run = init_wandb(args,f"{args.experiment_name}")
    log_values = results
    log_values['submission_final_prods'] =  wandb.Table(dataframe=df_probs)
    log_values['submission_final_preds'] =  wandb.Table(dataframe=df_preds)
    wandb.log(log_values,step=0)

    finish_process()



    """
    maxvit_tiny_tf_512.in1k
    maxvit_nano_rw_256.sw_in1k
    maxvit_tiny_rw_224.sw_in1k


    python 4.training2d.py \
    --device cuda:4 \
    --save_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/maxvit_nano_rw_256_flv3_8_0.1 \
    --experiment_name maxvit_nano_rw_256_flv3_8_0.1 \
    --loss_name focal_lossv3 \
    --slice_frac 0.1 \
    --threshold_brain_presence 0.2 \
    --batch_size 8 \
    --in_channels 1 \
    --is_numpy 1 \
    --image_size 256 \
    --base_model maxvit_nano_rw_256.sw_in1k \
    --focal_gamma 1.5 \
    --weight_method manual \
    --weight_beta 0.9 \
    --label_smoothing 0.05 \
    --epochs 2000 \
    --patience 20 \
    --type_modeling 2d \
    --lr 1e-5 \
    --weight_decay 1e-3 \
    --use_sampling 0 \
    --num_workers 8 \
    --n_splits 3

    
    """