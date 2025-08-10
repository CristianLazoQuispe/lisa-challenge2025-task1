# train_infer_lisa.py
import os, json, random, time, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

import sys
import gc
import os

sys.path.append("../")
# 👇 Usa tus clases ya creadas
from src.models3D import Model3DResnet
from src.dataset3D import MRIDataset3D

import os
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="/home/va0831/Projects/lisa-challenge2025/.env")



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_DIR"] = "/data/cristian/paper_2025/wandb_dir"  # Aquí se guardarán los archivos temporales y logs
os.environ["WANDB_CACHE_DIR"] = "/data/cristian/paper_2025/wandb_dir"
os.environ["WANDB_ARTIFACT_DIR"] = "/data/cristian/paper_2025/wandb_dir"



LABELS = ["Noise","Zipper","Positioning","Banding","Motion","Contrast","Distortion"]

# ------------------------------
# Utilidades
# ------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def make_strata(df):
    # Estrato simple: suma de severidades (0..14) + vista (axi/cor/sag) para que no concentre vistas
    # Puedes sofisticarlo luego (e.g., hashing multietiqueta)
    s = df[LABELS].sum(axis=1).clip(0, 14).astype(int).astype(str)
    v = df["view"].astype(str)
    return (s + "_" + v).values

def one_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses = []
    all_preds, all_tgts = [], []

    for (imgs, tgts, fnames, view_onehot) in tqdm(loader, desc="train" if is_train else "valid", leave=False):
        imgs = imgs.to(device)             # (B,1,D,H,W)
        view_onehot = view_onehot.to(device)  # (B,3)
        if tgts is not None and isinstance(tgts, torch.Tensor):
            tgts = tgts.to(device)         # (B,7)

        with torch.set_grad_enabled(is_train):
            logits = model(imgs, view_onehot)   # (B,7,3)

            if tgts is not None and tgts.ndim == 2:
                # CE por etiqueta, promedio en 7 labels
                loss = 0.0
                for l in range(len(LABELS)):
                    loss += criterion(logits[:, l, :], tgts[:, l])
                loss = loss / len(LABELS)
            else:
                loss = torch.tensor(0.0, device=device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        losses.append(loss.detach().item() if isinstance(loss, torch.Tensor) else 0.0)

        # Métricas
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()    # (B,7)
            all_preds.append(preds)
            if tgts is not None and isinstance(tgts, torch.Tensor):
                all_tgts.append(tgts.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0) if len(all_preds) else None
    if len(all_tgts):
        all_tgts = np.concatenate(all_tgts, axis=0)
        # F1 macro “plano” (todas las etiquetas como una sola lista)
        f1m = f1_score(all_tgts.reshape(-1), all_preds.reshape(-1), average="macro")
        acc = accuracy_score(all_tgts.reshape(-1), all_preds.reshape(-1))
    else:
        f1m, acc = None, None

    return np.mean(losses), f1m, acc, all_preds

def evaluate(model, loader, device):
    model.eval()
    file_names = []
    view_list = []
    preds_all = []

    with torch.no_grad():
        for (imgs, _, fnames, view_onehot) in tqdm(loader, desc="infer", leave=False):
            imgs = imgs.to(device)
            view_onehot = view_onehot.to(device)
            logits = model(imgs, view_onehot)  # (B,7,3)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds_all.append(preds)
            file_names += list(fnames)
    preds_all = np.concatenate(preds_all, axis=0)
    return file_names, preds_all

# ------------------------------
# Dataloaders
# ------------------------------
def build_loaders(df_tr, df_va, args):
    ds_tr = MRIDataset3D(
        df_tr, is_train=True, use_augmentation=args.use_aug,
        spatial_size=tuple(args.spatial_size), labels=LABELS
    )
    ds_va = MRIDataset3D(
        df_va, is_train=True, use_augmentation=False,
        spatial_size=tuple(args.spatial_size), labels=LABELS
    )
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return dl_tr, dl_va

def build_test_loader(df_te, args):
    ds_te = MRIDataset3D(
        df_te, is_train=False, use_augmentation=False,
        spatial_size=tuple(args.spatial_size), labels=LABELS
    )
    return DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

# ------------------------------
# Entrenamiento por fold
# ------------------------------
def train_one_fold(fold, df, tr_idx, va_idx, args, device):
    # W&B run por fold
    run = None
    if args.use_wandb:
        import wandb
        exp_name = args.experiment_name
        run = wandb.init(
            project=os.getenv("PROJECT_WANDB"),
            entity=os.getenv("ENTITY"),
            name=f"{exp_name}_fold{fold}",
            group=exp_name,
            config=vars(args),
            reinit=True,
        )

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[va_idx].reset_index(drop=True)
    dl_tr, dl_va = build_loaders(df_tr, df_va, args)

    model = Model3DResnet(
        in_channels=1, num_labels=len(LABELS), num_classes=3,
        pretrained=True, freeze_backbone=True, dropout_p=0.3, freeze_n=2
    ).to(device)

    # CE con label smoothing suave para combatir overfitting temprano
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience, verbose=False)

    best_val = (1e9, -1.0)  # (val_loss, f1)
    best_path = os.path.join(args.out_dir, f"{args.experiment_name}_fold{fold}.pth")
    patience_cnt = 0

    oof_preds = np.zeros((len(df_va), len(LABELS)), dtype=int)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_f1, tr_acc, _ = one_epoch(model, dl_tr, device, criterion, optimizer)
        va_loss, va_f1, va_acc, va_preds = one_epoch(model, dl_va, device, criterion, optimizer=None)

        # guarda OOF actuales para monitoreo (no es estricto, pero útil)
        if va_preds is not None:
            oof_preds[:va_preds.shape[0]] = va_preds

        # Scheduler
        scheduler.step(va_loss)

        # Log
        if args.use_wandb and run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss, "train/f1_macro": tr_f1, "train/acc": tr_acc,
                "valid/loss": va_loss, "valid/f1_macro": va_f1, "valid/acc": va_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Early stopping por val_loss (estable) con tie-break por F1
        cur = (va_loss, -(va_f1 if va_f1 is not None else 0.0))
        if cur < best_val:
            best_val = cur
            patience_cnt = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= args.es_patience:
                break

    # Carga best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # Métrica final en valid
    _, va_f1, va_acc, va_preds = one_epoch(model, dl_va, device, criterion=None, optimizer=None)

    # OOF dataframe
    df_oof = df_va.copy()
    for i, lab in enumerate(LABELS):
        df_oof[f"pred_{lab}"] = va_preds[:, i]
    return model, df_oof, (va_f1, va_acc), best_path

# ------------------------------
# Sanity split (back-testing) opcional
# ------------------------------
def quick_sanity_check(df, args, device):
    # Split agrupado por patient y estratificado por suma de labels
    strata = make_strata(df)
    patients = df["patient_id"]
    tr_idx, va_idx = next(GroupKFold(n_splits=5).split(df, y=strata, groups=patients))  # solo 1/5 como sanity
    df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
    dl_tr, dl_va = build_loaders(df_tr, df_va, args)

    model = Model3DResnet(
        in_channels=1, num_labels=len(LABELS), num_classes=3,
        pretrained=True, freeze_backbone=True, dropout_p=0.3, freeze_n=2
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # Una sola época rápida para detectar bugs de shapes, etc.
    one_epoch(model, dl_tr, device, criterion, optimizer)
    one_epoch(model, dl_va, device, criterion, optimizer=None)
    del model
    torch.cuda.empty_cache()

# ------------------------------
# Main
# ------------------------------
def main(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Lee CSVs
    df_train = pd.read_csv(args.train_csv)
    df_test  = pd.read_csv(args.test_csv)

    # Asegura patient_id
    if "patient_id" not in df_train.columns:
        df_train["patient_id"] = df_train["filename"].str.extract(r"(LISA_\d+)")
    if "patient_id" not in df_test.columns:
        # OJO: en validation suele venir "LISA_VALIDATION_####", ajusta si es necesario
        df_test["patient_id"]  = df_test["filename"].str.extract(r"(LISA_VALIDATION_\d+)")

    # Vista si no está
    if "view" not in df_train.columns:
        df_train["view"] = df_train["path"].str.extract(r"_LF_(\w+)\.nii")
    if "view" not in df_test.columns:
        df_test["view"] = df_test["path"].str.extract(r"_LF_(\w+)\.nii")

    # Sanity check rápido
    if args.run_sanity:
        print(">>> Running quick sanity check (1 fold slice)...")
        quick_sanity_check(df_train, args, device=args.device)

    # 5 folds agrupados por patient, estrato simple por distribución de labels+vista
    gkf = GroupKFold(n_splits=args.folds)
    strata = make_strata(df_train)
    groups = df_train["patient_id"]

    # OOF global
    all_oof = []
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df_train, y=strata, groups=groups)):
        print(f"\n========== Fold {fold} ==========")
        model, df_oof, (va_f1, va_acc), best_path = train_one_fold(
            fold, df_train, tr_idx, va_idx, args, device=args.device
        )
        fold_metrics.append({"fold": fold, "f1_macro": va_f1, "acc": va_acc, "best_path": best_path})
        all_oof.append(df_oof)

        # Limpieza para ahorrar VRAM
        del model
        torch.cuda.empty_cache()

    df_oof_all = pd.concat(all_oof, axis=0).reset_index(drop=True)
    df_oof_all.to_csv(os.path.join(args.out_dir, f"{args.experiment_name}_oof.csv"), index=False)

    # OFF-CV (macro F1 y acc)
    y_true = df_oof_all[LABELS].values.reshape(-1)
    y_pred = df_oof_all[[f"pred_{l}" for l in LABELS]].values.reshape(-1)
    off_f1 = f1_score(y_true, y_pred, average="macro")
    off_acc = accuracy_score(y_true, y_pred)
    print(f"OOF: F1_macro={off_f1:.4f}, acc={off_acc:.4f}")

    # Log final a W&B (opcional)
    if args.use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("PROJECT_WANDB"),
            entity=os.getenv("ENTITY"),
            name=f"{args.experiment_name}_summary",
            group=args.experiment_name,
            config=vars(args),
            reinit=True,
        )
        wandb.log({"oof/f1_macro": off_f1, "oof/acc": off_acc})
        wandb.finish()

    # Inference en test (promediar folds es fácil si quieres, aquí uso el MEJOR de cada fold con voto mayoritario simple)
    print("\n>>> Inference on test set")
    test_loader = build_test_loader(df_test, args)
    # Carga todos los modelos y promedia logits → mejor que votar argmax
    logits_sum = []
    for fm in fold_metrics:
        ckpt = torch.load(fm["best_path"], map_location=args.device)
        model = Model3DResnet(
            in_channels=1, num_labels=len(LABELS), num_classes=3,
            pretrained=False, freeze_backbone=False, dropout_p=0.3, freeze_n=0
        ).to(args.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Acumula logits
        fold_logits = []
        with torch.no_grad():
            for (imgs, _, fnames, view_onehot) in tqdm(test_loader, desc=f"infer_fold{fm['fold']}", leave=False):
                imgs = imgs.to(args.device); view_onehot = view_onehot.to(args.device)
                out = model(imgs, view_onehot)  # (B,7,3)
                fold_logits.append(out.cpu().numpy())
        fold_logits = np.concatenate(fold_logits, axis=0)   # (N,7,3)
        logits_sum.append(fold_logits)

        del model
        torch.cuda.empty_cache()

    # Promedio de logits y argmax final
    logits_mean = np.mean(np.stack(logits_sum, axis=0), axis=0)  # (N,7,3)
    preds_test = np.argmax(logits_mean, axis=-1)                 # (N,7)

    sub = df_test[["filename"]].copy()
    for i, lab in enumerate(LABELS):
        sub[lab] = preds_test[:, i]
    sub_path = os.path.join(args.out_dir, f"{args.experiment_name}_submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to: {sub_path}")

    # Guarda métricas de folds
    pd.DataFrame(fold_metrics).to_csv(os.path.join(args.out_dir, f"{args.experiment_name}_fold_metrics.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="../results/preprocessed_data/df_train_imgs.csv")
    parser.add_argument("--test_csv", type=str, default="../results/preprocessed_data/df_test_imgs.csv")
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_name", type=str, default="lisa3d_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--use_aug", action="store_true", help="activar augmentations suaves del Dataset")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--spatial_size", nargs="+", type=int, default=[40,120,120])
    parser.add_argument("--run_sanity", action="store_true", help="hacer un back-testing split rápido antes del CV")
    args = parser.parse_args()

    main(args)
    """
        --save_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/maxvit_nano_rw_256_df_8_0.1_dynamic \

    python 5.training3d.py \
    --out_dir /data/cristian/projects/med_data/rise-miccai/task-1/2d_models/results/baseline_3d \
    --experiment_name lisa3d_baseline \
    --use_aug \
    --run_sanity

    
    """