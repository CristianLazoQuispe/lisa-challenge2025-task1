import os, random, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import logging
import wandb

from .losses import get_per_label_criterions
from .dataset2D import MRIDataset2D
from .dataset3D import MRIDataset3D
from .models2D import Model2DTimm
from .models3D import Model3DResnet
from .mywandb import init_wandb
from .mixup import mixup_data
import scipy.stats
from . import utils

from collections import deque
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

def epoch_subsample(df, frac=0.5, seed=None):
    # Submuestrea dentro de cada paciente sin el warning
    rng = np.random.default_rng(seed)

    # Lista de patient_id únicos y muestreo del frac
    unique_patients = df["patient_id"].unique()
    n_select = max(1, int(len(unique_patients) * frac))
    selected_patients = rng.choice(unique_patients, size=n_select, replace=False)

    # Filtrar DataFrame
    df_selected = df[df["patient_id"].isin(selected_patients)]

    # Elegir un registro aleatorio por paciente
    df_sampled = (
        df_selected.groupby("patient_id", group_keys=False)
                   .sample(n=5, random_state=seed, replace=True)
                  .reset_index(drop=True)
                   #.apply(lambda g: g.sample(n=5, random_state=seed, replace=True))
                   #.reset_index(drop=True)
    )
    return df_sampled
    """
    return (
        df.groupby("patient_id", group_keys=False)
          .sample(frac=frac, random_state=seed)
          .reset_index(drop=True)
    )
    """
    
def apply_label_smoothing(y, smoothing=0.05, num_classes=3):
    """Convierte etiquetas one-hot a suavizadas."""
    with torch.no_grad():
        y_smooth = torch.full_like(y, smoothing / (num_classes - 1))
        y_smooth.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0 - smoothing)
    return y_smooth

def cutmix_data(x, y, alpha=1.0):
    """Aplicar CutMix."""
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generar caja aleatoria para CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class SlidingWindowAverage:
    def __init__(self, window_size=10):
        self.values = deque(maxlen=window_size)

    def update(self, val):
        self.values.append(val)

    def get(self):
        if not self.values:
            return 0.0
        return np.mean(self.values)

    def is_full(self):
        return len(self.values) == self.values.maxlen

def inference_dataset(val_loader,model,args):
    val_probs, val_trues, val_paths = [], [], []

    with torch.no_grad():
        for x, y, path, view in val_loader:
            x = x.to(args.device)
            view = view.to(args.device)
            with torch.cuda.amp.autocast():
                y_hat = model(x)
            pred = torch.argmax(y_hat, dim=-1).cpu()
            # 🧠 Guarda softmax por clase
            batch_probs = torch.stack([torch.softmax(y_hat[:, i], dim=-1) for i in range(len(args.label_cols))], dim=1)
            # shape (B, 7, 3)
            val_probs.append(batch_probs.cpu())
            val_trues.append(y.cpu())
            val_paths.extend(path)

    y_pred_val,y_prod_val,y_true_val,final_filenames = aggregate_predictions_by_img_path(val_probs,val_trues, val_paths)

    return y_pred_val,y_prod_val, y_true_val,final_filenames

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

def get_metrics(df_train, val_oof_records, args, prefix=""):
    val_df_oof = pd.DataFrame(val_oof_records)
    print("GET metrics")
    print(df_train.shape)
    print(val_df_oof.shape)
    val_df_oof = df_train[["filename", *args.label_cols]].merge(val_df_oof, on="filename", suffixes=("_true", "_pred"))

    y_true = val_df_oof[[f"{l}_true" for l in args.label_cols]].values
    y_pred = val_df_oof[[f"{l}_pred" for l in args.label_cols]].values

    results = {}

    # 🎯 Global
    results[f"{prefix}f1_macro"] = f1_score(y_true.flatten(), y_pred.flatten(), average="macro",zero_division=0)
    results[f"{prefix}f1_micro"] = f1_score(y_true.flatten(), y_pred.flatten(), average="micro",zero_division=0)
    results[f"{prefix}acc"] = accuracy_score(y_true.flatten(), y_pred.flatten())

    # 📊 Por etiqueta
    for i, label in enumerate(args.label_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        f1_macro = f1_score(yt, yp, average="macro",zero_division=0)
        f1_micro = f1_score(yt, yp, average="micro",zero_division=0)
        acc = accuracy_score(yt, yp)
        _, _, f2, _ = precision_recall_fscore_support(yt, yp, average="macro", beta=2,zero_division=0)

        results[f"{prefix}_{label}_f1_macro"] = f1_macro
        results[f"{prefix}_{label}_f1_micro"] = f1_micro
        results[f"{prefix}_{label}_f2"] = f2
        results[f"{prefix}_{label}_acc"] = acc

    return results

import pandas as pd
import numpy as np
from collections import defaultdict

def aggregate_predictions_by_img_path(val_probs,val_trues, val_paths):
    """
    Agrupa y promedia predicciones y etiquetas verdaderas por imagen base (ej. LISA_0001_LF_axi.nii.gz).
    
    Returns:
        y_pred (np.ndarray): (N, 7) clases predichas por clase (argmax)
    """
    probs_all = torch.cat(val_probs, dim=0).numpy()  # (N, 7, 3)
    trues_all = torch.cat(val_trues, dim=0).numpy()  # (N, 7)
    #/data/cristian/projects/med_data/rise-miccai/task-1-val/2d_images_all/train/LISA_0001_LF_axi_0.png
    paths_all = ["_".join(p.split("/")[-1].split(".")[0].split("_")[:-1])+".nii.gz" for p in val_paths]

    #print("paths_all:")
    #print(paths_all)
    #print()
    # Diccionarios para acumulación
    grouped_probs = defaultdict(list)
    grouped_trues = defaultdict(list)
    final_filenames = []
    for path, probs, true in zip(paths_all, probs_all, trues_all):
        grouped_probs[path].append(probs)  # (7, 3)
        grouped_trues[path].append(true)   # (7,)

    final_prods, final_preds, final_trues = [], [], []
    for key in grouped_probs:
        mean_probs = np.mean(grouped_probs[key], axis=0)      # (7, 3)
        pred = np.argmax(mean_probs, axis=1)                  # (7,)
        final_preds.append(pred)
        final_prods.append(mean_probs)

        mean_true = np.mean(grouped_trues[key], axis=0)       # (7,)
        true = np.round(mean_true).astype(int)                # (7,)
        final_trues.append(true)
        final_filenames.append(key)
    return np.stack(final_preds),np.stack(final_prods), np.stack(final_trues),final_filenames

import torch.nn.utils as nn_utils

def update_optimizer(model,optimizer, grad_scaler, clip_grad_max_norm,lr_scheduler):
    """Performs gradient update with AMP, including gradient clipping."""
    if grad_scaler:
        grad_scaler.unscale_(optimizer)  # Unscale gradients before clipping

    nn_utils.clip_grad_norm_(model.parameters(), clip_grad_max_norm)

    if grad_scaler:
        grad_scaler.step(optimizer)  # Step and automatically handle NaN/infs
        grad_scaler.update()  # Update scaling factor
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # Always reset gradients
    if lr_scheduler:
        lr_scheduler.step()


def train_with_cv(df_train, df_test, label_cols, args,save_dir="./models_new/"):
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(filename=f"{args.save_dir}/training.log", level=logging.INFO)
    weights_per_label = utils.compute_weights_from_df(df_train,args.label_cols,args.weight_method,
    args.weight_beta,          # para "effective"
    args.weight_alpha,          # para "invfreq"
    args.weight_cap,            # para "invfreq"
    args.device,
    dtype=torch.float32)
    
    print("weights_per_label:",weights_per_label)
    criterions = get_per_label_criterions(label_cols, weights_per_label, args)

    val_oof_records  = []
    test_preds_accum = []
    fold_scores = []
    log_values = {}
    for fold in range(args.n_splits):
        logging.info(f"🔁 Fold {fold}")
        run = init_wandb(args,f"{args.experiment_name}:f{fold}")
        log_values={}
        
        df_tr = df_train[df_train.fold != fold]
        df_val = df_train[df_train.fold == fold]

        ids_counts = df_tr["patient_id"].value_counts()
        ids_train  = ids_counts.index.tolist()
        df_common = df_val[df_val["patient_id"].isin(ids_train)].copy()
        print(f"train  : {df_train.shape}")
        print(f"df_tr  : {df_tr.shape}")
        print(f"df_val : {df_val.shape}")
        print(f"Train Test fold {fold} Casos comunes: {len(df_common)}")

        #raise

        for i, label in enumerate(label_cols):
            y = df_tr[label].values
            print(f"Train {label} true dist: {np.bincount(y, minlength=3)}")

        for i, label in enumerate(label_cols):
            y = df_val[label].values
            print(f"Val {label} true dist: {np.bincount(y, minlength=3)}")

        window_f1 = SlidingWindowAverage(window_size=5)
        df_tr_full = df_tr.copy()

        # Dataset
        if args.type_modeling == "2d":
            val_ds   = MRIDataset2D(df=df_val ,is_train=True,use_augmentation=False,is_numpy=bool(args.is_numpy),labels=args.label_cols,image_size=args.image_size)
            test_ds  = MRIDataset2D(df=df_test,is_train=True,use_augmentation=False,is_numpy=bool(args.is_numpy),labels=args.label_cols,image_size=args.image_size)
        else:
            val_ds   = MRIDataset3D(df=df_val ,is_train=True,use_augmentation=False,labels=args.label_cols)
            test_ds  = MRIDataset3D(df=df_test,is_train=True,use_augmentation=False,labels=args.label_cols)

        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)
        test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)


        if args.type_modeling == "2d":
            model = Model2DTimm(base_model=args.base_model,in_channels=args.in_channels,num_labels=len(args.label_cols),
                                        num_classes=3,dropout_p=args.dropout_p).to(args.device)
        else:
            model = Model3DResnet(num_labels=len(args.label_cols), num_classes=3,dropout_p=args.dropout_p,
                                        pretrained=True, freeze_backbone=False).to(args.device) 

        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=5)


        val_best_f1 = 0
        patience = args.patience
        step = 0
        for epoch in range(args.epochs):

            df_tr = epoch_subsample(df_tr_full, frac=args.slice_frac, seed=epoch)
            print(f"df_tr recortado: {df_tr.shape}")

            #for i, label in enumerate(label_cols):
            #    y = df_tr[label].values
            #    print(f"epoch {epoch} Train {label} true dist: {np.bincount(y, minlength=3)}")


            # Dataset
            if args.type_modeling == "2d":
                train_ds = MRIDataset2D(df=df_tr  ,is_train=True,use_augmentation=True, is_numpy=bool(args.is_numpy),labels=args.label_cols,image_size=args.image_size)
            else:
                train_ds = MRIDataset3D(df=df_tr  ,is_train=True,use_augmentation=True,labels=args.label_cols)
            #"""
            # Sampler sin alterar distribución del dataset
            if bool(args.use_sampling):
                print("USING use_sampling")
                weights = utils.compute_sample_weights(df_tr, label_cols)
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
            else:
                print("NO use_sampling")
                train_loader = DataLoader(train_ds, batch_size=args.batch_size,  shuffle=True, num_workers=args.num_workers, persistent_workers=True)

            loss_log_values = {}
            for i, label in enumerate(label_cols):
                loss_log_values[f"val_{label}_loss"] = 0

            model.train()
            total_loss = 0
            for x, y, path, view in tqdm(train_loader, desc=f"Train Fold {fold} Epoch {epoch}"):
                x, y = x.to(args.device), y.to(args.device)
                view = view.to(args.device)
                with torch.cuda.amp.autocast():

                    # Alternancia Mixup / CutMix
                    r = random.random()
                    if r < args.mixup_prob:
                        x, view, y_a, y_b, lam = mixup_data(x, view, y, alpha=args.mixup_alpha)
                        y_hat = model(x)
                        loss = sum(lam * criterions[i](y_hat[:, i], y_a[:, i]) +
                                   (1 - lam) * criterions[i](y_hat[:, i], y_b[:, i])
                                   for i in range(len(label_cols))) / len(label_cols)
                    elif r < args.mixup_prob + args.cutmix_prob:
                        x, y_a, y_b, lam = cutmix_data(x, y, alpha=args.cutmix_alpha)
                        y_hat = model(x)
                        loss = sum(lam * criterions[i](y_hat[:, i], y_a[:, i]) +
                                   (1 - lam) * criterions[i](y_hat[:, i], y_b[:, i])
                                   for i in range(len(label_cols))) / len(label_cols)
                    else:
                        y_hat = model(x)
                        loss = sum(criterions[i](y_hat[:, i], y[:, i]) for i in range(len(label_cols))) / len(label_cols)


                    #optimizer.zero_grad()
                    #loss.backward()
                    grad_scaler.scale(loss).backward()
                    update_optimizer(model,optimizer, grad_scaler, 0.5,None)
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    #optimizer.step()
                    total_loss += loss.item()

            # Validation
            model.eval()
            val_preds, val_trues = [], []
            val_probs, val_trues, val_paths = [], [], []
            val_total_loss = 0
            with torch.no_grad():
                for x, y, path, view in tqdm(val_loader, desc=f"Val Fold {fold} Epoch {epoch}"):
                    x, y = x.to(args.device), y.to(args.device)
                    view = view.to(args.device)
                    with torch.cuda.amp.autocast():
                        y_hat = model(x)
                        loss = sum(criterions[i](y_hat[:, i], y[:, i]) for i in range(len(label_cols))) / len(label_cols)
                        pred = torch.argmax(y_hat, dim=-1).cpu()
                        # 🧠 Guarda softmax por clase
                        batch_probs = torch.stack([torch.softmax(y_hat[:, i], dim=-1) for i in range(len(label_cols))], dim=1)
                        # shape (B, 7, 3)
                        val_probs.append(batch_probs.cpu())
                        val_trues.append(y.cpu())
                        val_paths.extend(path)
                        val_preds.append(pred)
                        val_total_loss+=loss.item()

                        # 🎯 Diagnóstico de pérdida por etiqueta
                        losses_by_label = [
                            criterions[i](y_hat[:, i], y[:, i]).item()
                            for i in range(len(label_cols))
                        ]

                        for i, label in enumerate(label_cols):
                            loss_log_values[f"val_{label}_loss"] += losses_by_label[i]

                        for i, label in enumerate(label_cols):
                            probs = torch.softmax(y_hat[:, i], dim=-1).detach().cpu().numpy()
                            entropies = scipy.stats.entropy(probs.T)  # (B,)
                            #print(f"{label} entropy mean: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")


            #y_pred_val = torch.cat(val_preds).cpu().numpy()
            #y_true_val = torch.cat(val_trues).cpu().numpy()
            y_pred_val,y_prod_val,y_true_val,final_filenames = aggregate_predictions_by_img_path(val_probs,val_trues, val_paths)

            for i, label in enumerate(label_cols):
                y_true = y_true_val[:, i]
                y_pred = y_pred_val[:, i]
                if label == "Banding":
                    print(f"\n🔎 {label} - Preds:", np.bincount(y_pred, minlength=3), "| Truth:", np.bincount(y_true, minlength=3))

            val_f1 = f1_score(y_true_val.flatten(), y_pred_val.flatten(), average="macro")

            scheduler.step(val_f1)
            val_acc = accuracy_score(y_true_val.flatten(), y_pred_val.flatten())
            window_f1.update(val_f1)
            val_f1_mean = window_f1.get()
            train_loss = total_loss / len(train_loader)
            val_loss   = val_total_loss/ len(val_loader)

            print(f"📈 Epoch {epoch} | Train loss: {train_loss:.4f} | Val loss : {val_loss:.4f}| F1: {val_f1:.4f} | Acc: {val_acc:.4f} | F1_mean: {val_f1_mean:.4f} | BestF1: {val_best_f1:4f}")


            if val_f1_mean > val_best_f1:
                val_best_f1 = val_f1_mean
                torch.save(model.state_dict(), f"{save_dir}/model_fold{fold}.pt")
                patience = args.patience
            else:
                patience -= 1
                if patience == 0:
                    print("⏹️ Early stopping")
                    break
            for i, label in enumerate(label_cols):
                loss_log_values[f"val_{label}_loss"] /= len(val_loader)
    
            log_values={}
            log_values[f"epoch"] = epoch
            log_values[f"train_loss"] = train_loss
            log_values[f"val_loss"] = val_loss
            log_values[f"val_f1_macro"] = val_f1
            log_values[f"val_acc"] = val_acc
            log_values[f"val_f1_macro_mean"] = val_f1_mean
            log_values[f"val_best_f1"] = val_best_f1
            log_values.update(loss_log_values)

            # 📊 Métricas por etiqueta
            for i, label in enumerate(label_cols):
                yt = y_true_val[:, i]
                yp = y_pred_val[:, i]
                f1 = f1_score(yt, yp, average="macro",zero_division=0)
                f1_micro = f1_score(yt, yp, average="micro",zero_division=0)
                acc = accuracy_score(yt, yp)
                _, _, f2, _ = precision_recall_fscore_support(yt, yp, average="macro", beta=2,zero_division=0)
                log_values[f"val_{label}_f1_macro"] = f1
                log_values[f"val_{label}_f1_micro"] = f1_micro
                log_values[f"val_{label}_f2_macro"] = f2
                log_values[f"val_{label}_acc"]      = acc

            wandb.log(log_values, step=step)
            step+=1

        # ✅ OOF pred con el mejor modelo
        fold_scores.append({"fold": fold, "val_f1": val_best_f1})

        print(f"🔁 Loading best model for fold {fold}")
        model.load_state_dict(torch.load(f"{save_dir}/model_fold{fold}.pt"))
        model.eval()

        y_pred_val,y_prod_val,y_true_val,val_files   = inference_dataset(val_loader,model,args)
        for yp, yt, f in zip(y_pred_val, y_true_val, val_files):
            val_oof_records.append({**{l: yp[i] for i, l in enumerate(args.label_cols)}, "filename": f})

        test_oof_records = []
        y_pred_test,y_prod_val,y_true_test,test_files = inference_dataset(test_loader,model,args)
        for yp, yt, f in zip(y_pred_test, y_true_test, test_files):
            test_oof_records.append({**{l: yp[i] for i, l in enumerate(args.label_cols)}, "filename": f})
        test_results = get_metrics(df_test,test_oof_records,args,prefix="test_oof_")
        log_values.update(test_results)
        wandb.log(log_values, step=step)

        if fold < (args.n_splits-1):
            wandb.finish()

    val_results  = get_metrics(df_train,val_oof_records,args,prefix="val_oof_")
    test_results = get_metrics(df_test,test_oof_records,args,prefix="test_oof_")


    # 🎯 Global CV F1 Macro    
    for val_key,test_key in zip(val_results.keys(),test_results.keys()):
        print(f"🎯 Global CV {val_key}: {val_results[val_key]:.4f} | CV {test_key}: {test_results[test_key]:.4f}")
    
    final_results = {
        "fold_scores": fold_scores,
        "oof_path": f"{save_dir}/oof_predictions.csv",
        "step":step
    }

    final_results.update(val_results)
    final_results.update(test_results)

    log_values.update(final_results)
    wandb.log(log_values, step=step)
    wandb.finish()

    return final_results



    