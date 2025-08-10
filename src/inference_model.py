import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from .trainer import aggregate_predictions_by_img_path
from . import aggregations_preds
import numpy as np

def inference_model(model, dataset, label_cols, device="cuda", batch_size=16, num_workers=4,args=None, method = "probs",thr2=0.10, thr1=0.30):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()

    all_trues, all_probs, all_preds, all_paths = [], [], [], []

    with torch.no_grad():
        for x,y,path, view in tqdm(loader, desc="🚀 Inference"):
            x = x.to(device)
            view  = view.to(device)
            logits = model(x)         # (B, 7, 3)
            #probs = F.softmax(logits, dim=-1)  # (B, 7, 3)
            pred = torch.argmax(logits, dim=-1)
            # 🧠 Guarda softmax por clase
            #print("x:",x.shape)
            #print("logits:",logits.shape)
            batch_probs = torch.stack([torch.softmax(logits[:, i], dim=-1) for i in range(len(label_cols))], dim=1)
            # shape (B, 7, 3)
            #print("batch_probs:",batch_probs.shape)
            all_probs.append(batch_probs.cpu())
            all_paths.extend(path)
            all_trues.append(y.cpu())
            all_preds.append(pred.cpu())

    
    if method=="probs":
        y_pred_val,y_prod_val,y_true_val,final_filenames = aggregate_predictions_by_img_path(all_probs,all_trues, all_paths)
    else:
        """
        all_probs = torch.cat(all_probs, dim=0).numpy()  # (N, 7, 3)
        all_preds = torch.cat(all_preds, dim=0).numpy()  # (N, 7, 3)
        y_pred_val = all_preds
        y_prod_val = all_probs
        y_true_val = all_trues
        final_filenames = ["_".join(p.split("/")[-1].split(".")[0].split("_")[:-1])+".nii.gz" for p in all_paths]
        #"""
        y_pred_val,y_prod_val,y_true_val,final_filenames = aggregations_preds.aggregate_predictions_by_img_path(all_probs,all_trues, all_paths, thr2=thr2, thr1=thr1)

    # Guardar como DataFrame
    records = []
    for i in range(len(final_filenames)):
        row = {"filename": final_filenames[i]}
        for j, label in enumerate(label_cols):
            for k in range(3):
                row[f"{label}_{k}"] = y_prod_val[i, j, k]
            row[label] = y_pred_val[i, j]
        records.append(row)

    df_out = pd.DataFrame(records)
    return df_out
