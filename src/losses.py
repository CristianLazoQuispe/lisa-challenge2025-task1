# ✅ focal_entropy_loss_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def entropy_regularization(logits, weight=0.05):
    probs = torch.softmax(logits, dim=-1) + 1e-8
    entropy = -(probs * probs.log()).sum(dim=-1)  # (B,)
    low_entropy_penalty = -weight * entropy.mean()
    return low_entropy_penalty


class FocalLoss_original(nn.Module):
    def __init__(self, gamma=2, weight=None,label_smoothing=0.2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)#,label_smoothing=label_smoothing)

    def forward(self, input, target):
        logp = torch.nn.functional.log_softmax(input, dim=1)
        ce_loss = self.ce(input, target)
        #p = torch.exp(logp[range(len(target)), target])
        p = torch.softmax(input, dim=1)[range(len(target)), target]
        focal_loss = (1 - p) ** self.gamma * ce_loss

        return focal_loss.mean()

        #keep_ratio = 0.8
        #keep_indices = torch.argsort(focal_loss)[-int(keep_ratio * len(focal_loss)):]
        #loss = focal_loss[keep_indices].mean()        
        #return loss

def get_per_label_criterions(label_cols, weights_per_label, args):
    criterions = []
    for label in label_cols:
        print("LOSS label:",label)
        weight = weights_per_label[label].to(args.device) if weights_per_label else None

        #if label == "Banding":  # Especial para Banding
            #print("weight:",weight)
        #    class_weights = torch.tensor([0.1, 1990.0,2000.0]).to(args.device)
            #crit = lambda input, target: FocalLoss(gamma=2.0, weight=class_weights,debug=False)(input, target) \
            #                             + entropy_regularization(input, weight=0.01)

        #if label in ["Banding"]:
        #    weight = torch.tensor([0.1, 19.0, 20.0]).to(args.device)  # ejemplo fuerte

        print(f"{label} - weight:",weight)
        if args.loss_name == "focal_loss":
            crit = FocalLoss_original(gamma=2.0, weight=weight,label_smoothing=0.2)
            #crit = FocalLoss( gamma=2.0,alpha=weight,task_type="multi-class",num_classes=3,reduction="mean")
        elif args.loss_name == "cross_simple":
            crit = nn.CrossEntropyLoss(label_smoothing=0.2)
        elif args.loss_name == "cross_weighted":
            crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.2)
        else:
            class_weights = torch.tensor([1.0, 3.0, 5.0]).to(args.device)
            crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

        criterions.append(crit)
    return criterions
