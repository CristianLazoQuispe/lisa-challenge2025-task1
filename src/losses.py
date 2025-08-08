# ✅ focal_entropy_loss_module_fixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot_smooth(target, num_classes, eps):
    # target: (B,) long -> (B,C) float suavizado
    with torch.no_grad():
        y = torch.zeros(target.size(0), num_classes, device=target.device, dtype=torch.float)
        y.scatter_(1, target.unsqueeze(1), 1.0)
        if eps > 0:
            y = y * (1 - eps) + eps / num_classes
    return y

class FocalLossv2(nn.Module):
    """
    Focal Loss multi-clase con:
      - weights por clase (alpha)
      - label smoothing (soft-target)
      - reducción 'mean'
    input:  logits (B, C)
    target: long (B,) con clases 0..C-1
    """
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0, reduction="mean", eps=1e-8, num_classes=3):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight if weight is not None else None)  # (C,)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.eps = eps
        self.num_classes = num_classes

    def forward(self, input, target):
        # probs y log-probs
        logp = F.log_softmax(input, dim=-1)               # (B,C)
        p    = logp.exp()                                  # (B,C)

        # y_hat suavizado
        y = one_hot_smooth(target, num_classes=self.num_classes, eps=self.label_smoothing)  # (B,C)

        # CE con soft labels: -sum y * logp
        ce = -(y * logp).sum(dim=-1)                      # (B,)

        # p_t = sum y * p  (prob objetivo con etiquetas suaves)
        pt = (y * p).sum(dim=-1).clamp_min(self.eps)      # (B,)

        # factor focal
        focal = (1.0 - pt).pow(self.gamma)                # (B,)

        # alpha (peso por clase): alpha_t = sum y * alpha
        if self.weight is not None:
            alpha_t = (y * self.weight.unsqueeze(0)).sum(dim=-1)  # (B,)
        else:
            alpha_t = 1.0

        loss = alpha_t * focal * ce                        # (B,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'

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




class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def get_per_label_criterions(label_cols, weights_per_label, args):
    criterions = []
    for label in label_cols:
        weight = weights_per_label[label].to(args.device) if weights_per_label else None

        if args.loss_name == "focal_lossv1":
            crit = FocalLossv2(
                gamma=args.focal_gamma,                 # e.g., 1.5 or 2.0
                weight=weight,                          # tensor (3,)
                label_smoothing=args.label_smoothing,   # e.g., 0.05
                num_classes=3,
                reduction="mean"
            )
        elif args.loss_name == "focal_lossv2":
            crit = FocalLoss(gamma=args.focal_gamma,alpha=weight,task_type="multi-class",num_classes=3,reduction="mean")
        elif args.loss_name == "focal_lossv3":
            crit = FocalLoss_original(gamma=args.focal_gamma, weight=weight,label_smoothing=args.label_smoothing)
        elif args.loss_name == "cross_simple":
            crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        elif args.loss_name == "cross_weighted":
            crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        criterions.append(crit.to(args.device))
    return criterions
