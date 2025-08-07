import numpy as np
import torch
def mixup_data(x,view, y, alpha=0.4):
    """Retorna datos mezclados x, y y el coef lambda"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_view = lam * view + (1 - lam) * view[index]
    y_a, y_b = y, y[index]
    return mixed_x,mixed_view, y_a, y_b, lam
