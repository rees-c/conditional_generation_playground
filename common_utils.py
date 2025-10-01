import torch
import torch.nn as nn


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def compute_grad_norm(model: nn.Module) -> float:
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    total_grad_norm = torch.cat(grads).norm()
    return total_grad_norm
