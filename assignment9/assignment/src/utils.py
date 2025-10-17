"""
Training utilities for budget-optimized ResNet50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Mixup:
    """Mixup augmentation"""
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, data, targets):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = data.size(0)
        index = torch.randperm(batch_size).to(data.device)

        mixed_data = lam * data + (1 - lam) * data[index, :]
        targets_a, targets_b = targets, targets[index]
        return mixed_data, targets_a, targets_b, lam


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save model checkpoint"""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_acc1', 0)


def adjust_learning_rate(optimizer, epoch, initial_lr, total_epochs=75, warmup_epochs=5):
    """
    BEST-IN-CLASS learning rate schedule for 75%+ accuracy
    - Warmup prevents early divergence
    - Cosine decay provides smooth convergence
    - Proven effective for ImageNet training
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    # Apply learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr
