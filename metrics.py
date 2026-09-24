import torch

def pixel_accuracy(logits, targets, threshold=0.5):
    preds = (torch.sigmoid(logits) > threshold).float()

    correct = (preds == targets).float().sum()
    total = torch.numel(targets)

    return correct / total


def iou_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + eps) / (union + eps)


def dice_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum()
    return (2 * intersection + eps) / (preds.sum() + targets.sum() + eps)