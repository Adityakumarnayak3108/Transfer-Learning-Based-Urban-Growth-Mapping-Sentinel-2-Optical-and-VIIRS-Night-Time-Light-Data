import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class ComboLoss(nn.Module):
    """BCE + Dice, handles class imbalance well for sparse building masks."""
    def __init__(self, pos_weight=None, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return (1 - self.dice_weight) * self.bce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)