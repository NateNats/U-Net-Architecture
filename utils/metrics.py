import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Dice Loss untuk training."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred         = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union        = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice         = (2. * intersection + self.smooth) / \
                       (union + self.smooth)
        return 1 - dice.mean()
    
class BCEDiceLoss(nn.Module):
    """BCE + Dice Loss untuk training."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(pred, target) + \
               0.5 * self.dice(pred, target)

def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float=0.5, smooth: float=1e-6) -> torch.Tensor:
    """
    computes the dice score for binary segmentation.
    :param pred: Tensor of model predictions (B, in_c, H, W).
    :param target: Tensor of ground truth (B, in_c, H, W).
    :param threshold: threshold for converting predictions to binary.
    :param smooth: smoothing factor to avoid division by zero.
    :return: torch.Tensor
    """

    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice.mean() 

def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float=0.5, smooth: float=1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection)

    total = (intersection + smooth) / (union + smooth)

    return total.mean()

class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth

    # rumus sigmoid = 1 / (1 + exp(-x))
    def compute(self, prediction: torch.Tensor, target: torch.Tensor) -> dict:
        pred_bin = (torch.sigmoid(prediction) >= self.threshold).float()
        model_pred = pred_bin.view(-1)
        actual = target.float().view(-1)
        s = self.smooth

        TP = (model_pred * actual).sum()
        TN = ((1 - model_pred) * (1 - actual)).sum()
        FP = (model_pred * (1 - actual)).sum()
        FN = ((1 - model_pred) * actual).sum()

        return {
            "dice"          : ((2 * TP + s) / (2 * TP + FP + FN + s)).item(),
            "iou"           : ((TP + s) / (TP + FP + FN + s)).item(),
            "accuracy"      : ((TP + TN + s) / (TP + TN + FP + FN + s)).item(),
            "precision"     : ((TP + s) / (TP + FP + s)).item(),
            "recall"        : ((TP + s) / (TP + FN + s)).item(),
            "specificity"   : ((TN + s) / (TN + FP + s)).item()
        }