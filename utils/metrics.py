import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5,
               smooth: float = 1e-6) -> torch.Tensor:
    pred         = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union        = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice         = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5,
              smooth: float = 1e-6) -> torch.Tensor:
    pred         = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union        = pred.sum(dim=(2, 3)) + \
                   target.sum(dim=(2, 3)) - intersection
    total        = (intersection + smooth) / (union + smooth)
    return total.mean()


def pixel_error(pred: torch.Tensor,
                target: torch.Tensor,
                threshold: float = 0.5) -> torch.Tensor:
    """
    Pixel Error = proporsi piksel yang salah diklasifikasikan.

    Rumus: (jumlah piksel salah) / (total piksel)
    Range : 0.0 (sempurna) → 1.0 (terburuk)

    Referensi: Arganda-Carreras et al. (2015)
    """
    pred      = (torch.sigmoid(pred) >= threshold).float()
    incorrect = (pred != target).float()
    return incorrect.mean()


def rand_error(pred: torch.Tensor,
               target: torch.Tensor,
               threshold: float = 0.5,
               smooth: float = 1e-6) -> torch.Tensor:
    """
    Rand Error = 1 - Rand Index
    Mengukur ketidaksesuaian pengelompokan piksel antara
    prediksi dan ground truth.

    Rand Index mengukur frekuensi pasangan piksel yang
    mendapat label sama di kedua segmentasi atau label
    berbeda di kedua segmentasi.

    Rumus Rand Index:
        RI = (TP + TN) / (TP + TN + FP + FN)
    dimana:
        TP = pasangan piksel sama di pred & target
        TN = pasangan piksel beda di pred & target
        FP = pasangan piksel sama di pred, beda di target
        FN = pasangan piksel beda di pred, sama di target

    Rand Error = 1 - RI

    Range: 0.0 (sempurna) → 1.0 (terburuk)

    Referensi: Arganda-Carreras et al. (2015)
               Unnikrishnan et al. (2007)
    """
    # Binarisasi prediksi
    pred_bin = (torch.sigmoid(pred) >= threshold).float()

    # Flatten per gambar dalam batch
    batch_size = pred_bin.shape[0]
    rand_errors = []

    for i in range(batch_size):
        # Ambil satu gambar
        p = pred_bin[i].view(-1)   # prediksi
        t = target[i].float().view(-1)  # ground truth

        # Hitung pasangan piksel
        # TP: keduanya sama-sama 1
        TP = (p * t).sum()

        # TN: keduanya sama-sama 0
        TN = ((1 - p) * (1 - t)).sum()

        # FP: pred=1, target=0
        FP = (p * (1 - t)).sum()

        # FN: pred=0, target=1
        FN = ((1 - p) * t).sum()

        # Rand Index
        rand_idx = (TP + TN + smooth) / \
                   (TP + TN + FP + FN + smooth)

        # Rand Error = 1 - Rand Index
        rand_err = 1.0 - rand_idx
        rand_errors.append(rand_err)

    return torch.stack(rand_errors).mean()


def warping_error(pred: torch.Tensor,
                  target: torch.Tensor,
                  threshold: float = 0.5) -> torch.Tensor:
    """
    Warping Error = mengukur ketidaksesuaian TOPOLOGI
    antara prediksi dan ground truth.

    Warping error menghitung perbedaan antara kontur/tepi
    prediksi dengan kontur/tepi ground truth menggunakan
    pendekatan berbasis morfologi.

    Tidak menghukum perbedaan lokasi batas yang kecil,
    tapi menghukum perbedaan topologi (splits & mergers).

    Range: 0.0 (sempurna) → 1.0 (terburuk)

    Referensi: Arganda-Carreras et al. (2015)
               Jain et al. (2010)
    """
    import cv2

    # Binarisasi prediksi
    pred_bin = (torch.sigmoid(pred) >= threshold).float()

    batch_size = pred_bin.shape[0]
    warp_errors = []

    for i in range(batch_size):
        # Konversi ke numpy
        p = pred_bin[i].squeeze().cpu().numpy().astype(np.uint8) * 255
        t = target[i].squeeze().cpu().numpy().astype(np.uint8)   * 255

        # ── Deteksi kontur prediksi ──────────────────────
        contours_pred, _ = cv2.findContours(
            p,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Deteksi kontur ground truth ──────────────────
        contours_gt, _ = cv2.findContours(
            t,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Buat mask kontur ─────────────────────────────
        mask_pred = np.zeros_like(p)
        mask_gt   = np.zeros_like(t)

        cv2.drawContours(mask_pred, contours_pred, -1, 255, 1)
        cv2.drawContours(mask_gt,   contours_gt,   -1, 255, 1)

        # ── Warping error = perbedaan kontur ─────────────
        # Piksel kontur yang berbeda antara pred dan GT
        diff     = np.abs(mask_pred.astype(int) -
                          mask_gt.astype(int))
        warp_err = diff.mean() / 255.0

        warp_errors.append(warp_err)

    # Konversi ke tensor
    return torch.tensor(np.mean(warp_errors),
                        dtype=torch.float32)


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5,
                 smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth    = smooth

    def compute(self, prediction: torch.Tensor,
                target: torch.Tensor) -> dict:
        """
        Hitung semua metrik sekaligus termasuk
        Pixel Error, Rand Error, dan Warping Error.
        """
        pred_bin   = (torch.sigmoid(prediction) >=
                      self.threshold).float()
        model_pred = pred_bin.view(-1)
        actual     = target.float().view(-1)
        s          = self.smooth

        TP = (model_pred * actual).sum()
        TN = ((1 - model_pred) * (1 - actual)).sum()
        FP = (model_pred * (1 - actual)).sum()
        FN = ((1 - model_pred) * actual).sum()

        return {
            # ── Metrik utama ─────────────────────────────
            "dice"         : ((2*TP+s) /
                              (2*TP+FP+FN+s)).item(),
            "iou"          : ((TP+s) /
                              (TP+FP+FN+s)).item(),
            "pixel_error"  : pixel_error(
                                 prediction, target,
                                 self.threshold).item(),
            "rand_error"   : rand_error(
                                 prediction, target,
                                 self.threshold,
                                 self.smooth).item(),
            "warping_error": warping_error(
                                 prediction, target,
                                 self.threshold).item(),
        }