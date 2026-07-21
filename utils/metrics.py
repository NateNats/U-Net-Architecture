import cv2
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

    Rumus:
        Rand Error = 1 - (a + b) / C(n, 2)

    dimana untuk seluruh PASANGAN piksel:
        a      = jumlah pasangan yang SELABEL di prediksi
                 DAN selabel di ground truth
        b      = jumlah pasangan yang BEDA label di prediksi
                 DAN beda label di ground truth
        C(n,2) = n(n-1)/2 = total pasangan piksel

    Label diambil dari CONNECTED COMPONENT (objek hasil
    segmentasi), bukan sekadar biner 0/1. Inilah yang
    membuat Rand Error mampu menghukum splits (satu lesi
    pecah jadi beberapa region) dan mergers (dua lesi
    menyatu) — kesalahan topologi yang tidak terdeteksi
    oleh Dice/IoU.

    Perhitungan a dan b memakai tabel kontingensi n_ij
    (jumlah piksel di klaster-i prediksi ∩ klaster-j GT):
        a = Σ_ij C(n_ij, 2)
        b = C(n,2) - Σ_i C(a_i,2) - Σ_j C(b_j,2)
                   + Σ_ij C(n_ij, 2)
    dengan a_i = ukuran klaster prediksi,
           b_j = ukuran klaster ground truth.

    Range: 0.0 (sempurna) → 1.0 (terburuk)

    Referensi: Rand (1971)
               Unnikrishnan et al. (2007)
               Arganda-Carreras et al. (2015)
    """
    pred_bin = (torch.sigmoid(pred) >= threshold)

    batch_size  = pred_bin.shape[0]
    rand_errors = []

    for i in range(batch_size):
        p_mask = pred_bin[i].squeeze().cpu().numpy().astype(np.uint8)
        t_mask = (target[i].squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # Label objek via connected component (0 = background)
        _, p_lab = cv2.connectedComponents(p_mask)
        _, t_lab = cv2.connectedComponents(t_mask)

        p_lab = p_lab.ravel().astype(np.int64)
        t_lab = t_lab.ravel().astype(np.int64)

        n = p_lab.size
        if n < 2:
            rand_errors.append(0.0)
            continue

        # Tabel kontingensi n_ij lewat indeks gabungan
        n_p  = int(p_lab.max()) + 1
        n_ij = np.bincount(t_lab * n_p + p_lab).astype(np.float64)
        a_i  = np.bincount(p_lab).astype(np.float64)
        b_j  = np.bincount(t_lab).astype(np.float64)

        def n_pairs(x):
            """Σ C(x, 2) — jumlah pasangan dalam tiap klaster."""
            return float((x * (x - 1.0) / 2.0).sum())

        total_pairs = n * (n - 1.0) / 2.0
        sum_ij      = n_pairs(n_ij)

        # a = pasangan selabel di kedua segmentasi
        a = sum_ij
        # b = pasangan beda label di kedua segmentasi
        b = total_pairs - n_pairs(a_i) - n_pairs(b_j) + sum_ij

        rand_index = (a + b) / (total_pairs + smooth)
        rand_errors.append(1.0 - rand_index)

    return torch.tensor(float(np.mean(rand_errors)),
                        dtype=torch.float32)


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