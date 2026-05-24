import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def visualize_segmentation(image_path: str, gt_path: str, pred_path: str, save_path=None, metrics=None, title=None, mode='comparison'):
    img_raw = cv2.imread(image_path)
    gt_raw = cv2.imread(gt_path)
    pred_raw = cv2.imread(pred_path)

    if img_raw is None or gt_raw is None or pred_raw is None:
        raise FileNotFoundError('Salah satu file gambar tidak ditemukan')

    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(gt_raw, cv2.COLOR_BGR2RGB)
    pred = cv2.cvtColor(pred_raw, cv2.COLOR_BGR2RGB)

    gt_mask = _to_binary_mask(gt)
    pred_mask = _to_binary_mask(pred)

    if mode == 'single':
        fig = _single_visualize(img, gt_mask, pred_mask)
    elif mode == 'comparison':
        fig = _comparison_visualize(img, gt_mask, pred_mask)
    elif mode == 'multi':
        fig = _multi_visualize(img, gt_mask, pred_mask, metrics)
    else:
        raise ValueError(f'mode tidak ada dalam pada fungsi, silahkan masukkan single/comparison/multi')

    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def _to_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        gray = mask
    return (gray > 127).astype(np.uint8)


def _make_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Buat peta error berwarna: TP=hijau, FP=merah, FN=biru, TN=hitam."""
    h, w = gt_mask.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    tp = (gt_mask == 1) & (pred_mask == 1)
    fp = (gt_mask == 0) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    error_map[tp] = [0, 200, 0]    # hijau
    error_map[fp] = [220, 0, 0]    # merah
    error_map[fn] = [0, 80, 220]   # biru
    return error_map


def _overlay_mask_on_image(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    overlay = img.copy().astype(np.float32)
    color_layer = np.zeros_like(img, dtype=np.float32)
    color_layer[mask == 1] = color
    blended = np.where(mask[:, :, None] == 1,
                       overlay * (1 - alpha) + color_layer * alpha,
                       overlay)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _add_metrics_text(ax, gt_mask: np.ndarray, pred_mask: np.ndarray):
    tp = np.sum((gt_mask == 1) & (pred_mask == 1))
    fp = np.sum((gt_mask == 0) & (pred_mask == 1))
    fn = np.sum((gt_mask == 1) & (pred_mask == 0))
    tn = np.sum((gt_mask == 0) & (pred_mask == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    text = (
        f"IoU      : {iou:.4f}\n"
        f"Dice     : {dice:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"Pixel Acc: {pixel_acc:.4f}"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))


def _single_visualize(img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray):
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))

    gt_overlay = _overlay_mask_on_image(img, gt_mask, color=(50, 205, 50))
    pred_overlay = _overlay_mask_on_image(img, pred_mask, color=(255, 80, 80))
    error_map = _make_error_map(gt_mask, pred_mask)

    legend = [
        Patch(color='green', label='True Positive'),
        Patch(color='red', label='False Positive'),
        Patch(color='blue', label='False Negative'),
    ]

    panels = [
        ('Original', img, None),
        ('Ground Truth', gt_overlay, None),
        ('Prediction', pred_overlay, None),
        ('Error Map', error_map, legend),
    ]
    for ax, (label, data, leg) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')
        if leg:
            ax.legend(handles=leg, loc='lower right', fontsize=8,
                      framealpha=0.85, fancybox=True)

    plt.tight_layout()
    return fig


def _comparison_visualize(img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    gt_overlay = _overlay_mask_on_image(img, gt_mask, color=(50, 205, 50))
    pred_overlay = _overlay_mask_on_image(img, pred_mask, color=(255, 80, 80))

    panels = [('Original', img), ('Ground Truth', gt_overlay), ('Prediction', pred_overlay)]
    for ax, (label, data) in zip(axes, panels):
        ax.imshow(data)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')

    gt_patch = Patch(color='#32CD32', label='Ground Truth Mask')
    pred_patch = Patch(color='#FF5050', label='Prediction Mask')
    fig.legend(handles=[gt_patch, pred_patch], loc='lower center',
               ncol=2, fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    return fig


def _multi_visualize(img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, metrics=None):
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    ax_orig   = fig.add_subplot(gs[0, 0])
    ax_gt     = fig.add_subplot(gs[0, 1])
    ax_pred   = fig.add_subplot(gs[0, 2])
    ax_err    = fig.add_subplot(gs[1, 0])
    ax_metric = fig.add_subplot(gs[1, 1:])

    gt_overlay   = _overlay_mask_on_image(img, gt_mask,   color=(50, 205, 50))
    pred_overlay = _overlay_mask_on_image(img, pred_mask, color=(255, 80, 80))
    error_map    = _make_error_map(gt_mask, pred_mask)

    ax_orig.imshow(img);           ax_orig.set_title('Original', fontweight='bold');      ax_orig.axis('off')
    ax_gt.imshow(gt_overlay);      ax_gt.set_title('Ground Truth', fontweight='bold');    ax_gt.axis('off')
    ax_pred.imshow(pred_overlay);  ax_pred.set_title('Prediction', fontweight='bold');    ax_pred.axis('off')
    ax_err.imshow(error_map);      ax_err.set_title('Error Map', fontweight='bold');      ax_err.axis('off')

    err_legend = [
        Patch(color='#00C800', label='True Positive (TP)'),
        Patch(color='#DC0000', label='False Positive (FP)'),
        Patch(color='#0050DC', label='False Negative (FN)'),
        Patch(color='#111111', label='True Negative (TN)'),
    ]
    ax_err.legend(handles=err_legend, loc='lower right', fontsize=8,
                  framealpha=0.85, fancybox=True)

    _draw_metrics_panel(ax_metric, gt_mask, pred_mask, external_metrics=metrics)

    return fig


def _draw_metrics_panel(ax, gt_mask: np.ndarray, pred_mask: np.ndarray, external_metrics=None):
    tp = int(np.sum((gt_mask == 1) & (pred_mask == 1)))
    fp = int(np.sum((gt_mask == 0) & (pred_mask == 1)))
    fn = int(np.sum((gt_mask == 1) & (pred_mask == 0)))
    tn = int(np.sum((gt_mask == 0) & (pred_mask == 0)))

    precision  = tp / (tp + fp + 1e-8)
    recall     = tp / (tp + fn + 1e-8)
    dice       = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou        = tp / (tp + fp + fn + 1e-8)
    pixel_acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    if external_metrics:
        iou       = external_metrics.get('iou', iou)
        dice      = external_metrics.get('dice', dice)
        precision = external_metrics.get('precision', precision)
        recall    = external_metrics.get('recall', recall)
        pixel_acc = external_metrics.get('pixel_acc', pixel_acc)

    names  = ['IoU', 'Dice', 'Precision', 'Recall', 'Pixel Acc', 'Specificity']
    values = [iou, dice, precision, recall, pixel_acc, specificity]
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

    bars = ax.barh(names, values, color=colors, height=0.55, edgecolor='white')
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_title('Segmentation Metrics', fontsize=12, fontweight='bold')
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

    count_text = f"TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}"
    ax.text(0.5, -0.08, count_text, transform=ax.transAxes,
            ha='center', fontsize=9, color='gray')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=10)
