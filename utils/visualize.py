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

    if mode == 'single':
        fig = _single_visualize(img, gt, pred)
    elif mode == 'comparison':
        fig = _comparison_visualize(img, gt, pred)
    elif mode == 'multi':
        fig = _multi_visualize(img, gt, pred)
    else:
        raise ValueError(f'mode tidak ada dalam pada fungsi, silahkan masukkan single/comparison/multi')

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    return fig


def _single_visualize(img: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    fig, axes = plt.subplots(1, 4, figsize=(30, 12))

    overlay = cv2.addWeighted(img, 0.6, pred, 0.4, 0)

    panels = [('Original', img), ('Ground Truth', gt), ('Prediction', pred), ('Overlay', overlay)]
    for ax, (label, data) in zip(axes, panels):
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
        ax.set_title(label)
        ax.axis('off')

    plt.tight_layout()
    return fig


def _comparison_visualize(img: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    panels = [('Original', img), ('Ground Truth', gt), ('Prediction', pred)]
    for ax, (label, data) in zip(axes, panels):
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
        ax.set_title(label)
        ax.axis('off')

    plt.tight_layout()
    return fig


def _multi_visualize(img: np.ndarray, gt: np.ndarray, pred: np.ndarray):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    diff = cv2.absdiff(gt, pred)

    panels = [('Original', img), ('Ground Truth', gt), ('Prediction', pred), ('Difference', diff)]
    for ax, (label, data) in zip(axes.flat, panels):
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
        ax.set_title(label)
        ax.axis('off')

    plt.tight_layout()
    return fig
