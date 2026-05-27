"""
Cara pakai:
  python plot_history.py

Atau dari notebook:
  %run plot_history.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os

BASE = 'C:/Users/Cerdas05/Skripshot/U-Net-Architecture'

HISTORY_FILES = {
    'Original' : f'{BASE}/checkpoints/original/history.json',
    'Bothat'   : f'{BASE}/checkpoints/bothat/history.json',
    'Laplacian': f'{BASE}/checkpoints/laplacian/history.json',
}

OUTPUT_DIR = f'{BASE}/output/kurva_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'train': '#2196F3',   
    'val'  : '#FF5722',   
}


# ═════════════════════════════════════════
# 1. LOAD HISTORY
# ═════════════════════════════════════════
def load_history(path: str) -> dict:
    """Load history.json dan kembalikan sebagai dict."""
    with open(path) as f:
        return json.load(f)

def get_best_epoch(history: dict) -> dict:
    """Cari epoch terbaik berdasarkan val_dice tertinggi."""
    val      = history['val']
    best_idx = max(range(len(val)), key=lambda i: val[i]['dice'])
    return {
        'epoch'      : best_idx + 1,
        'dice'       : val[best_idx]['dice'],
        'iou'        : val[best_idx]['iou'],
        'accuracy'   : val[best_idx]['accuracy'],
        'precision'  : val[best_idx]['precision'],
        'recall'     : val[best_idx]['recall'],
        'specificity': val[best_idx]['specificity'],
        'loss'       : val[best_idx]['loss'],
    }


# ═════════════════════════════════════════
# 2. PLOT KURVA TRAINING — PER EKSPERIMEN
# ═════════════════════════════════════════
def plot_single_experiment(history: dict, experiment: str,
                            save_path: str = None):
    """
    Plot kurva training untuk satu eksperimen.
    Menampilkan: Loss, Dice, IoU, Accuracy, Recall
    """
    train  = history['train']
    val    = history['val']
    epochs = range(1, len(train) + 1)
    best   = get_best_epoch(history)

    metrics = [
        ('loss',     'Loss'),
        ('dice',     'Dice Coefficient'),
        ('iou',      'IoU'),
        ('accuracy', 'Accuracy'),
        ('recall',   'Recall'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f'Kurva Training — {experiment.upper()}\n'
        f'Best Epoch: {best["epoch"]} | '
        f'Val Dice: {best["dice"]*100:.2f}% | '
        f'Val IoU: {best["iou"]*100:.2f}%',
        fontsize=12, fontweight='bold', y=1.02
    )

    for ax, (key, label) in zip(axes, metrics):
        train_vals = [ep[key] for ep in train]
        val_vals   = [ep[key] for ep in val]

        ax.plot(epochs, train_vals, color=COLORS['train'],
                linewidth=1.8, label='Training', alpha=0.9)
        ax.plot(epochs, val_vals, color=COLORS['val'],
                linewidth=1.8, label='Validation', alpha=0.9)

        # Tandai best epoch
        ax.axvline(x=best['epoch'], color='green',
                   linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'Best (ep{best["epoch"]})')
        ax.scatter(best['epoch'], val_vals[best['epoch']-1],
                   color='green', s=60, zorder=5)

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.show()
    plt.close()


# ═════════════════════════════════════════
# 3. PLOT KURVA 3 EKSPERIMEN — LOSS & DICE
# ═════════════════════════════════════════
def plot_comparison_curves(histories: dict,
                            save_path: str = None):
    """
    Plot kurva Loss dan Dice ketiga eksperimen
    dalam satu gambar untuk perbandingan.
    """
    exp_colors = {
        'Original' : '#1565C0',
        'Bothat'   : '#E65100',
        'Laplacian': '#2E7D32',
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Perbandingan Kurva Training — '
        'Original vs Bothat vs Laplacian',
        fontsize=14, fontweight='bold', y=1.01
    )

    metrics = [
        ('loss', 'Loss'),
        ('dice', 'Dice Coefficient'),
        ('iou',  'IoU'),
    ]

    for col, (key, label) in enumerate(metrics):
        # Baris 1 — Training
        ax_train = axes[0, col]
        ax_train.set_title(f'{label} (Training)',
                            fontsize=10, fontweight='bold')

        # Baris 2 — Validation
        ax_val = axes[1, col]
        ax_val.set_title(f'{label} (Validation)',
                          fontsize=10, fontweight='bold')

        for exp, history in histories.items():
            train  = history['train']
            val    = history['val']
            epochs = range(1, len(train) + 1)
            color  = exp_colors[exp]
            best   = get_best_epoch(history)

            t_vals = [ep[key] for ep in train]
            v_vals = [ep[key] for ep in val]

            ax_train.plot(epochs, t_vals, color=color,
                          linewidth=1.8, label=exp, alpha=0.85)
            ax_val.plot(epochs, v_vals, color=color,
                        linewidth=1.8, label=exp, alpha=0.85)

            # Best epoch marker
            ax_val.scatter(best['epoch'], v_vals[best['epoch']-1],
                           color=color, s=80, zorder=5,
                           marker='*')

        for ax in [ax_train, ax_val]:
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_ylabel(label, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.show()
    plt.close()


# ═════════════════════════════════════════
# 4. PLOT SEMUA METRIK — SATU EKSPERIMEN
# ═════════════════════════════════════════
def plot_all_metrics(history: dict, experiment: str,
                     save_path: str = None):
    """
    Plot semua 6 metrik + loss dalam grid 2x4
    untuk satu eksperimen.
    """
    train  = history['train']
    val    = history['val']
    epochs = range(1, len(train) + 1)
    best   = get_best_epoch(history)

    all_metrics = [
        ('loss',        'Loss'),
        ('dice',        'Dice'),
        ('iou',         'IoU'),
        ('accuracy',    'Accuracy'),
        ('precision',   'Precision'),
        ('recall',      'Recall'),
        ('specificity', 'Specificity'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        f'Semua Metrik Training — {experiment.upper()}\n'
        f'Best Epoch: {best["epoch"]} | '
        f'Dice: {best["dice"]*100:.2f}% | '
        f'IoU: {best["iou"]*100:.2f}% | '
        f'Recall: {best["recall"]*100:.2f}%',
        fontsize=12, fontweight='bold', y=1.02
    )

    axes_flat = axes.flat
    for ax, (key, label) in zip(axes_flat, all_metrics):
        t_vals = [ep[key] for ep in train]
        v_vals = [ep[key] for ep in val]

        ax.plot(epochs, t_vals, color=COLORS['train'],
                linewidth=1.8, label='Train', alpha=0.9)
        ax.plot(epochs, v_vals, color=COLORS['val'],
                linewidth=1.8, label='Val', alpha=0.9)
        ax.axvline(x=best['epoch'], color='green',
                   linestyle='--', linewidth=1, alpha=0.6)
        ax.scatter(best['epoch'], v_vals[best['epoch']-1],
                   color='green', s=50, zorder=5)

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Sembunyikan subplot kosong (index 7)
    list(axes_flat)[-1].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.show()
    plt.close()


# ═════════════════════════════════════════
# 5. TABEL RINGKASAN
# ═════════════════════════════════════════
def print_summary_table(histories: dict):
    """Print tabel ringkasan hasil training."""
    print('\n' + '='*70)
    print(f'  RINGKASAN HASIL TRAINING')
    print('='*70)
    print(f'  {"Eksperimen":<12} {"Epoch":>6} {"Best Ep":>8} '
          f'{"Dice":>8} {"IoU":>8} {"Recall":>8} {"Loss":>8}')
    print('  ' + '-'*62)

    for exp, history in histories.items():
        best = get_best_epoch(history)
        n    = len(history['train'])
        print(
            f'  {exp:<12} {n:>6} {best["epoch"]:>8} '
            f'{best["dice"]*100:>7.2f}% '
            f'{best["iou"]*100:>7.2f}% '
            f'{best["recall"]*100:>7.2f}% '
            f'{best["loss"]:>8.4f}'
        )
    print('='*70 + '\n')


# ═════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════
if __name__ == '__main__':
    print('\n── Loading history files... ──')

    histories = {}
    for exp, path in HISTORY_FILES.items():
        if Path(path).exists():
            histories[exp] = load_history(path)
            print(f'  ✅ {exp}: {len(histories[exp]["train"])} epochs')
        else:
            print(f'  ❌ {exp}: file tidak ditemukan ({path})')

    if not histories:
        print('Tidak ada file history yang ditemukan!')
        exit()

    # ── Tabel ringkasan ───────────────────
    print_summary_table(histories)

    # ── Plot per eksperimen ───────────────
    print('── Membuat kurva per eksperimen... ──')
    for exp, history in histories.items():
        plot_single_experiment(
            history, exp,
            save_path=f'{OUTPUT_DIR}/kurva_{exp.lower()}.png'
        )

    # ── Plot semua metrik per eksperimen ──
    print('── Membuat plot semua metrik... ──')
    for exp, history in histories.items():
        plot_all_metrics(
            history, exp,
            save_path=f'{OUTPUT_DIR}/semua_metrik_{exp.lower()}.png'
        )

    # ── Plot perbandingan ketiga ──────────
    if len(histories) > 1:
        print('── Membuat plot perbandingan... ──')
        plot_comparison_curves(
            histories,
            save_path=f'{OUTPUT_DIR}/perbandingan_ketiga.png'
        )

    print(f'\n✅ Semua plot disimpan di: {OUTPUT_DIR}')