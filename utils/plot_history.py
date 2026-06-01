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
import matplotlib.patches as mpatches
from pathlib import Path
import os

BASE = 'D:/Kuliah/Semester Last/U-Net-Architecture'

# ── History untuk kurva training terbaik per skenario ──────────────────────
HISTORY_FILES = {
    'Original' : f'{BASE}/checkpoints (epoch 50)/original_adamw_lr0.0001_history.json',
    'Bothat'   : f'{BASE}/checkpoints (epoch 50)/bothat_adamw_lr0.0001_history.json',
    'Laplacian': f'{BASE}/checkpoints (epoch 50)/laplacian_adamw_lr0.0001_history.json',
}

# ── History semua eksperimen (untuk bar chart, heatmap, LR sensitivity) ────
# Format: {scenario}_{optimizer}_lr{lr}
ALL_HISTORY_FILES = {
    'original_adamw_0.01'   : f'{BASE}/checkpoints (epoch 50)/original_adamw_lr0.01_history.json',
    'original_adamw_0.001'  : f'{BASE}/checkpoints (epoch 50)/original_adamw_lr0.001_history.json',
    'original_adamw_0.0001' : f'{BASE}/checkpoints (epoch 50)/original_adamw_lr0.0001_history.json',
    'original_lion_0.01'    : f'{BASE}/checkpoints (epoch 50)/original_lion_lr0.01_history.json',
    'original_lion_0.001'   : f'{BASE}/checkpoints (epoch 50)/original_lion_lr0.001_history.json',
    'original_lion_0.0001'  : f'{BASE}/checkpoints (epoch 50)/original_lion_lr0.0001_history.json',
    'bothat_adamw_0.01'     : f'{BASE}/checkpoints (epoch 50)/bothat_adamw_lr0.01_history.json',
    'bothat_adamw_0.001'    : f'{BASE}/checkpoints (epoch 50)/bothat_adamw_lr0.001_history.json',
    'bothat_adamw_0.0001'   : f'{BASE}/checkpoints (epoch 50)/bothat_adamw_lr0.0001_history.json',
    'bothat_lion_0.01'      : f'{BASE}/checkpoints (epoch 50)/bothat_lion_lr0.01_history.json',
    'bothat_lion_0.001'     : f'{BASE}/checkpoints (epoch 50)/bothat_lion_lr0.001_history.json',
    'bothat_lion_0.0001'    : f'{BASE}/checkpoints (epoch 50)/bothat_lion_lr0.0001_history.json',
    'laplacian_adamw_0.01'  : f'{BASE}/checkpoints (epoch 50)/laplacian_adamw_lr0.01_history.json',
    'laplacian_adamw_0.001' : f'{BASE}/checkpoints (epoch 50)/laplacian_adamw_lr0.001_history.json',
    'laplacian_adamw_0.0001': f'{BASE}/checkpoints (epoch 50)/laplacian_adamw_lr0.0001_history.json',
    'laplacian_lion_0.01'   : f'{BASE}/checkpoints (epoch 50)/laplacian_lion_lr0.01_history.json',
    'laplacian_lion_0.001'  : f'{BASE}/checkpoints (epoch 50)/laplacian_lion_lr0.001_history.json',
    'laplacian_lion_0.0001' : f'{BASE}/checkpoints (epoch 50)/laplacian_lion_lr0.0001_history.json',
}

OUTPUT_DIR = f'{BASE}/output/kurva_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'train': '#2196F3',
    'val'  : '#FF5722',
}

SCENARIO_COLORS = {
    'Original' : '#1565C0',
    'Bothat'   : '#E65100',
    'Laplacian': '#2E7D32',
}

OPTIMIZER_COLORS = {
    'AdamW': '#5C6BC0',
    'Lion' : '#EF6C00',
}

LR_MARKERS = {
    '0.01'  : 'o',
    '0.001' : 's',
    '0.0001': '^',
}


# ═════════════════════════════════════════
# 1. LOAD HISTORY
# ═════════════════════════════════════════
def load_history(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_best_epoch(history: dict, metric: str = 'iou') -> dict:
    """Cari epoch terbaik berdasarkan metrik val yang dipilih (default: iou)."""
    val      = history['val']
    best_idx = max(range(len(val)), key=lambda i: val[i][metric])
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


def load_all_histories(files: dict) -> dict:
    """Load semua file history, skip yang tidak ditemukan."""
    out = {}
    for key, path in files.items():
        p = Path(path)
        if p.exists():
            out[key] = load_history(path)
        else:
            print(f'  ⚠ Tidak ditemukan: {path}')
    return out


def parse_key(key: str):
    """
    Parse key dari nama file atau dict key.
    Mendukung dua format:
      - Nama file  : 'bothat_adamw_lr0_0001'  → (Bot Hat, Adamw, 0.0001)
      - Dict key   : 'bothat_adamw_0.0001'    → (Bot Hat, Adamw, 0.0001)
    """
    parts = key.split('_')

    # Deteksi format nama file: bagian LR berupa 'lr0' + '01'/'001'/'0001'
    if len(parts) >= 4 and parts[-2] == 'lr0':
        digits     = parts[-1]          # '01', '001', '0001'
        lr         = '0.' + digits.lstrip('0') or '0.' + digits
        # Normalisasi: '0.1' tidak ada, pastikan mapping benar
        lr_map = {'01': '0.01', '001': '0.001', '0001': '0.0001'}
        lr         = lr_map.get(digits, lr)
        optimizer  = parts[-3].capitalize()
        scenario_raw = '_'.join(parts[:-3])
    else:
        # Format dict key: last token adalah lr string
        lr         = parts[-1]
        optimizer  = parts[-2].capitalize()
        scenario_raw = '_'.join(parts[:-2])

    opt_map = {'adamw': 'AdamW', 'lion': 'Lion'}
    optimizer = opt_map.get(optimizer.lower(), optimizer.capitalize())

    scenario_map = {
        'original' : 'Original',
        'bothat'   : 'Bot Hat',
        'laplacian': 'Laplacian',
    }
    scenario = scenario_map.get(scenario_raw.lower(), scenario_raw.capitalize())
    return scenario, optimizer, lr


# ═════════════════════════════════════════
# 2. PLOT KURVA TRAINING — PER EKSPERIMEN
# ═════════════════════════════════════════
def plot_single_experiment(history: dict, experiment: str,
                           save_path: str = None):
    """Plot kurva training untuk satu eksperimen (Loss, Dice, IoU, Accuracy, Recall)."""
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
        ax.axvline(x=best['epoch'], color='green',
                   linestyle='--', linewidth=1.2, alpha=0.7,
                   label=f'Best (ep{best["epoch"]})')
        ax.scatter(best['epoch'], val_vals[best['epoch'] - 1],
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
    plt.close()


# ═════════════════════════════════════════
# 3. PLOT PERBANDINGAN KURVA 3 SKENARIO
# ═════════════════════════════════════════
def plot_comparison_curves(histories: dict, save_path: str = None):
    """Plot kurva Loss, Dice, IoU ketiga skenario dalam satu gambar."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        'Perbandingan Kurva Training — Original vs Bot Hat vs Laplacian',
        fontsize=14, fontweight='bold', y=1.01
    )

    metrics = [('loss', 'Loss'), ('dice', 'Dice Coefficient'), ('iou', 'IoU')]

    for col, (key, label) in enumerate(metrics):
        ax_train = axes[0, col]
        ax_val   = axes[1, col]
        ax_train.set_title(f'{label} (Training)', fontsize=10, fontweight='bold')
        ax_val.set_title(f'{label} (Validation)', fontsize=10, fontweight='bold')

        for exp, history in histories.items():
            train  = history['train']
            val    = history['val']
            epochs = range(1, len(train) + 1)
            color  = SCENARIO_COLORS.get(exp, '#888888')
            best   = get_best_epoch(history)

            t_vals = [ep[key] for ep in train]
            v_vals = [ep[key] for ep in val]

            ax_train.plot(epochs, t_vals, color=color, linewidth=1.8,
                          label=exp, alpha=0.85)
            ax_val.plot(epochs, v_vals, color=color, linewidth=1.8,
                        label=exp, alpha=0.85)
            ax_val.scatter(best['epoch'], v_vals[best['epoch'] - 1],
                           color=color, s=80, zorder=5, marker='*')

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
    plt.close()


# ═════════════════════════════════════════
# 4. PLOT SEMUA METRIK — SATU EKSPERIMEN
# ═════════════════════════════════════════
def plot_all_metrics(history: dict, experiment: str, save_path: str = None):
    """Plot semua 7 metrik + loss dalam grid 2×4 untuk satu eksperimen."""
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
        f'Best Epoch: {best["epoch"]} | Dice: {best["dice"]*100:.2f}% | '
        f'IoU: {best["iou"]*100:.2f}% | Recall: {best["recall"]*100:.2f}%',
        fontsize=12, fontweight='bold', y=1.02
    )

    axes_flat = list(axes.flat)
    for ax, (key, label) in zip(axes_flat, all_metrics):
        t_vals = [ep[key] for ep in train]
        v_vals = [ep[key] for ep in val]

        ax.plot(epochs, t_vals, color=COLORS['train'], linewidth=1.8,
                label='Train', alpha=0.9)
        ax.plot(epochs, v_vals, color=COLORS['val'], linewidth=1.8,
                label='Val', alpha=0.9)
        ax.axvline(x=best['epoch'], color='green', linestyle='--',
                   linewidth=1, alpha=0.6)
        ax.scatter(best['epoch'], v_vals[best['epoch'] - 1],
                   color='green', s=50, zorder=5)

        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═════════════════════════════════════════
# 5. TABEL RINGKASAN
# ═════════════════════════════════════════
def print_summary_table(histories: dict):
    print('\n' + '=' * 70)
    print('  RINGKASAN HASIL TRAINING')
    print('=' * 70)
    print(f'  {"Eksperimen":<12} {"Epoch":>6} {"Best Ep":>8} '
          f'{"Dice":>8} {"IoU":>8} {"Recall":>8} {"Loss":>8}')
    print('  ' + '-' * 62)
    for exp, history in histories.items():
        best = get_best_epoch(history)
        n    = len(history['train'])
        print(f'  {exp:<12} {n:>6} {best["epoch"]:>8} '
              f'{best["dice"]*100:>7.2f}% '
              f'{best["iou"]*100:>7.2f}% '
              f'{best["recall"]*100:>7.2f}% '
              f'{best["loss"]:>8.4f}')
    print('=' * 70 + '\n')


# ═══════════════════════════════════════════════════════════════════
# 6. [BARU] BAR CHART — PERBANDINGAN IoU SEMUA KONFIGURASI
# ═══════════════════════════════════════════════════════════════════
def plot_bar_iou_all(all_histories: dict, save_path: str = None):
    """
    Bar chart IoU validasi terbaik untuk semua 18 konfigurasi,
    dikelompokkan per skenario dengan warna berbeda per optimizer.
    """
    scenarios = ['Original', 'Bot Hat', 'Laplacian']
    optimizers = ['AdamW', 'Lion']
    lrs = ['0.01', '0.001', '0.0001']

    # Kumpulkan data
    results = {}
    for key, history in all_histories.items():
        scenario, optimizer, lr = parse_key(key)
        best = get_best_epoch(history)
        results[(scenario, optimizer, lr)] = best['iou']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        'Perbandingan IoU Validasi Terbaik — Semua Konfigurasi',
        fontsize=13, fontweight='bold', y=1.02
    )

    x      = np.arange(len(lrs))
    width  = 0.35
    colors = {'AdamW': '#5C6BC0', 'Lion': '#EF6C00'}

    for ax, scenario in zip(axes, scenarios):
        for i, optimizer in enumerate(optimizers):
            vals = [results.get((scenario, optimizer, lr), 0) for lr in lrs]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=optimizer, color=colors[optimizer],
                          alpha=0.85, edgecolor='white', linewidth=0.8)

            # Nilai di atas bar
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f'{val:.4f}', ha='center', va='bottom',
                            fontsize=7.5, fontweight='bold', rotation=90)

        ax.set_title(scenario, fontsize=11, fontweight='bold')
        ax.set_xlabel('Learning Rate', fontsize=9)
        ax.set_ylabel('IoU' if ax == axes[0] else '', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(lrs, fontsize=9)
        ax.set_ylim(0.60, 0.85)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7. [BARU] HEATMAP — IoU per Optimizer × LR × Skenario
# ═══════════════════════════════════════════════════════════════════
def plot_heatmap_iou(all_histories: dict, save_path: str = None):
    """
    Heatmap IoU validasi terbaik.
    Baris = kombinasi Skenario × Optimizer, Kolom = Learning Rate.
    """
    scenarios  = ['Original', 'Bot Hat', 'Laplacian']
    optimizers = ['AdamW', 'Lion']
    lrs        = ['0.01', '0.001', '0.0001']

    row_labels = [f'{s} — {o}' for s in scenarios for o in optimizers]
    matrix     = np.zeros((len(row_labels), len(lrs)))

    results = {}
    for key, history in all_histories.items():
        scenario, optimizer, lr = parse_key(key)
        results[(scenario, optimizer, lr)] = get_best_epoch(history)['iou']

    for r, (s, o) in enumerate([(s, o) for s in scenarios for o in optimizers]):
        for c, lr in enumerate(lrs):
            matrix[r, c] = results.get((s, o, lr), np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=matrix[~np.isnan(matrix)].min() - 0.01,
                   vmax=matrix[~np.isnan(matrix)].max() + 0.01)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('IoU', fontsize=10)

    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels(lrs, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_title('Heatmap IoU Validasi Terbaik\n(Skenario × Optimizer × Learning Rate)',
                 fontsize=12, fontweight='bold')

    # Nilai di dalam cell
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if not np.isnan(val):
                # Tentukan warna teks (putih kalau background gelap)
                norm_val = (val - matrix.min()) / (matrix.max() - matrix.min())
                txt_color = 'white' if norm_val > 0.65 else 'black'
                ax.text(c, r, f'{val:.4f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=txt_color)

    # Garis pemisah antar skenario
    for y in [1.5, 3.5]:
        ax.axhline(y=y, color='white', linewidth=2.5)

    # Label skenario di sisi kanan
    ax.annotate('Original',  xy=(1.01, 5/6), xycoords='axes fraction',
                fontsize=10, fontweight='bold', va='center',
                color=SCENARIO_COLORS['Original'],
                annotation_clip=False)
    ax.annotate('Bot Hat',   xy=(1.01, 3/6), xycoords='axes fraction',
                fontsize=10, fontweight='bold', va='center',
                color=SCENARIO_COLORS['Bothat'],
                annotation_clip=False)
    ax.annotate('Laplacian', xy=(1.01, 1/6), xycoords='axes fraction',
                fontsize=10, fontweight='bold', va='center',
                color=SCENARIO_COLORS['Laplacian'],
                annotation_clip=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 8. [BARU] LINE CHART — PENGARUH LEARNING RATE TERHADAP IoU
# ═══════════════════════════════════════════════════════════════════
def plot_lr_sensitivity(all_histories: dict, save_path: str = None):
    """
    Line chart IoU terbaik vs Learning Rate untuk setiap
    kombinasi Skenario × Optimizer. Menunjukkan sensitivitas LR.
    """
    scenarios  = ['Original', 'Bot Hat', 'Laplacian']
    optimizers = ['AdamW', 'Lion']
    lrs_num    = [0.01, 0.001, 0.0001]
    lrs_str    = ['0.01', '0.001', '0.0001']

    results = {}
    for key, history in all_histories.items():
        scenario, optimizer, lr = parse_key(key)
        results[(scenario, optimizer, lr)] = get_best_epoch(history)['iou']

    linestyles = {'AdamW': '-', 'Lion': '--'}
    scenario_markers = {'Original': 'o', 'Bot Hat': 's', 'Laplacian': '^'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        'Pengaruh Learning Rate terhadap IoU Validasi Terbaik',
        fontsize=13, fontweight='bold', y=1.02
    )

    for ax, optimizer in zip(axes, optimizers):
        ax.set_title(f'Optimizer: {optimizer}', fontsize=11, fontweight='bold')
        for scenario in scenarios:
            color = SCENARIO_COLORS.get(
                scenario if scenario != 'Bot Hat' else 'Bothat', '#888888'
            )
            iou_vals = [results.get((scenario, optimizer, lr), np.nan)
                        for lr in lrs_str]
            ax.plot(lrs_num, iou_vals,
                    color=color, linestyle=linestyles[optimizer],
                    marker=scenario_markers[scenario],
                    linewidth=2, markersize=8, label=scenario, alpha=0.9)

            # Anotasi nilai di setiap titik
            for lr_n, iou in zip(lrs_num, iou_vals):
                if not np.isnan(iou):
                    ax.annotate(f'{iou:.4f}',
                                xy=(lr_n, iou),
                                xytext=(0, 8), textcoords='offset points',
                                ha='center', fontsize=7.5, color=color)

        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate (skala log)', fontsize=10)
        ax.set_ylabel('IoU' if optimizer == 'AdamW' else '', fontsize=10)
        ax.set_xticks(lrs_num)
        ax.set_xticklabels(lrs_str, fontsize=9)
        ax.invert_xaxis()  # 0.01 → 0.001 → 0.0001 (kiri ke kanan = besar ke kecil)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 9. [BARU] GROUPED BAR — PERBANDINGAN SEMUA METRIK TERBAIK
#          (untuk 3 skenario terbaik saja)
# ═══════════════════════════════════════════════════════════════════
def plot_best_metrics_comparison(all_histories: dict, save_path: str = None):
    """
    Grouped bar chart membandingkan semua metrik dari
    konfigurasi TERBAIK masing-masing skenario.
    Konfigurasi terbaik = IoU val tertinggi di semua optimizer & LR.
    """
    scenarios   = ['Original', 'Bot Hat', 'Laplacian']
    metric_keys = ['loss', 'dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity']
    metric_lbls = ['Loss', 'Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity']

    # Cari konfigurasi terbaik per skenario
    best_per_scenario = {}
    for key, history in all_histories.items():
        scenario, optimizer, lr = parse_key(key)
        best = get_best_epoch(history)
        prev = best_per_scenario.get(scenario)
        if prev is None or best['iou'] > prev['iou']:
            best_per_scenario[scenario] = {**best,
                                           'optimizer': optimizer,
                                           'lr': lr}

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        'Perbandingan Metrik Konfigurasi Terbaik per Skenario\n'
        '(Loss: semakin rendah semakin baik  |  Metrik lain: semakin tinggi semakin baik)',
        fontsize=12, fontweight='bold', y=1.03
    )

    x     = np.arange(len(metric_keys))
    width = 0.25

    for i, scenario in enumerate(scenarios):
        data  = best_per_scenario.get(scenario, {})
        vals  = [data.get(k, 0) for k in metric_keys]
        color = SCENARIO_COLORS.get(
            scenario if scenario != 'Bot Hat' else 'Bothat', '#888888'
        )
        offset = (i - 1) * width
        opt    = data.get('optimizer', '')
        lr     = data.get('lr', '')
        label  = f'{scenario} ({opt}, lr={lr})'
        bars   = ax.bar(x + offset, vals, width,
                        label=label, color=color, alpha=0.85,
                        edgecolor='white', linewidth=0.8)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, rotation=90, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_lbls, fontsize=10)
    ax.set_ylabel('Nilai', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════
if __name__ == '__main__':
    print('\n── Loading history files (best per scenario)... ──')
    histories = {}
    for exp, path in HISTORY_FILES.items():
        if Path(path).exists():
            histories[exp] = load_history(path)
            print(f'  ✅ {exp}: {len(histories[exp]["train"])} epochs')
        else:
            print(f'  ❌ {exp}: file tidak ditemukan ({path})')

    print('\n── Loading all experiment histories... ──')
    all_histories = load_all_histories(ALL_HISTORY_FILES)
    print(f'  ✅ Berhasil load: {len(all_histories)} dari {len(ALL_HISTORY_FILES)} file')

    # ── Tabel ringkasan ───────────────────────────────────────────
    if histories:
        print_summary_table(histories)

    # ── Plot per eksperimen (kurva terbaik) ───────────────────────
    if histories:
        print('── Kurva per skenario terbaik... ──')
        for exp, history in histories.items():
            plot_single_experiment(
                history, exp,
                save_path=f'{OUTPUT_DIR}/kurva_{exp.lower()}.png'
            )

    # ── Plot semua metrik per eksperimen ──────────────────────────
    if histories:
        print('── Semua metrik per skenario terbaik... ──')
        for exp, history in histories.items():
            plot_all_metrics(
                history, exp,
                save_path=f'{OUTPUT_DIR}/semua_metrik_{exp.lower()}.png'
            )

    # ── Plot perbandingan kurva 3 skenario ────────────────────────
    if len(histories) > 1:
        print('── Perbandingan kurva 3 skenario... ──')
        plot_comparison_curves(
            histories,
            save_path=f'{OUTPUT_DIR}/perbandingan_ketiga.png'
        )

    # ── [BARU] Bar chart IoU semua konfigurasi ────────────────────
    if all_histories:
        print('── [BARU] Bar chart IoU semua konfigurasi... ──')
        plot_bar_iou_all(
            all_histories,
            save_path=f'{OUTPUT_DIR}/barchart_iou_semua.png'
        )

    # ── [BARU] Heatmap IoU ────────────────────────────────────────
    if all_histories:
        print('── [BARU] Heatmap IoU... ──')
        plot_heatmap_iou(
            all_histories,
            save_path=f'{OUTPUT_DIR}/heatmap_iou.png'
        )

    # ── [BARU] Line chart pengaruh LR ────────────────────────────
    if all_histories:
        print('── [BARU] Pengaruh learning rate... ──')
        plot_lr_sensitivity(
            all_histories,
            save_path=f'{OUTPUT_DIR}/lr_sensitivity.png'
        )

    # ── [BARU] Grouped bar metrik konfigurasi terbaik ────────────
    if all_histories:
        print('── [BARU] Perbandingan metrik konfigurasi terbaik... ──')
        plot_best_metrics_comparison(
            all_histories,
            save_path=f'{OUTPUT_DIR}/best_metrics_comparison.png'
        )

    print(f'\n✅ Semua plot disimpan di: {OUTPUT_DIR}')