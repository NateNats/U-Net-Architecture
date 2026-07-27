"""
Script untuk memvisualisasikan hasil testing model U-Net.

Cara pakai:
  python plot_test_results.py

Output disimpan di folder output/kurva_training/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# BASE = 'D:/Kuliah/Semester Last/U-Net-Architecture'
BASE = 'C:/Users/Cerdas05/Skripshot/U-Net-Architecture'

TEST_FILES = {
    'Original' : f'{BASE}/test_result(1000)/original_adamw_lr0.0001_test_results.json',
    'Bot Hat'  : f'{BASE}/test_result(1000)/bothat_adamw_lr0.0001_test_results.json',
    'Laplacian': f'{BASE}/test_result(1000)/laplacian_adamw_lr0.0001_test_results.json',
}

# Nilai validasi terbaik (dari training) untuk perbandingan
VAL_BEST = {
    'Original' : {'loss':0.1412,'dice':0.8570,'iou':0.7648,'accuracy':0.9236,'precision':0.8351,'recall':0.9138,'specificity':0.9420},
    'Bot Hat'  : {'loss':0.1252,'dice':0.8781,'iou':0.7866,'accuracy':0.9369,'precision':0.8710,'recall':0.8897,'specificity':0.9550},
    'Laplacian': {'loss':0.1227,'dice':0.8772,'iou':0.7854,'accuracy':0.9373,'precision':0.8808,'recall':0.8789,'specificity':0.9596},
}

OUTPUT_DIR = f'{BASE}/output/kurva_training'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENARIOS   = ['Original', 'Bot Hat', 'Laplacian']
SC_COLORS   = {'Original': '#1565C0', 'Bot Hat': '#E65100', 'Laplacian': '#2E7D32'}
METRICS     = ['loss', 'dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity']
METRIC_LBLS = ['Loss', 'Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity']


# ═════════════════════════════════════════
# LOAD DATA
# ═════════════════════════════════════════
def load_test_results(files: dict) -> dict:
    data = {}
    for name, path in files.items():
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data[name] = json.load(f)
            print(f'  ✅ {name}: {len(data[name]["per_sample"])} sampel')
        else:
            print(f'  ❌ {name}: file tidak ditemukan ({path})')
    return data


# ═════════════════════════════════════════
# 1. BAR CHART — SEMUA METRIK TESTING
# ═════════════════════════════════════════
def plot_test_metrics_comparison(test_data: dict, save_path: str = None):
    """Grouped bar chart semua metrik testing untuk 3 skenario."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(
        'Hasil Testing — Perbandingan Semua Metrik Antar Skenario\n'
        '(AdamW, lr=0.0001)',
        fontsize=12, fontweight='bold', y=1.03
    )

    x     = np.arange(len(METRICS))
    width = 0.25

    for i, sc in enumerate(SCENARIOS):
        vals   = [test_data[sc]['test_aggregate'][m] for m in METRICS]
        color  = SC_COLORS[sc]
        offset = (i - 1) * width
        bars   = ax.bar(x + offset, vals, width, label=sc,
                        color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, rotation=90, color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LBLS, fontsize=10)
    ax.set_ylabel('Nilai', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═════════════════════════════════════════
# 2. BAR CHART — VALIDASI VS TESTING
# ═════════════════════════════════════════
def plot_val_vs_test(test_data: dict, save_path: str = None):
    """Bar chart perbandingan val vs test per skenario (tanpa Loss)."""
    metrics_no_loss = [m for m in METRICS if m != 'loss']
    lbls_no_loss    = [l for m, l in zip(METRICS, METRIC_LBLS) if m != 'loss']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle(
        'Perbandingan Nilai Validasi vs Testing per Skenario\n'
        '(Konfigurasi Terbaik: AdamW, lr=0.0001)',
        fontsize=13, fontweight='bold', y=1.03
    )

    x     = np.arange(len(metrics_no_loss))
    width = 0.35

    for ax, sc in zip(axes, SCENARIOS):
        color     = SC_COLORS[sc]
        val_vals  = [VAL_BEST[sc][m]  for m in metrics_no_loss]
        test_vals = [test_data[sc]['test_aggregate'][m] for m in metrics_no_loss]

        b1 = ax.bar(x - width/2, val_vals,  width, label='Validasi',
                    color=color, alpha=0.90, edgecolor='white')
        b2 = ax.bar(x + width/2, test_vals, width, label='Testing',
                    color=color, alpha=0.45, edgecolor='white')

        for bar, val in zip(b1, val_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6.5, rotation=90, color=color, fontweight='bold')
        for bar, val in zip(b2, test_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=6.5, rotation=90, color='#555555', fontweight='bold')

        ax.set_title(sc, fontsize=11, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls_no_loss, fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Nilai' if sc == 'Original' else '')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═════════════════════════════════════════
# 3. BOXPLOT — DISTRIBUSI IoU PER SAMPEL
# ═════════════════════════════════════════
def plot_iou_boxplot(test_data: dict, save_path: str = None):
    """Boxplot distribusi IoU per sampel untuk ketiga skenario."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        'Distribusi IoU Per Sampel pada Data Testing (n=1.000)',
        fontsize=13, fontweight='bold'
    )

    iou_data = [[s['iou'] for s in test_data[sc]['per_sample']] for sc in SCENARIOS]

    bp = ax.boxplot(
        iou_data, patch_artist=True, notch=False,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=2, alpha=0.4)
    )

    for patch, sc in zip(bp['boxes'], SCENARIOS):
        patch.set_facecolor(SC_COLORS[sc])
        patch.set_alpha(0.7)

    for i, (sc, iou_vals) in enumerate(zip(SCENARIOS, iou_data), 1):
        mean_val   = np.mean(iou_vals)
        median_val = np.median(iou_vals)
        ax.scatter(i, mean_val, color=SC_COLORS[sc], marker='D',
                   s=60, zorder=5, label=f'{sc}  mean={mean_val:.4f}  median={median_val:.4f}')
        ax.text(i + 0.07, mean_val + 0.01, f'{mean_val:.4f}',
                fontsize=8.5, color=SC_COLORS[sc], fontweight='bold')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(SCENARIOS, fontsize=11)
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═════════════════════════════════════════
# 4. HISTOGRAM — DISTRIBUSI IoU PER SAMPEL
# ═════════════════════════════════════════
def plot_iou_histogram(test_data: dict, save_path: str = None):
    """Histogram distribusi IoU per sampel untuk ketiga skenario."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(
        'Histogram Distribusi IoU Per Sampel — Data Testing',
        fontsize=13, fontweight='bold', y=1.02
    )

    bins = np.linspace(0, 1, 21)

    for ax, sc in zip(axes, SCENARIOS):
        iou_vals = [s['iou'] for s in test_data[sc]['per_sample']]
        color    = SC_COLORS[sc]
        mean_v   = np.mean(iou_vals)
        med_v    = np.median(iou_vals)

        ax.hist(iou_vals, bins=bins, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        ax.axvline(mean_v, color='black',  linestyle='--',
                   linewidth=1.5, label=f'Mean={mean_v:.4f}')
        ax.axvline(med_v,  color='orange', linestyle=':',
                   linewidth=1.5, label=f'Median={med_v:.4f}')

        ax.set_title(sc, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('IoU', fontsize=9)
        ax.set_ylabel('Frekuensi' if sc == 'Original' else '', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✅ Disimpan: {save_path}')
    plt.close()


# ═════════════════════════════════════════
# 5. STACKED BAR — KATEGORI KUALITAS IoU
# ═════════════════════════════════════════
def plot_iou_bucket(test_data: dict, save_path: str = None):
    """Stacked bar distribusi kualitas segmentasi berdasarkan bucket IoU."""
    bucket_labels = ['IoU < 0,5', '0,5 ≤ IoU < 0,7',
                     '0,7 ≤ IoU < 0,8', '0,8 ≤ IoU < 0,9', 'IoU ≥ 0,9']
    bucket_colors = ['#EF5350', '#FFA726', '#FFEE58', '#66BB6A', '#1E88E5']

    def get_buckets(iou_vals):
        b = [0] * 5
        for v in iou_vals:
            if   v < 0.5: b[0] += 1
            elif v < 0.7: b[1] += 1
            elif v < 0.8: b[2] += 1
            elif v < 0.9: b[3] += 1
            else:         b[4] += 1
        return b

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        'Distribusi Kualitas Segmentasi per Kategori IoU\nData Testing (n=1.000)',
        fontsize=12, fontweight='bold', y=1.02
    )

    x       = np.arange(len(SCENARIOS))
    width   = 0.55
    bottoms = np.zeros(len(SCENARIOS))

    for bi, (blabel, bcolor) in enumerate(zip(bucket_labels, bucket_colors)):
        vals = [get_buckets([s['iou'] for s in test_data[sc]['per_sample']])[bi]
                for sc in SCENARIOS]
        bars = ax.bar(x, vals, width, bottom=bottoms,
                      label=blabel, color=bcolor, edgecolor='white', linewidth=0.5)
        for bar, val, bot in zip(bars, vals, bottoms):
            if val > 20:
                txt_color = 'white' if bcolor in ['#EF5350', '#1E88E5', '#66BB6A'] else 'black'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bot + val / 2,
                        f'{val}\n({val/10:.0f}%)',
                        ha='center', va='center',
                        fontsize=8, fontweight='bold', color=txt_color)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=11)
    ax.set_ylabel('Jumlah Sampel', fontsize=10)
    ax.set_ylim(0, 1100)
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1))
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
    print('\n── Loading test result files... ──')
    test_data = load_test_results(TEST_FILES)

    if not test_data:
        print('Tidak ada file test result yang ditemukan!')
        exit()

    print('\n── Membuat grafik hasil testing... ──')

    plot_test_metrics_comparison(
        test_data,
        save_path=f'{OUTPUT_DIR}/test_metrics_comparison.png'
    )

    plot_val_vs_test(
        test_data,
        save_path=f'{OUTPUT_DIR}/test_val_comparison.png'
    )

    plot_iou_boxplot(
        test_data,
        save_path=f'{OUTPUT_DIR}/test_iou_boxplot.png'
    )

    plot_iou_histogram(
        test_data,
        save_path=f'{OUTPUT_DIR}/test_iou_histogram.png'
    )

    plot_iou_bucket(
        test_data,
        save_path=f'{OUTPUT_DIR}/test_iou_bucket.png'
    )

    print(f'\n✅ Semua grafik testing disimpan di: {OUTPUT_DIR}')