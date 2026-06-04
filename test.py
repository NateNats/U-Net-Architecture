import os
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.isic_dataset import ISICDataset
from models.u_net import UNet
from utils.metrics import SegmentationMetrics, DiceLoss
from utils.augmentation import get_val_transforms

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

SCENARIO_LABELS = {
    'original' : 'Original',
    'bothat'   : 'Bot Hat',
    'laplacian': 'Laplacian',
}
SCENARIO_COLORS = {
    'Original' : '#1565C0',
    'Bot Hat'  : '#E65100',
    'Laplacian': '#2E7D32',
}
OPTIMIZER_COLORS = {
    'adamw': '#5C6BC0',
    'lion' : '#EF6C00',
}

BASE = 'C:/Users/Cerdas05/Skripshot/U-Net-Architecture/processed'

EXPERIMENT_DIRS = {
    'original' : f"{BASE}/1_resize",
    'bothat'   : f"{BASE}/2_bothat",
    'laplacian': f"{BASE}/2_laplacian",
}


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_test(model, loader, criterion, device, metrics_fn):
    total_loss = 0.0
    agg = {k: 0.0 for k in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity']}
    per_sample = []

    for images, masks in tqdm(loader, desc='  Test', leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(images)
        total_loss += criterion(logits, masks).item()

        for i in range(images.size(0)):
            m = metrics_fn.compute(logits[i:i+1].detach(), masks[i:i+1].detach())
            per_sample.append({k: round(v, 6) for k, v in m.items()})

        batch_m = metrics_fn.compute(logits.detach(), masks.detach())
        for k, v in batch_m.items():
            agg[k] += v

    n = len(loader)
    aggregate = {'loss': total_loss / n, **{k: v / n for k, v in agg.items()}}
    return aggregate, per_sample


def _infer_experiment(checkpoint_path: str, ckpt: dict) -> str:
    experiment = ckpt.get('experiment') or ckpt.get('config', {}).get('experiment', '')
    if experiment in EXPERIMENT_DIRS:
        return experiment
    for exp in EXPERIMENT_DIRS:
        if exp in Path(checkpoint_path).parent.name.lower():
            return exp
    return ''


def run_test(checkpoint_path: str, output_dir: str, batch_size: int) -> tuple:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, ckpt = load_checkpoint(checkpoint_path, device)
    experiment  = _infer_experiment(checkpoint_path, ckpt)
    run_name    = Path(checkpoint_path).parent.name

    root_dir = EXPERIMENT_DIRS.get(experiment)
    if not root_dir:
        raise ValueError(
            f"Experiment '{experiment}' tidak dikenal. "
            f"Isi field 'experiment' di checkpoint atau pastikan nama folder mengandung "
            f"salah satu dari: {list(EXPERIMENT_DIRS.keys())}"
        )

    test_ds = ISICDataset(
        split    = 'test',
        root_dir = root_dir,
        transform = get_val_transforms(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
    )

    criterion  = DiceLoss()
    metrics_fn = SegmentationMetrics(threshold=0.5)

    print(f"\n{'='*58}")
    print(f"  Run        : {run_name}")
    print(f"  Experiment : {experiment.upper()}")
    print(f"  Test set   : {len(test_ds)} gambar")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : epoch {ckpt.get('epoch', '?')} "
          f"| train best_dice: {ckpt.get('best_dice', 0):.4f}")
    print(f"{'='*58}\n")

    aggregate, per_sample = evaluate_test(model, test_loader, criterion, device, metrics_fn)

    print(f"  Hasil Testing:")
    for k, v in aggregate.items():
        print(f"    {k:<12}: {v:.4f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result = {
        'run'             : run_name,
        'checkpoint_path' : checkpoint_path,
        'experiment'      : experiment,
        'epoch'           : ckpt.get('epoch'),
        'train_best_dice' : ckpt.get('best_dice'),
        'config'          : ckpt.get('config', {}),
        'test_aggregate'  : {k: round(v, 6) for k, v in aggregate.items()},
        'per_sample'      : per_sample,
        'tested_at'       : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = out_path / f"{run_name}_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  Disimpan : {json_path}")

    # ── Grafik per run ────────────────────────────────────────
    print(f"\n  Membuat grafik...")

    plot_per_sample_boxplot(
        per_sample, run_name,
        save_path=str(out_path / f"{run_name}_boxplot.png"),
    )
    plot_sample_predictions(
        model, test_ds, device,
        save_path=str(out_path / f"{run_name}_predictions.png"),
        n_samples=8, per_sample=per_sample,
    )

    ckpt_meta = {
        'experiment'      : experiment,
        'epoch'           : ckpt.get('epoch'),
        'train_best_dice' : ckpt.get('best_dice', 0),
        'config'          : ckpt.get('config', {}),
    }
    return run_name, aggregate, ckpt_meta


def _parse_run_name(run_name: str) -> tuple:
    """Ekstrak optimizer dan lr dari nama run, misal: original_adamw_lr0.001"""
    parts = run_name.split('_')
    lr_val, opt_val = '', ''
    for i, p in enumerate(parts):
        if p.startswith('lr'):
            lr_val  = p[2:]
            opt_val = parts[i - 1] if i > 0 else ''
            break
    return opt_val, lr_val


def save_summary_csv(all_results: list, output_dir: str):
    csv_path = Path(output_dir) / "test_summary.csv"
    fieldnames = [
        'run', 'experiment', 'optimizer', 'lr', 'epoch',
        'train_best_dice', 'loss', 'dice', 'iou',
        'accuracy', 'precision', 'recall', 'specificity',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, agg, meta in all_results:
            opt_val, lr_val = _parse_run_name(run_name)
            writer.writerow({
                'run'             : run_name,
                'experiment'      : meta['experiment'],
                'optimizer'       : opt_val,
                'lr'              : lr_val,
                'epoch'           : meta['epoch'],
                'train_best_dice' : f"{meta['train_best_dice']:.4f}",
                'loss'            : f"{agg['loss']:.4f}",
                'dice'            : f"{agg['dice']:.4f}",
                'iou'             : f"{agg['iou']:.4f}",
                'accuracy'        : f"{agg['accuracy']:.4f}",
                'precision'       : f"{agg['precision']:.4f}",
                'recall'          : f"{agg['recall']:.4f}",
                'specificity'     : f"{agg['specificity']:.4f}",
            })

    print(f"\n  CSV ringkasan disimpan: {csv_path}")
    _print_summary_table(all_results)


def _print_summary_table(all_results: list):
    print('\n' + '=' * 90)
    print(f"  {'Run':<35} {'Dice':>8} {'IoU':>8} {'Recall':>8} {'Precision':>10} {'Loss':>8}")
    print('  ' + '-' * 84)
    for run_name, agg, _ in sorted(all_results, key=lambda x: x[1]['iou'], reverse=True):
        print(
            f"  {run_name:<35}"
            f"  {agg['dice']*100:>7.2f}%"
            f"  {agg['iou']*100:>7.2f}%"
            f"  {agg['recall']*100:>7.2f}%"
            f"  {agg['precision']*100:>9.2f}%"
            f"  {agg['loss']:>8.4f}"
        )
    print('=' * 90 + '\n')


# ─────────────────────────────────────────────────────────────
# GRAFIK 1: Visualisasi prediksi (image | GT | prediksi)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def plot_sample_predictions(model, dataset, device, save_path: str,
                            n_samples: int = 8, per_sample: list = None):
    """
    Tampilkan n_samples gambar: input | ground truth | prediksi.
    Jika per_sample tersedia, pilih kasus terbaik, median, dan terburuk
    berdasarkan Dice score.
    """
    model.eval()
    total = len(dataset)

    if per_sample and len(per_sample) == total:
        dices     = [s['dice'] for s in per_sample]
        sorted_idx = sorted(range(total), key=lambda i: dices[i])
        n_each    = max(1, n_samples // 3)
        worst_idx  = sorted_idx[:n_each]
        best_idx   = sorted_idx[-n_each:]
        mid        = total // 2
        median_idx = sorted_idx[mid: mid + (n_samples - 2 * n_each)]
        indices    = worst_idx + median_idx + best_idx
        labels     = (
            [f'Terburuk\nDice={dices[i]:.3f}' for i in worst_idx] +
            [f'Median\nDice={dices[i]:.3f}'   for i in median_idx] +
            [f'Terbaik\nDice={dices[i]:.3f}'  for i in best_idx]
        )
    else:
        step    = max(1, total // n_samples)
        indices = list(range(0, min(n_samples * step, total), step))[:n_samples]
        labels  = [f'#{i}' for i in indices]

    n_cols = len(indices)
    fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 2.8, 8))
    fig.suptitle('Hasil Prediksi U-Net pada Test Set\n'
                 '(Baris: Input | Ground Truth | Prediksi)',
                 fontsize=13, fontweight='bold', y=1.01)

    row_titles = ['Input', 'Ground Truth', 'Prediksi']
    for row in range(3):
        axes[row, 0].set_ylabel(row_titles[row], fontsize=10, fontweight='bold')

    for col, (idx, lbl) in enumerate(zip(indices, labels)):
        image, mask = dataset[idx]

        logits = model(image.unsqueeze(0).to(device))
        pred   = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()

        # Denormalisasi gambar
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * _IMAGENET_STD + _IMAGENET_MEAN).clip(0, 1)

        mask_np = mask.squeeze().numpy()

        axes[0, col].imshow(img_np)
        axes[1, col].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[2, col].imshow(pred,    cmap='gray', vmin=0, vmax=1)

        axes[0, col].set_title(lbl, fontsize=8)
        for row in range(3):
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan : {save_path}")


# ─────────────────────────────────────────────────────────────
# GRAFIK 2: Box plot distribusi skor per gambar
# ─────────────────────────────────────────────────────────────
def plot_per_sample_boxplot(per_sample: list, run_name: str, save_path: str):
    """
    Box plot distribusi metrik per gambar pada test set.
    Berguna untuk melihat sebaran dan outlier.
    """
    metric_keys = ['dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity']
    metric_lbls = ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall', 'Specificity']

    data = [[s[k] for s in per_sample] for k in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='red', linewidth=2))

    colors = ['#42A5F5', '#66BB6A', '#FFA726', '#AB47BC', '#26C6DA', '#EC407A']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Anotasi median di atas setiap box
    for i, d in enumerate(data, start=1):
        med = float(np.median(d))
        ax.text(i, med + 0.005, f'{med:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='darkred')

    ax.set_xticks(range(1, len(metric_keys) + 1))
    ax.set_xticklabels(metric_lbls, fontsize=10)
    ax.set_ylabel('Nilai', fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title(f'Distribusi Metrik Per Gambar — Test Set\n({run_name})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan : {save_path}")


# ─────────────────────────────────────────────────────────────
# GRAFIK 3: Bar chart perbandingan semua run (mode checkpoint_dir)
# ─────────────────────────────────────────────────────────────
def plot_all_runs_bar(all_results: list, save_path: str):
    """
    Bar chart IoU dan Dice untuk semua run, diurutkan dari terbaik.
    """
    sorted_res  = sorted(all_results, key=lambda x: x[1]['iou'], reverse=True)
    run_names   = [r[0] for r in sorted_res]
    iou_vals    = [r[1]['iou']  for r in sorted_res]
    dice_vals   = [r[1]['dice'] for r in sorted_res]
    experiments = [r[2]['experiment'] for r in sorted_res]

    x     = np.arange(len(run_names))
    width = 0.38
    colors = [SCENARIO_COLORS.get(SCENARIO_LABELS.get(e, e), '#90A4AE')
              for e in experiments]

    fig, ax = plt.subplots(figsize=(max(10, len(run_names) * 1.1), 5))
    bars_iou  = ax.bar(x - width/2, iou_vals,  width, label='IoU',  alpha=0.85,
                       color=colors, edgecolor='white', linewidth=0.8)
    bars_dice = ax.bar(x + width/2, dice_vals, width, label='Dice', alpha=0.55,
                       color=colors, edgecolor='white', linewidth=0.8, hatch='//')

    for bar, val in zip(bars_iou, iou_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)
    for bar, val in zip(bars_dice, dice_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    # Legend skenario warna
    patches = [mpatches.Patch(color=c, label=SCENARIO_LABELS.get(e, e))
               for e, c in zip(
                   ['original', 'bothat', 'laplacian'],
                   ['#1565C0', '#E65100', '#2E7D32']
               )]
    patches += [
        mpatches.Patch(facecolor='gray', alpha=0.85, label='IoU'),
        mpatches.Patch(facecolor='gray', alpha=0.55, hatch='//', label='Dice'),
    ]
    ax.legend(handles=patches, fontsize=8, loc='lower right')

    ax.set_xticks(x)
    ax.set_xticklabels(run_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Nilai', fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title('Perbandingan IoU dan Dice — Semua Konfigurasi (Test Set)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan : {save_path}")


# ─────────────────────────────────────────────────────────────
# GRAFIK 4: Heatmap IoU per skenario × optimizer × LR
# ─────────────────────────────────────────────────────────────
def plot_heatmap(all_results: list, metric: str, save_path: str):
    """
    Heatmap nilai metrik (misal IoU atau Dice) per konfigurasi.
    Baris = Skenario × Optimizer, Kolom = Learning Rate.
    """
    scenarios  = ['original', 'bothat', 'laplacian']
    optimizers = ['adamw', 'lion']
    lrs        = ['0.01', '0.001', '0.0001']

    row_labels = [f'{SCENARIO_LABELS[s]} — {o.upper()}'
                  for s in scenarios for o in optimizers]
    matrix     = np.full((len(row_labels), len(lrs)), np.nan)

    for run_name, agg, meta in all_results:
        opt_val, lr_val = _parse_run_name(run_name)
        exp = meta['experiment']
        try:
            row = [s for s in scenarios for o in optimizers].index(exp) * 1
            row = next(
                i for i, (s, o) in enumerate(
                    [(s, o) for s in scenarios for o in optimizers]
                ) if s == exp and o == opt_val.lower()
            )
            col = lrs.index(lr_val)
            matrix[row, col] = agg[metric]
        except (ValueError, StopIteration):
            pass

    valid = matrix[~np.isnan(matrix)]
    if valid.size == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=valid.min() - 0.01, vmax=valid.max() + 0.01)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric.upper(), fontsize=10)

    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels(lrs, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_title(f'Heatmap {metric.upper()} Test Set\n(Skenario × Optimizer × LR)',
                 fontsize=12, fontweight='bold')

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            if not np.isnan(val):
                norm_val  = (val - valid.min()) / (valid.max() - valid.min() + 1e-9)
                txt_color = 'white' if norm_val > 0.6 else 'black'
                ax.text(c, r, f'{val:.4f}', ha='center', va='center',
                        fontsize=9, fontweight='bold', color=txt_color)

    for y in [1.5, 3.5]:
        ax.axhline(y=y, color='white', linewidth=2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Disimpan : {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluasi U-Net pada test set')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--checkpoint', type=str,
        help='Path ke satu file best_model.pth',
    )
    group.add_argument(
        '--checkpoint_dir', type=str,
        help='Path ke folder checkpoint; semua subdirektori dengan best_model.pth akan diuji',
    )

    parser.add_argument('--output_dir',  type=str, default='test_results',
                        help='Folder untuk menyimpan hasil (default: test_results/)')
    parser.add_argument('--batch_size',  type=int, default=4)

    args = parser.parse_args()

    all_results = []

    if args.checkpoint:
        run_name, agg, meta = run_test(args.checkpoint, args.output_dir, args.batch_size)
        all_results.append((run_name, agg, meta))
        _print_summary_table(all_results)

    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints    = sorted(checkpoint_dir.rglob('best_model.pth'))

        if not checkpoints:
            print(f"Tidak ada best_model.pth ditemukan di: {checkpoint_dir}")
            raise SystemExit(1)

        print(f"Ditemukan {len(checkpoints)} checkpoint.\n")

        for ckpt_path in checkpoints:
            try:
                run_name, agg, meta = run_test(str(ckpt_path), args.output_dir, args.batch_size)
                all_results.append((run_name, agg, meta))
            except Exception as e:
                print(f"  SKIP {ckpt_path.parent.name}: {e}")

        if all_results:
            save_summary_csv(all_results, args.output_dir)

            print(f"\n  Membuat grafik perbandingan semua run...")
            out_path = Path(args.output_dir)
            plot_all_runs_bar(
                all_results,
                save_path=str(out_path / "comparison_bar.png"),
            )
            plot_heatmap(
                all_results, metric='iou',
                save_path=str(out_path / "heatmap_iou.png"),
            )
            plot_heatmap(
                all_results, metric='dice',
                save_path=str(out_path / "heatmap_dice.png"),
            )

    print("\nTESTING SELESAI!")
    print(f"  Semua output tersimpan di: {args.output_dir}/")
