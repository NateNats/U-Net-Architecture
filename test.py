import os
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.isic_dataset import ISICDataset
from models.u_net import UNet
from utils.metrics import SegmentationMetrics, DiceLoss
from utils.augmentation import get_val_transforms

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

    print("TESTING SELESAI!")
