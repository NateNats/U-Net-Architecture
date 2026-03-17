import os
import time
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from dataset.isic_dataset import ISICDataset
from models.u_net import UNet
from utils.metrics import dice_score, iou_score, DiceLoss, BCEDiceLoss, SegmentationMetrics
from utils.augmentation import get_train_transforms, get_val_transforms

CONFIG = {
    'img_size': 256,
    'batch_size': 4,
    'num_workers': 2,
    'epochs': 10,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'patience': 15,
    'scheduler': 'cosine',
    'root_dir': 'dataset/resized',
    'save_dir': 'checkpoints',
}

def train_one_epoch(model, loader, criterion, optimizer, device, metrics_fn) -> dict:
    model.train()
    total_loss = 0.0
    all_metrics = {k: 0.0 for k in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity']}

    for images, masks in tqdm(loader, desc=' Train', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        m = metrics_fn.compute(logits.detach(), masks.detach())

        for k, v in m.items():
            all_metrics[k] += v

    n = len(loader)

    return {'loss': total_loss / n, **{k:v / n for k, v in all_metrics.items()}}

@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics_fn, desc: str = "Val") -> dict:
    model.eval()
    total_loss = 0.0
    all_metrics = {k: 0.0 for k in ['dice', 'iou', 'accuracy', 'precision', 'recall', 'specificity']}
    
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item()
        m = metrics_fn.compute(logits.detach(), masks.detach())
        for k, v, in m.items():
            all_metrics[k] += v

    n = len(loader)

    return {'loss': total_loss / n, **{k:v / n for k, v in all_metrics.items()}}

def plot_history(history: dict, experiment: str, save_path: str):
    epochs = range(1, len(history['train']) + 1)
    plots  = [("loss", "Loss"), ("dice", "Dice"), ("iou", "IoU")]
    colors = {'train': 'blue', 'val': 'orange'}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='white')
    fig.suptitle(f"Training History - {experiment}", fontsize=16, color='black')

    for ax, (key, label) in zip(axes, plots):
        ax.set_facecolor('white')
        for split, color in colors.items():
            vals = [ep[key] for ep in history[split]]
            ax.plot(epochs, vals, color=color, label=split.capitalize(), linewidth=1.8)
            ax.set_title(label, fontsize=14, color='white')
            ax.set_xlabel("Epoch", fontsize=12, color='white')
            ax.set_ylabel(label, fontsize=12, color='black')
            ax.tick_params(colors='gray')
            ax.legend(facecolor='#333', labelcolor='white', fontsize=8)
            ax.grid(True, alpha=0.3, color='#555')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#111')
            print(f"Saved training history plot to {save_path}")

def train(experiment: str, config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['save_dir']) / experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"    Experiment  : {experiment.upper()}")
    print(f"    Device      : {device}")
    print(f"    epochs      : {config['epochs']}")
    print(f"    batch size  : {config['batch_size']}")
    print(f"    LR          : {config['lr']}")
    print(f"{'='*55}\n")

    train_ds = ISICDataset(
        split = 'training',
        root_dir = config['root_dir'],
        transform = get_train_transforms()
    )

    val_ds = ISICDataset(
        split = 'validation',
        root_dir = config['root_dir'],
        transform = get_val_transforms()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size = config['batch_size'],
        shuffle = True,
        num_workers = config['num_workers'],
        pin_memory = True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size= config['batch_size'],
        shuffle = False,
        num_workers = config['num_workers'],
        pin_memory = True
    )

    model = UNet(in_channels=3, num_classes=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters for U-Net: {total_params:,}")

    criterion = DiceLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr = config['lr'],
        weight_decay = config['weight_decay']
    )

    if config["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=1e-6
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

    metrics_fn = SegmentationMetrics(threshold=0.5)
    early_stop = EarlyStopping(patience=config['patience'])

    history = {'train': [], 'val': []}
    best_dice = 0.0
    best_epoch = 0

    print(f"    Mulai training...\n")
    for epoch in range(1, config['epochs'] + 1):
        t0 = time.time()

        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device, metrics_fn)
        val_res = evaluate(model, val_loader, criterion, device, metrics_fn)

        if config["scheduler"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_res["loss"])

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(
            f"  Epoch [{epoch:03d}/{config['epochs']}]"
            f"  LR: {lr_now:.1e}"
            f"  | Train → Loss: {train_res['loss']:.4f}"
            f"  Dice: {train_res['dice']:.4f}"
            f"  | Val → Loss: {val_res['loss']:.4f}"
            f"  Dice: {val_res['dice']:.4f}"
            f"  IoU: {val_res['iou']:.4f}"
            f"  | {elapsed:.1f}s"
        )

        history['train'].append(train_res)
        history['val'].append(val_res)

        if val_res['dice'] > best_dice:
            best_dice = val_res['dice']
            best_epoch = epoch
            torch.save({
                "epoch"      : epoch,
                "experiment" : experiment,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_dice"  : best_dice,
                "config"     : config,
            },
            save_dir / "best_model.pth")
            print(f"  ★ Model terbaik disimpan! "
                  f"Epoch {epoch} | Dice: {best_dice:.4f}")
        
        if early_stop(val_res["loss"]):
            print(f"\n  ⚠ Early stopping pada epoch {epoch}.")
            print(f"  Model terbaik: Epoch {best_epoch}"
                  f" | Dice: {best_dice:.4f}")
            break

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  History disimpan: {save_dir}/history.json")

    plot_history(history, experiment, save_path=str(save_dir/"training_curve.png"))

    print(f"Training [{experiment.upper()}] selesai!!")
    print(f"    Best dice: {best_dice:.4f} (epoch {best_epoch})\n")

    return history
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net for segmentation task"
    )

    parser.add_argument(
        "--experiment",
        type = str,
        default = "all",
        choices = ["original", "bothat", "laplacian", "all"],
        help = "Choose experimental mode: original | bothat | laplacian | all"
    )

    parser.add_argument(
        "--epochs",
        type = int,
        default = 50,
        help = "How much epochs do you want??"
    )

    parser.add_argument(
        "--batch_size",
        type = int,
        default = 2,
        help = "Batch size training"
    )

    args = parser.parse_args()

    CONFIG['batch_size'] = args.batch_size
    CONFIG['epochs'] = args.epochs

    if args.experiment == 'all':
        experiments = ['original', 'bothat', 'laplacian']
    else:
        experiments = [args.experiment]

    all_histories = {}
    for exp in experiments:
        history = train(exp, CONFIG)
        all_histories[exp] = history


    print("\n" + "="*55)
    print("  SEMUA TRAINING SELESAI!")
    print("  Hasil checkpoint: checkpoints/<experiment>/best_model.pth")
    print("  Jalankan evaluasi untuk melihat perbandingan.")
    print("="*55) 
