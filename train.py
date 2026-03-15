import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset.isic_dataset import ISICDataset
from models.u_net import UNet
from utils.metrics import dice_score, iou_score, DiceLoss, BCEDiceLoss, SegmentationMetrics
from utils.augmentation import get_train_transforms, get_val_transforms

CONFIG = {
    'img_size': 256,
    'batch_size': 8,
    'num_workers': 2,
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'patience': 15,
    'scheduler': 'cosine',
    'root_dir': 'dataset',
    'save_dir': 'checkpoints',
}

def train_one_epochs(model, loader, criterion, optimizer, device, metrics_fn) -> dict:
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

        for k, v, in m.item():
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
        loss.backward()

        total_loss += loss.item()
        m = metrics_fn.compute(logits.detach(), masks.detach())
        for k, v, in m.item():
            all_metrics[k] += v

    n = len(loader)

    return {'loss': total_loss / n, **{k:v / n for k, v in all_metrics.items()}}

def plot_history(history: dict, experiment: str, save_path: str = None):
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
            ax.set_title(label, fontsize=14, color='black')
            ax.set_xlabel("Epoch", fontsize=12, color='black')
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
            plt.show()

def train(experiment: str, config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(config['save_dir']) / experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"    Eksperimen  : {experiment.upper()}")
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
    optimizer = optim.adaW(
        model.parameters(),
        lr = config['lr'],
        weight_decay = config['weight_decay']
    )

    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config['patience'])

    metrics_fn = SegmentationMetrics(threshold=0.5)

