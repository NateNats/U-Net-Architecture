import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from typing import Optional, Callable
import albumentations as A
from utils.augmentation import get_train_transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ISICDataset(Dataset):
    def __init__(self, split: str, # train, val, test
                 root_dir: str="dataset",
                 transform = get_train_transforms()):
        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.transform = transform

        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = cv2.imread(image_path)
        assert image is not None, f"Gagal membaca: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Gagal membaca: {mask_path}"
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.unsqueeze(0)

        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask