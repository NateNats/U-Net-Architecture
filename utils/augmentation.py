import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_transforms():
    compose = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5,
                                   brightness_limit=0.1, 
                                   contrast_limit=0.1),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return compose

def get_val_transforms():
    compose = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return compose