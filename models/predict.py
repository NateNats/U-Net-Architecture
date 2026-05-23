import cv2
import os
import numpy as np
import torch
from utils import get_val_transforms
from utils import SegmentationMetrics

def predict(model, img_path: str, mask_path: str, device, save_path: str = None):
    img_name = img_path.split("/")[-1]
    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'File in path {img_path} is not found.')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transforms = get_val_transforms()
    img_t = transforms(image=img_rgb)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        pred = (torch.sigmoid(logits) >= 0.5).squeeze().cpu().numpy().astype(np.uint8)
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        success = cv2.imwrite(f"{save_path}/{img_name}", pred * 255)
        if not success:
            raise ValueError(f"gagal menyimpan gambar ke {save_path}/{img_name}")

    if mask_path is None:
        return pred
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"{mask_path} is not found")
    
    mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    metrics = SegmentationMetrics(threshold=0.5).compute(logits, mask_t)

    return pred, metrics