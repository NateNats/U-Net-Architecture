import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

IMG_SIZE = (256, 256)

def resize_images(input_dir: str, output_dir: str, size: tuple = IMG_SIZE):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted([f for f in input_path.glob('*.*')
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if len(images) == 0:
        print(f"[WARNING] tidak ada gambar di {input_dir}")
        return
    
    print(f"\n  Resize {len(images)} gambar -> {size}...")

    for img_path in tqdm(images, desc=f"    {input_path.name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [SKIP] gambar gagal dibaca: {input_path.name}")
            continue
            
        resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)

        save_path = output_path / img_path.name
        cv2.imwrite(str(save_path), resized)
    
    print(f"    Selesai! gambar berhasil disimpan: {output_path}\n")

def resize_masks(input_dir: str, output_dir: str = None, size: tuple = IMG_SIZE):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    masks = sorted([
        f for f in input_path.glob('*.*')
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])

    if len(masks) == 0:
        print(f'[WARNING] tidak ada mask di dalam: {input_dir}')
        return
    
    print(f"    Resize {len(masks)} gambar -> {size}...")

    for mask_path in tqdm(masks, desc=f"    {input_path.name}"):
        img = cv2.imread(mask_path)
        if img is None:
            print(f"    [SKIP] gambar tidak berhasil dibaca ({mask_path.name})")
            continue
            
        resized = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        save_path = output_path / mask_path.name
        cv2.imwrite(str(save_path), resized)
    
    print(f"    Selesai! Disimpan di {output_path}\n")

if __name__ == "__main__":
    splits = ['training', 'validation', 'testing']

    for split in splits:
        print(f"\n{'='*45}")
        print(f"    split: {split.upper()}")
        print(f"{'='*45}")

        resize_images(input_dir=f"dataset/{split}/images", output_dir=f"dataset/resized/{split}/images")
        resize_masks(input_dir=f"dataset/{split}/masks", output_dir=f"dataset/resized/{split}/masks")
    
    print(f" Semua gambar dan mask berhasil di-resize!\n")