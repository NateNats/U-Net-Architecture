import cv2
import numpy as np

def resize_image(image_path, target_size=(256, 256)) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan. Cek kembali path/lokasi file.")
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

def resize_mask(mask_path, target_size=(256, 256)) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Mask tidak ditemukan. Cek kembali path/lokasi file.")
    resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return resized_mask