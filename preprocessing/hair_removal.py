import cv2
import os
import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt
from typing import Union, Tuple, Literal
from pathlib import Path
from tqdm import tqdm

"""
create def(s) that contains of Bothat and Laplacian preprocessing

Bothat:
- convert to grayscale *
- Average image filtering
- Laplacian image filtering *
- Subtract filtered images (average image and laplacian image filtering) 
- Bottom hat transform se0
- Bottom hat transform se45
- Bottom hat transform se90
- Add images (se0 + se45 + se90)
- Image adjustment
- Global image thresholding to obtain binary mask
- Image dilation  *
- Red channel hair pixel replacement using interpolation *
- Green channel hair pixel replacement using interpolation *
- Blue channel hair pixel replacement using interpolation *
- Combine 3 of them to produce final image *

Laplacian:
- convert to grayscale *
- Laplacian image filtering *
- Subtract filtered images (grayscale and laplacian image)
- Noise reduction filtering
- Obtain binary mask using log edge detection
- Morphological operations
- Morphological image closing se0
- Morphological image closing se45
- Morphological image closing se90
- Image dilation *
- Red channel hair pixel replacement using interpolation *
- Green channel hair pixel replacement using interpolation *
- Blue channel hair pixel replacement using interpolation *
- Combine 3 of them to produce final image *
"""

def laplacian_hr(input_data: Union[str, np.ndarray], debug: bool = False, save_debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # read an image

    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise ValueError(f"Failed to read image from {input_data}")
    else:
        img = input_data
    
    # Ensure img is numpy array
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array or image path, got {type(img)}")

    # convert read image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # apply read image into laplacian operation
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_64f = cv2.convertScaleAbs(laplacian)

    # subtract gray and laplacian images
    subtracted = cv2.subtract(gray, laplacian_64f)
    
    # apply wiener filter with 3x3 kernel to reduce subtracted image
    reduced = wiener(subtracted, (3, 3), 0)
    reduced = np.nan_to_num(reduced)
    reduced = np.clip(reduced, 0, 255)
    reduced = reduced.astype(np.uint8)

    # log binary mask
    blur = cv2.GaussianBlur(reduced, (3, 3), 0)
    log_edges = cv2.Laplacian(blur, cv2.CV_64F)
    log_edges_8u = cv2.convertScaleAbs(log_edges)
    _, binary_mask = cv2.threshold(log_edges_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological operation (clean) (bridge) & (diag)
    se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    se_ellip = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_d = np.ones((3, 3), np.uint8)
    morph1 = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, se_ellip) # opening morph
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, se_ellip) # closing morph
    morph3 = cv2.dilate(morph2, kernel_d, iterations=1)

    # se0 = Horizontal, se45 = diagonal, se90 = vertical
    se0  = cv2.getStructuringElement(cv2.MORPH_RECT,    (17, 1))
    se45 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    se90 = cv2.getStructuringElement(cv2.MORPH_RECT,    (1, 17))
    
    # apply se0
    img_se0 = cv2.morphologyEx(morph3, cv2.MORPH_CLOSE, se0)
    img_se45 = cv2.morphologyEx(morph3, cv2.MORPH_CLOSE, se45)
    img_se90 = cv2.morphologyEx(morph3, cv2.MORPH_CLOSE, se90)

    # combine all images
    final_mask = cv2.bitwise_or(cv2.bitwise_or(img_se0, img_se45), img_se90)
    dilated =  cv2.dilate(final_mask, kernel_d, iterations=1)

    # inpainting / restoration
    final_img = cv2.inpaint(img, dilated, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


    if debug:
        # __debugging__([img, final_img, gray, laplacian_64f, reduced, binary_mask, img_clean, img_bridge, img_se0, img_se45, img_se90, img_dilate])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(3, 4, figsize=(20, 10))

        ax[0, 0].imshow(img_rgb)
        ax[0, 0].set_title("original image")
        ax[0, 1].imshow(gray, cmap="gray")
        ax[0, 1].set_title("grayscale")
        ax[0, 2].imshow(laplacian_64f, cmap="gray")
        ax[0, 2].set_title("laplacian")
        ax[0, 3].imshow(subtracted, cmap="gray")
        ax[0, 3].set_title("subtracted")
        ax[1, 0].imshow(reduced, cmap="gray")
        ax[1, 0].set_title("reduced")
        ax[1, 1].imshow(blur, cmap="gray")
        ax[1, 1].set_title("blurred")
        ax[1, 2].imshow(binary_mask, cmap="gray")
        ax[1, 2].set_title("binary")
        ax[1, 3].imshow(morph1, cmap="gray")
        ax[1, 3].set_title("clean")
        ax[2, 0].imshow(morph2, cmap="gray")
        ax[2, 0].set_title("bridge")
        ax[2, 1].imshow(morph3, cmap="gray")
        ax[2, 1].set_title("diag")
        ax[2, 2].imshow(final_mask, cmap="gray")
        ax[2, 2].set_title("final mask")
        ax[2, 3].imshow(final_rgb)
        ax[2, 3].set_title("final image")

        for a in ax.flat:
            a.axis('off')

        plt.tight_layout()
        plt.show()

    if save_debug:
        all_results = [img, final_img, gray, laplacian_64f, reduced, binary_mask, morph1, morph2, morph3, img_se0, img_se45, img_se90, final_mask]
        all_tags = ["original image", "final image", "grayscale", "laplacian 64f", "reduced", "binary mask", "morph clean", "morph bridge", "morph diag", "se0 close", "se45 close", "se90 close", "final mask"]
        debug_dir = Path("/home/reva/G/skripsi/output") / "laplacian"
        debug_dir.mkdir(parents=True, exist_ok=True)
        c = 1
        for result_img, tag in zip(all_results, all_tags):
            if result_img.ndim == 3:
                save_img = result_img
            else:
                save_img = result_img
            
            # Convert to uint8 if needed
            if save_img.dtype != np.uint8:
                save_img = np.clip(save_img, 0, 255).astype(np.uint8)
            
            output_path = debug_dir / f"{c}_{tag.replace(' ', '_')}.png"
            cv2.imwrite(str(output_path), save_img)
            c += 1

    return final_img, morph3

def bothat_hr(input_data: Union[str, np.ndarray], debug: bool = False, save_debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            raise ValueError(f"Failed to read image from {input_data}")
    else:
        img = input_data
    
    # Ensure img is numpy array
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array or image path, got {type(img)}")

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # average image filtering
    kernel_size = [5, 5]
    blurred_img = cv2.blur(gray, kernel_size)

    # laplacian image filtering
    laplacian = cv2.Laplacian(blurred_img, cv2.CV_64F)
    laplacian_64f = cv2.convertScaleAbs(laplacian)

    # subtract average image filtering with laplacian image filtering
    subtracted = cv2.subtract(blurred_img, laplacian_64f)

    # transform se0, se45, se90
    se0 = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    se45 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    se90 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 17))

    img_se0 = cv2.morphologyEx(subtracted, cv2.MORPH_BLACKHAT, se0)
    img_se45 = cv2.morphologyEx(subtracted, cv2.MORPH_BLACKHAT, se45)
    img_se90 = cv2.morphologyEx(subtracted, cv2.MORPH_BLACKHAT, se90)

    # add images
    added = cv2.add(cv2.add(img_se0, img_se45), img_se90)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    adjusted = clahe.apply(added)

    # thresholding using Otsu's method
    _, binary_mask = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # image dilation using a line structuring element of 3 pixel length
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.dilate(binary_mask, kernel_dilate, iterations=1)

    # Red, Green, Blue channel hair pixel replacement using interpolation
    repainted = cv2.inpaint(img, final_mask, 3, cv2.INPAINT_TELEA)

    if debug:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(repainted, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(3, 4, figsize=(20, 10))

        ax[0, 0].imshow(img_rgb)
        ax[0, 0].set_title("original image")
        ax[0, 1].imshow(gray, cmap='gray')
        ax[0, 1].set_title("grayscale image")
        ax[0, 2].imshow(blurred_img, cmap='gray')
        ax[0, 2].set_title("average image filtering")
        ax[0, 3].imshow(laplacian_64f, cmap='gray')
        ax[0, 3].set_title("laplacian image filtering")
        ax[1, 0].imshow(subtracted, cmap='gray')
        ax[1, 0].set_title("subtracted")
        ax[1, 1].imshow(img_se0, cmap='gray')
        ax[1, 1].set_title("se0")
        ax[1, 2].imshow(img_se45, cmap='gray')
        ax[1, 2].set_title("se45")
        ax[1, 3].imshow(img_se90, cmap='gray')
        ax[1, 3].set_title("se90")
        ax[2, 0].imshow(added, cmap='gray')
        ax[2, 0].set_title("se0 + se45 + se90")
        ax[2, 1].imshow(adjusted, cmap='gray')
        ax[2, 1].set_title("CLAHE")
        ax[2, 2].imshow(final_mask, cmap='gray')
        ax[2, 2].set_title("dilation")
        ax[2, 3].imshow(final_rgb)
        ax[2, 3].set_title("result")

        for a in ax.flat:
            a.axis('off')

    if save_debug:
        all_results = [img, repainted, gray, blurred_img, laplacian_64f, subtracted, img_se0, img_se45, img_se90, added, adjusted, binary_mask, final_mask]
        all_tags = ["original image", "final image", "grayscale", "average image","laplacian 64f", "subtracted", "se0", "se45", "se90", "Se0 + se45 + se90", "adjusted", "binary mask", "final mask"]
        debug_dir = Path("/home/reva/G/skripsi/output") / "bothat"
        debug_dir.mkdir(parents=True, exist_ok=True)
        c = 1
        for result_img, tag in zip(all_results, all_tags):
            if result_img.ndim == 3:
                # Color image - keep as is (already BGR from OpenCV)
                save_img = result_img
            else:
                # Grayscale image - keep as is
                save_img = result_img
            
            # Convert to uint8 if needed
            if save_img.dtype != np.uint8:
                save_img = np.clip(save_img, 0, 255).astype(np.uint8)
            
            output_path = debug_dir / f"{c}_{tag.replace(' ', '_')}.png"
            cv2.imwrite(str(output_path), save_img)
            c += 1

    return repainted, final_mask

def apply_bothat_all(input_dir: str, output_dir:str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    images = sorted(f for f in input_path.glob("*.*")
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"])
    
    if len(images) == 0:
        print(f"[WARNING] Gambar gagal ditemukan pada {input_dir}")
        return 

    print(f"\n  Bothat Hair Removal -> {len(images)} gambar...")
    failed = []

    for img_path in tqdm(images, desc="  Bothat HR"):
        try:
            result, _ = bothat_hr(str(img_path), False)
            out_path = output_path / img_path.name
            cv2.imwrite(str(out_path), result) 

        except Exception as e:
            print(f"\n  [SKIP] {img_path.name} → {e}")
            failed.append(img_path.name)
    
    print(f"\n  ✅ Selesai!")
    print(f"     Berhasil : {len(images) - len(failed)}")
    print(f"     Gagal    : {len(failed)}")
    print(f"     Output   : {output_path}\n")

    if failed:
        print("  File yang gagal:")
        for f in failed:
            print(f"    - {f}")

def apply_laplacian_all(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in input_path.glob("*.*")
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"])
    
    if len(images) == 0:
        print(f"[WARNING] Gambar gagal ditemukan pada {input_dir}")
        return 

    print(f"\n  Laplacian Hair Removal -> {len(images)} gambar...")
    failed = []

    for img_path in tqdm(images, desc=" Laplacian HR"):
        try:
            result, _ = laplacian_hr(input_data=str(img_path), debug=False)
            out_path = output_path / img_path.name
            cv2.imwrite(str(out_path), result)
        
        except Exception as e:
            print(f"\n  [SKIP] {img_path.name} → {e}")
            failed.append(img_path.name)
    
    print(f"\n  ✅ Selesai!")
    print(f"     Berhasil : {len(images) - len(failed)}")
    print(f"     Gagal    : {len(failed)}")
    print(f"     Output   : {output_path}\n")

    if failed:
        print("  File yang gagal:")
        for f in failed:
            print(f"    - {f}")            

def apply_hr(method: Literal['laplacian', 'bothat'], input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if method not in ["laplacian", "bothat"]:
        raise ValueError(f"method harus 'bothat' atau 'laplacian, bukan '{method}'")

    images = sorted(f for f in input_path.glob("*.*")
                    if f.suffix.lower() in [".jpg", ".jpeg", ".png"])
    
    if len(images) == 0:
        print(f"[WARNING] Gambar gagal ditemukan pada {input_dir}")
        return 

    print(f"\n  {method} Hair Removal -> {len(images)} gambar...")
    failed = []

    for img_path in tqdm(images, desc=f" {method.capitalize()} HR"):
        try:
            result, _ = laplacian_hr(input_data=str(img_path), debug=False) \
                  if method == 'laplacian' \
                      else bothat_hr(input_data=str(img_path), debug=False)
            out_path = output_path / img_path.name
            cv2.imwrite(str(out_path), result)
        
        except Exception as e:
            print(f"\n  [SKIP] {img_path.name} → {e}")
            failed.append(img_path.name)
    
    print(f"\n  ✅ Selesai!")
    print(f"     Berhasil : {len(images) - len(failed)}")
    print(f"     Gagal    : {len(failed)}")
    print(f"     Output   : {output_path}\n")

    if failed:
        print("  File yang gagal:")
        for f in failed:
            print(f"    - {f}")  

if __name__ =="__main__":
    for split in ['training', 'testing', 'validation']:
        apply_laplacian_all(input_dir=f"processed/1_resize/{split}/images", output_dir=f"processed/2_laplacian/{split}/images")