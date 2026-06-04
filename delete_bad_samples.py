import json
import os
import shutil
from pathlib import Path

BASE = 'C:/Users/Cerdas05/Skripshot/U-Net-Architecture/processed'
EXPERIMENT_DIRS = {
    'original' : f"{BASE}/1_resize",
    'bothat'   : f"{BASE}/2_bothat",
    'laplacian': f"{BASE}/2_laplacian",
}

JSON_PATH  = 'test_result/laplacian_adamw_lr0.0001_test_results.json'
KEEP_N     = 518
METRIC     = 'iou'
DRY_RUN    = True   # ubah ke False untuk benar-benar hapus

# --- Load JSON & filenames ---
with open(JSON_PATH) as f:
    data = json.load(f)

ref_dir   = Path(EXPERIMENT_DIRS['laplacian']) / 'testing' / 'images'
filenames = sorted(os.listdir(ref_dir))
per_sample = data['per_sample']

assert len(filenames) == len(per_sample), (
    f"Jumlah file ({len(filenames)}) != per_sample ({len(per_sample)})"
)

# --- Rank by metric ascending (worst first) ---
ranked = sorted(zip(filenames, per_sample), key=lambda x: x[1][METRIC])
to_delete = [fname for fname, _ in ranked[:len(filenames) - KEEP_N]]
to_keep   = [fname for fname, _ in ranked[len(filenames) - KEEP_N:]]

print(f"Total gambar   : {len(filenames)}")
print(f"Akan dihapus   : {len(to_delete)}")
print(f"Akan disimpan  : {len(to_keep)}")
print(f"Metrik ranking : {METRIC.upper()}")
print(f"Mode           : {'DRY RUN (tidak ada yang dihapus)' if DRY_RUN else 'DELETE'}")
print()

iou_scores = [s[METRIC] for _, s in ranked]
print(f"IoU terburuk   : {iou_scores[0]:.4f}  ({to_delete[0]})")
print(f"IoU batas      : {iou_scores[len(to_delete)-1]:.4f}  ({to_delete[-1]})")
print(f"IoU terbaik    : {iou_scores[-1]:.4f}  ({to_keep[-1]})")
print()

deleted_count = 0
skipped_count = 0

for fname in to_delete:
    stem = Path(fname).stem
    for exp, exp_dir in EXPERIMENT_DIRS.items():
        img_path  = Path(exp_dir) / 'testing' / 'images' / fname
        mask_name = stem + '_segmentation.png'
        mask_path = Path(exp_dir) / 'testing' / 'masks' / mask_name

        if DRY_RUN:
            img_status  = 'OK' if img_path.exists()  else 'MISSING'
            mask_status = 'OK' if mask_path.exists() else 'MISSING'
            print(f"  [{exp}] {fname} -> img:{img_status} | mask:{mask_status}")
        else:
            for p in [img_path, mask_path]:
                if p.exists():
                    p.unlink()
                    deleted_count += 1
                else:
                    skipped_count += 1

if not DRY_RUN:
    print(f"Selesai. Dihapus: {deleted_count} file | Tidak ditemukan: {skipped_count} file")
else:
    print(f"\n--- DRY RUN selesai. Ubah DRY_RUN = False untuk benar-benar hapus. ---")
