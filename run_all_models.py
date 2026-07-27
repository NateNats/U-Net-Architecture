"""
Jalankan evaluasi (dice, iou, pixel_error, rand_error, warping_error)
untuk semua 18 model di folder all_model/, simpan hasilnya ke test_result_revisi/.

Cara pakai (di Anaconda Prompt):
    conda activate torch-pip
    cd "c:\\Users\\Cerdas05\\Skripshot\\U-Net-Architecture"
    python run_all_models.py
"""
from pathlib import Path

from test import run_test, save_summary_csv, plot_all_runs_bar, plot_heatmap

CHECKPOINT_DIR = 'all_model'
OUTPUT_DIR     = 'test_result_revisi_1'
BATCH_SIZE     = 16


def main():
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoints    = sorted(checkpoint_dir.rglob('best_model.pth'))

    if not checkpoints:
        print(f"Tidak ada best_model.pth ditemukan di: {checkpoint_dir}")
        raise SystemExit(1)

    print(f"Ditemukan {len(checkpoints)} checkpoint.\n")

    all_results = []
    for ckpt_path in checkpoints:
        try:
            run_name, agg, meta = run_test(str(ckpt_path), OUTPUT_DIR, BATCH_SIZE)
            all_results.append((run_name, agg, meta))
        except Exception as e:
            print(f"  SKIP {ckpt_path.parent.name}: {e}")

    if all_results:
        save_summary_csv(all_results, OUTPUT_DIR)

        print(f"\n  Membuat grafik perbandingan semua run...")
        out_path = Path(OUTPUT_DIR)
        plot_all_runs_bar(all_results, save_path=str(out_path / "comparison_bar.png"))
        plot_heatmap(all_results, metric='iou',  save_path=str(out_path / "heatmap_iou.png"))
        plot_heatmap(all_results, metric='dice', save_path=str(out_path / "heatmap_dice.png"))

    print("\nTESTING SELESAI!")
    print(f"  Semua output tersimpan di: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
