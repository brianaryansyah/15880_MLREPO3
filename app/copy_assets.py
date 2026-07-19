import shutil
from pathlib import Path
import sys

# Reconfigure stdout to handle UTF-8 if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Definisikan path
BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
STATIC_IMAGES_DIR = BASE_DIR / 'static' / 'images'

# Buat folder static/images jika belum ada
STATIC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Definisikan file sumber dan tujuan
assets_map = {
    # EDA
    PROJECT_ROOT / 'runs' / 'resnet50' / 'eda' / 'class_distribution.png': STATIC_IMAGES_DIR / 'eda_class_dist.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'eda' / 'image_dimensions.png': STATIC_IMAGES_DIR / 'eda_img_dims.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'eda' / 'bbox_analysis.png': STATIC_IMAGES_DIR / 'eda_bbox_analysis.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'eda' / 'file_size_distribution.png': STATIC_IMAGES_DIR / 'eda_file_size_dist.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'eda' / 'bbox_area_comparison.png': STATIC_IMAGES_DIR / 'eda_bbox_area_comp.png',
    
    # EVALUATION
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'comparison_chart.png': STATIC_IMAGES_DIR / 'eval_comparison_chart.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'confusion_matrix.png': STATIC_IMAGES_DIR / 'eval_confusion_matrix.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'roc_curve.png': STATIC_IMAGES_DIR / 'eval_roc_curve.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'pr_curve.png': STATIC_IMAGES_DIR / 'eval_pr_curve.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'training_curves.png': STATIC_IMAGES_DIR / 'eval_resnet_training.png',
    PROJECT_ROOT / 'training_curves.png': STATIC_IMAGES_DIR / 'eval_yolov8_training.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'random_forest_feature_importance.png': STATIC_IMAGES_DIR / 'eval_rf_feature_importance.png',
    
    # INTERPRETABILITY
    PROJECT_ROOT / 'runs' / 'resnet50' / 'interpretability' / 'gradcam' / 'gradcam_grid.png': STATIC_IMAGES_DIR / 'interp_gradcam.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'interpretability' / 'lime' / 'lime_grid.png': STATIC_IMAGES_DIR / 'interp_lime.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'interpretability' / 'saliency' / 'saliency_grid.png': STATIC_IMAGES_DIR / 'interp_saliency.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'interpretability' / 'feature_maps' / 'featuremap_grid.png': STATIC_IMAGES_DIR / 'interp_featuremap.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'interpretability' / 'embeddings' / 'pca.png': STATIC_IMAGES_DIR / 'interp_pca.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'interpretability' / 'embeddings' / 'tsne.png': STATIC_IMAGES_DIR / 'interp_tsne.png',
    PROJECT_ROOT / 'runs' / 'resnet50' / 'plots' / 'random_forest_shap_summary.png': STATIC_IMAGES_DIR / 'interp_rf_shap.png',
}

print("Memulai penyalinan aset analisis ke static/images...")
copied_count = 0
for src, dest in assets_map.items():
    if src.exists():
        try:
            shutil.copy2(src, dest)
            print(f"[OK] Berhasil menyalin: {src.name} -> {dest.name}")
            copied_count += 1
        except Exception as e:
            print(f"[ERROR] Gagal menyalin {src.name}: {str(e)}")
    else:
        print(f"[WARN] File tidak ditemukan: {src}")

print(f"Selesai! Berhasil menyalin {copied_count} dari {len(assets_map)} file aset.")
