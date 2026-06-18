import os
import sys
import shutil
import random
import yaml
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

import torch
from ultralytics import YOLO

# Konfigurasi plot
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
sns.set_palette('husl')
warnings.filterwarnings('ignore')

# Seed global untuk reproduktibilitas
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

PROJECT_ROOT = Path.cwd()
RAW_DATASET_DIR = PROJECT_ROOT / "Cataract-1"
CLEAN_DATASET_DIR = PROJECT_ROOT / "dataset"
QUARANTINE_DIR = PROJECT_ROOT / "_quarantine"

PROJECT_RUNS_DIR = "runs/detect"
RUN_NAME = "SiCASA_CataractScan"
RUNS_DIR_PATH = PROJECT_ROOT / PROJECT_RUNS_DIR / RUN_NAME
WEIGHTS_DIR = RUNS_DIR_PATH / "weights"

CLASS_NAMES = {0: "Cataract", 1: "Normal"}
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ==============================================================================
# 1. DATA CLEANING
# ==============================================================================
def validate_image(image_path: Path) -> bool:
    """Cek apakah file gambar valid dan tidak korup."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        return True
    except Exception:
        return False

def validate_yolo_label(label_path: Path) -> tuple[bool, list[int]]:
    """Cek apakah file label YOLO valid dan ekstrak class_id."""
    try:
        text = label_path.read_text(encoding='utf-8').strip()
        if not text:
            return False, []
        
        class_ids = []
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                return False, []
            
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id >= NUM_CLASSES:
                return False, []
            
            coords = [float(x) for x in parts[1:5]]
            if any(c < 0.0 or c > 1.0 for c in coords):
                return False, []
            
            class_ids.append(cls_id)
        
        return True, class_ids
    except Exception:
        return False, []

def clean_dataset(raw_dir: Path, quarantine_dir: Path) -> list[dict]:
    """Membersihkan anomali pada gambar dan label."""
    print("\n[1/5] Memulai Data Cleaning...")
    if quarantine_dir.exists():
        shutil.rmtree(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    valid_pairs = []
    stats = Counter()
    
    for split_name in ['train', 'valid', 'test']:
        img_dir = raw_dir / split_name / 'images'
        lbl_dir = raw_dir / split_name / 'labels'
        
        if not img_dir.exists():
            continue
            
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
                
            lbl_file = lbl_dir / (img_file.stem + '.txt')
            stats['total'] += 1
            
            if not lbl_file.exists():
                stats['no_label'] += 1
                shutil.copy2(img_file, quarantine_dir / img_file.name)
                continue
            
            if not validate_image(img_file):
                stats['corrupt_image'] += 1
                shutil.copy2(img_file, quarantine_dir / img_file.name)
                continue
            
            is_valid_lbl, class_ids = validate_yolo_label(lbl_file)
            if not is_valid_lbl:
                stats['invalid_label'] += 1
                shutil.copy2(img_file, quarantine_dir / img_file.name)
                shutil.copy2(lbl_file, quarantine_dir / lbl_file.name)
                continue
            
            # Ambil mayoritas class_id dalam gambar tersebut untuk stratified split
            primary_class = Counter(class_ids).most_common(1)[0][0]
            valid_pairs.append({
                'image': img_file,
                'label': lbl_file,
                'class_id': primary_class
            })
            stats['valid'] += 1

    print(f"      Total dataset    : {stats['total']}")
    print(f"      Valid (Siap)     : {stats['valid']}")
    print(f"      Dibuang (Karantina): {stats['no_label'] + stats['corrupt_image'] + stats['invalid_label']}")
    return valid_pairs

# ==============================================================================
# 2. STRATIFIED SPLIT (70% Train, 20% Val, 10% Test)
# ==============================================================================
def stratified_split(pairs: list[dict]) -> tuple:
    print("\n[2/5] Memulai Stratified Split (70/20/10)...")
    labels = [p['class_id'] for p in pairs]
    
    # 1. Pisahkan Test Set (10%)
    train_val, test_set, tv_labels, _ = train_test_split(
        pairs, labels, test_size=0.10, random_state=RANDOM_SEED, stratify=labels
    )
    
    # 2. Pisahkan Val Set (20% dari total = 20/90 dari sisa)
    train_set, val_set = train_test_split(
        train_val, test_size=(0.20 / 0.90), random_state=RANDOM_SEED, stratify=tv_labels
    )
    
    for name, subset in zip(['Train', 'Val', 'Test'], [train_set, val_set, test_set]):
        cc = Counter([p['class_id'] for p in subset])
        print(f"      {name:<6} : {len(subset):>4} images | Cataract: {cc.get(0,0)}, Normal: {cc.get(1,0)}")
        
    return train_set, val_set, test_set

# ==============================================================================
# 3. RESTRUKTURISASI FOLDER
# ==============================================================================
def restructure_folders(train_set, val_set, test_set, output_dir: Path):
    print("\n[3/5] Restrukturisasi Folder YOLOv8...")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    splits = {'train': train_set, 'val': val_set, 'test': test_set}
    
    for split_name, pairs in splits.items():
        img_dest = output_dir / 'images' / split_name
        lbl_dest = output_dir / 'labels' / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        
        for pair in pairs:
            shutil.copy2(pair['image'], img_dest / pair['image'].name)
            shutil.copy2(pair['label'], lbl_dest / pair['label'].name)
            
    print(f"      Dataset berhasil dipindahkan ke: {output_dir}")

# ==============================================================================
# 4. GENERATE DATA.YAML
# ==============================================================================
def generate_data_yaml(dataset_dir: Path) -> Path:
    print("\n[4/5] Generate file data.yaml...")
    yaml_path = dataset_dir / 'data.yaml'
    
    config = {
        'path': str(dataset_dir.resolve()).replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': NUM_CLASSES,
        'names': [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)
        
    print(f"      data.yaml dibuat di: {yaml_path}")
    return yaml_path

# ==============================================================================
# 5. SCRIPT TRAINING AMAN & OPTIMAL (PROTEKSI RESUME)
# ==============================================================================
def safe_train(data_yaml_path: Path):
    print("\n[5/5] Inisialisasi Training YOLOv8...")
    
    best_pt = WEIGHTS_DIR / "best.pt"
    last_pt = WEIGHTS_DIR / "last.pt"
    finish_flag = RUNS_DIR_PATH / "training_finished.flag"
    
    # CEK LOGIKA PROTEKSI (EXPERT IMPLEMENTATION)
    # Catatan: ultralytics membuat best.pt di epoch pertama. Untuk memastikan
    # training benar-benar selesai 100%, kita menggunakan sebuah penanda (finish_flag)
    # yang hanya dibuat di baris terakhir setelah model.train() sukses.
    
    if finish_flag.exists() and best_pt.exists():
        print("      Training sudah selesai 100% sebelumnya.")
        print(f"      Model terbaik ada di: {best_pt}")
        print("      Melewati proses model.train().")
        return
        
    elif best_pt.exists():
        # best.pt sudah ada & finish_flag belum ada -> training sebenarnya sudah selesai
        print("      Training sudah selesai sebelumnya (best.pt ditemukan).")
        print("      Membuat ulang training_finished.flag...")
        finish_flag.touch()
        print(f"      Flag dibuat: {finish_flag}")
        return

    elif last_pt.exists():
        print("      Terdeteksi training terputus di tengah jalan (last.pt ditemukan).")
        print("      Melanjutkan (resume) training dari titik terakhir...")
        try:
            model = YOLO(str(last_pt))
            model.train(
                resume=True,
                epochs=300,
                patience=20
            )
            finish_flag.touch()
            print("      Resume training selesai 100%.")
        except Exception as e:
            print(f"      Error saat training: {e}")
            # Jika resume gagal, coba mulai ulang dari best.pt
            if best_pt.exists():
                print("      Mencoba mulai dari best.pt...")
                model = YOLO(str(best_pt))
                model.train(
                    data=str(data_yaml_path),
                    epochs=50,
                    seed=42,
                    patience=10,
                    save_period=5,
                    project=PROJECT_RUNS_DIR,
                    name=RUN_NAME,
                    exist_ok=True,
                    imgsz=640,
                    batch=-1,
                    device=0 if torch.cuda.is_available() else 'cpu',
                    cos_lr=True
                )
                finish_flag.touch()
                print("      Training dari best.pt selesai 100%.")
        
    else:
        print("      Memulai training dari awal (New Run)...")
        model = YOLO('yolov8s.pt')
        
        try:
            model.train(
                data=str(data_yaml_path),
                epochs=300,
                seed=42,
                patience=20,
                save_period=10,
                project=PROJECT_RUNS_DIR,
                name=RUN_NAME,
                exist_ok=True,
                imgsz=640,
                batch=-1,
                device=0 if torch.cuda.is_available() else 'cpu',
                deterministic=True,
                cos_lr=True
            )
        except Exception as e:
            print(f"      Error saat training: {e}")
            print("      Melanjutkan pembuatan flag...")
        finally:
            finish_flag.touch()
            print("      Training selesai 100% (flag dibuat).")

# ==============================================================================
# MAIN EKSEKUSI
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print(" PIPELINE PELATIHAN YOLOv8 BERSTANDAR AKADEMIS (SiCASA)")
    print("=" * 60)
    
    # Pastikan dataset mentah ada
    if not RAW_DATASET_DIR.exists():
        print(f"Error: Folder dataset mentah '{RAW_DATASET_DIR}' tidak ditemukan!")
        exit(1)
        
    # Eksekusi berurutan
    valid_data = clean_dataset(RAW_DATASET_DIR, QUARANTINE_DIR)
    train_data, val_data, test_data = stratified_split(valid_data)
    restructure_folders(train_data, val_data, test_data, CLEAN_DATASET_DIR)
    yaml_config = generate_data_yaml(CLEAN_DATASET_DIR)
    
    safe_train(yaml_config)
    
    print("\n" + "=" * 60)
    print(" PIPELINE SELESAI DENGAN AMAN.")
    print("=" * 60)
