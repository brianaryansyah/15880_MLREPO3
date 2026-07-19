from pathlib import Path
import shutil

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
TEST_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "test"
TEST_LABELS_DIR = PROJECT_ROOT / "dataset" / "labels" / "test"
SAMPLES_DEST_DIR = BASE_DIR / "static" / "images" / "samples"

# Buat folder destinasi jika belum ada
SAMPLES_DEST_DIR.mkdir(parents=True, exist_ok=True)

print("Mencari sampel gambar katarak dan normal...")
cataract_found = None
normal_found = None

# Loop pada file label untuk mencari class_id
if TEST_LABELS_DIR.exists() and TEST_IMAGES_DIR.exists():
    for label_path in TEST_LABELS_DIR.glob("*.txt"):
        try:
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            
            # Ambil class_id dari baris pertama (format YOLO: class_id x_center y_center width height)
            first_line = content.splitlines()[0]
            class_id = int(first_line.split()[0])
            
            # Hubungkan ke file gambar
            img_name_jpg = label_path.stem + ".jpg"
            img_name_png = label_path.stem + ".png"
            
            img_path = None
            if (TEST_IMAGES_DIR / img_name_jpg).exists():
                img_path = TEST_IMAGES_DIR / img_name_jpg
            elif (TEST_IMAGES_DIR / img_name_png).exists():
                img_path = TEST_IMAGES_DIR / img_name_png
                
            if img_path:
                if class_id == 0 and not cataract_found:
                    cataract_found = img_path
                    print(f"✅ Sampel katarak ditemukan: {img_path.name}")
                elif class_id == 1 and not normal_found:
                    normal_found = img_path
                    print(f"✅ Sampel normal ditemukan: {img_path.name}")
                    
            if cataract_found and normal_found:
                break
        except Exception as e:
            continue

# Salin file ke static/images/samples
if cataract_found:
    shutil.copy2(cataract_found, SAMPLES_DEST_DIR / "cataract_sample.jpg")
    print(f"Copied cataract sample to: {SAMPLES_DEST_DIR / 'cataract_sample.jpg'}")
else:
    print("⚠️ Sampel katarak tidak ditemukan!")

if normal_found:
    shutil.copy2(normal_found, SAMPLES_DEST_DIR / "normal_sample.jpg")
    print(f"Copied normal sample to: {SAMPLES_DEST_DIR / 'normal_sample.jpg'}")
else:
    print("⚠️ Sampel normal tidak ditemukan!")
