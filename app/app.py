"""
DEDIKAT - Sistem Deteksi Dini Katarak
Flask Web Application Backend
"""

import os
import sys
import uuid
import time
import base64
import logging
from pathlib import Path
from io import BytesIO

from flask import (
    Flask, render_template, request,
    jsonify, send_from_directory, url_for
)
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

# ──────────────────────────────────────────────
# Konfigurasi App
# ──────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'dedikat-secret-key-2026'

BASE_DIR       = Path(__file__).parent
UPLOAD_FOLDER  = BASE_DIR / 'static' / 'uploads'
MODEL_PATH     = BASE_DIR / 'best.pt'
ALLOWED_EXTS   = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER']      = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DEDIKAT')

# ──────────────────────────────────────────────
# Load Model YOLOv8
# ──────────────────────────────────────────────
model = None
MODEL_LOADED = False
MODEL_STATUS = "Belum dimuat"

def load_model():
    global model, MODEL_LOADED, MODEL_STATUS
    try:
        from ultralytics import YOLO
        if MODEL_PATH.exists():
            logger.info(f"Loading model dari: {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            MODEL_LOADED = True
            MODEL_STATUS = f"✅ Model dimuat: {MODEL_PATH.name}"
            logger.info("Model berhasil dimuat!")
        else:
            MODEL_STATUS = f"⚠️ Model belum tersedia (best.pt tidak ditemukan). Jalankan training dulu!"
            logger.warning(f"Model tidak ditemukan di: {MODEL_PATH}")
    except Exception as e:
        MODEL_STATUS = f"❌ Error memuat model: {str(e)}"
        logger.error(f"Error load model: {e}")

load_model()

# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS

def cleanup_old_files(max_files: int = 50):
    """Hapus file lama jika upload folder terlalu penuh."""
    files = sorted(UPLOAD_FOLDER.glob('*'), key=os.path.getmtime)
    if len(files) > max_files:
        for f in files[:len(files) - max_files]:
            try:
                f.unlink()
            except Exception:
                pass

def image_to_base64(img_array_bgr: np.ndarray) -> str:
    """Konversi numpy array BGR ke base64 string."""
    img_rgb = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def run_inference(image_path: str, conf: float = 0.25, iou: float = 0.45) -> dict:
    """Jalankan inferensi YOLOv8 pada gambar."""
    if not MODEL_LOADED or model is None:
        return {
            'success': False,
            'error': 'Model belum dimuat. Silakan training model terlebih dahulu.',
            'model_status': MODEL_STATUS
        }
    
    try:
        start_time = time.time()
        
        # Run inference
        import torch
        device = 0 if torch.cuda.is_available() else 'cpu'
        results = model(
            image_path,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )
        
        inference_time = (time.time() - start_time) * 1000  # ms
        result = results[0]
        
        # Gambar hasil deteksi
        result_img = result.plot()  # BGR numpy array
        result_b64 = image_to_base64(result_img)
        
        # Parse deteksi
        detections = []
        class_names = result.names
        
        for box in result.boxes:
            cls_id  = int(box.cls)
            conf_sc = float(box.conf)
            xyxy    = box.xyxy[0].tolist()
            cls_name = class_names.get(cls_id, f'Class_{cls_id}')
            
            # Tentukan severity
            if conf_sc >= 0.8:
                severity = 'Tinggi'
                sev_color = '#ef4444'
            elif conf_sc >= 0.5:
                severity = 'Sedang'
                sev_color = '#f97316'
            else:
                severity = 'Rendah'
                sev_color = '#eab308'
            
            detections.append({
                'class_id'    : cls_id,
                'class_name'  : cls_name,
                'confidence'  : round(conf_sc * 100, 1),
                'bbox'        : [round(x, 1) for x in xyxy],
                'severity'    : severity,
                'sev_color'   : sev_color,
            })
        
        # Sortir berdasarkan confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Status keseluruhan
        cataract_detections = [
            d for d in detections
            if 'katarak' in d['class_name'].lower() or 'cataract' in d['class_name'].lower()
        ]
        
        if len(cataract_detections) == 0:
            overall_status = 'normal'
            status_text    = 'Tidak Terdeteksi Katarak'
            status_desc    = 'Berdasarkan analisis AI, tidak ditemukan indikasi katarak pada gambar mata ini.'
            status_color   = '#10b981'
        else:
            overall_status = 'detected'
            status_text    = f'{len(cataract_detections)} Objek Terdeteksi'
            status_desc    = 'Terdeteksi indikasi katarak. Segera konsultasikan dengan dokter mata!'
            status_color   = '#ef4444'
        
        return {
            'success'        : True,
            'detections'     : detections,
            'num_detections' : len(cataract_detections),
            'result_image'   : result_b64,
            'inference_time' : round(inference_time, 1),
            'overall_status' : overall_status,
            'status_text'    : status_text,
            'status_desc'    : status_desc,
            'status_color'   : status_color,
            'device_used'    : 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU',
        }
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {
            'success': False,
            'error': f'Error saat deteksi: {str(e)}'
        }

# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', model_status=MODEL_STATUS, model_loaded=MODEL_LOADED)


@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint utama untuk deteksi katarak."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file yang dikirim'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': f'Format file tidak didukung. Gunakan: {", ".join(ALLOWED_EXTS)}'})
    
    try:
        # Bersihkan file lama
        cleanup_old_files()
        
        # Simpan file
        ext      = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Validasi gambar
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'File bukan gambar yang valid'})
        
        # Ambil parameter
        conf = float(request.form.get('conf', 0.25))
        iou  = float(request.form.get('iou', 0.45))
        conf = max(0.01, min(0.99, conf))
        iou  = max(0.01, min(0.99, iou))
        
        # Jalankan deteksi
        result = run_inference(filepath, conf=conf, iou=iou)
        
        # Tambahkan URL gambar original
        if result.get('success'):
            result['original_image_url'] = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Upload/detect error: {e}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})


@app.route('/status')
def status():
    """Cek status model dan GPU."""
    import torch
    return jsonify({
        'model_loaded' : MODEL_LOADED,
        'model_status' : MODEL_STATUS,
        'model_path'   : str(MODEL_PATH),
        'gpu_available': torch.cuda.is_available(),
        'gpu_name'     : torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'app_version'  : '1.0.0',
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    print('='*60)
    print('  🚀 DEDIKAT — Sistem Deteksi Dini Katarak')
    print('='*60)
    print(f'  Model   : {MODEL_STATUS}')
    print(f'  URL     : http://localhost:5000')
    print('='*60)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
