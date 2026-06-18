/* ================================================
   SiCASA — Main JavaScript
   Handles: Upload, Detection, UI interactions
   ================================================ */

// ──────────────────────────────────────
// DOM Elements
// ──────────────────────────────────────
const uploadArea      = document.getElementById('uploadArea');
const fileInput       = document.getElementById('fileInput');
const previewArea     = document.getElementById('previewArea');
const previewImg      = document.getElementById('previewImg');
const previewInfo     = document.getElementById('previewInfo');
const btnRemove       = document.getElementById('btnRemove');
const btnDetect       = document.getElementById('btnDetect');
const resultPanel     = document.getElementById('resultPanel');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent   = document.getElementById('resultContent');
const loadingOverlay  = document.getElementById('loadingOverlay');
const toast           = document.getElementById('toast');
const toastMsg        = document.getElementById('toastMsg');
const confSlider      = document.getElementById('confSlider');
const confValue       = document.getElementById('confValue');
const iouSlider       = document.getElementById('iouSlider');
const iouValue        = document.getElementById('iouValue');
const settingsToggle  = document.getElementById('settingsToggle');
const settingsContent = document.getElementById('settingsContent');
const navbar          = document.getElementById('navbar');
const btnReset        = document.getElementById('btnReset');
const btnDownload     = document.getElementById('btnDownload');

// ──────────────────────────────────────
// State
// ──────────────────────────────────────
let selectedFile = null;
let lastResult   = null;

// ──────────────────────────────────────
// Navbar scroll effect
// ──────────────────────────────────────
window.addEventListener('scroll', () => {
    if (window.scrollY > 60) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// ──────────────────────────────────────
// Smooth scroll untuk nav links
// ──────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const target = document.querySelector(link.getAttribute('href'));
        if (target) {
            const offset = 80;
            const top = target.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top, behavior: 'smooth' });
        }
    });
});

// ──────────────────────────────────────
// Toast Notification
// ──────────────────────────────────────
function showToast(message, duration = 3000) {
    toastMsg.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

// ──────────────────────────────────────
// File Upload — Click
// ──────────────────────────────────────
uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// ──────────────────────────────────────
// Drag & Drop
// ──────────────────────────────────────
uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

// ──────────────────────────────────────
// Handle File
// ──────────────────────────────────────
function handleFile(file) {
    // Validasi tipe
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
    if (!allowed.includes(file.type)) {
        showToast('❌ Format file tidak didukung. Gunakan JPG, PNG, atau WEBP.');
        return;
    }

    // Validasi ukuran (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showToast('❌ File terlalu besar. Maksimal 16MB.');
        return;
    }

    selectedFile = file;

    // Preview
    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';

        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        previewInfo.textContent = `${file.name} — ${sizeMB} MB`;

        btnDetect.disabled = false;
        showToast('✅ Gambar berhasil dipilih!');

        // Reset hasil sebelumnya
        showPlaceholder();
    };
    reader.readAsDataURL(file);
}

// ──────────────────────────────────────
// Remove File
// ──────────────────────────────────────
btnRemove.addEventListener('click', () => {
    resetUpload();
});

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewArea.style.display = 'none';
    uploadArea.style.display = 'flex';
    btnDetect.disabled = true;
    showPlaceholder();
}

// ──────────────────────────────────────
// Settings Toggle
// ──────────────────────────────────────
settingsToggle.addEventListener('click', () => {
    settingsContent.classList.toggle('open');
    settingsToggle.querySelector('.toggle-icon').classList.toggle('open');
});

confSlider.addEventListener('input', () => {
    confValue.textContent = confSlider.value + '%';
});

iouSlider.addEventListener('input', () => {
    iouValue.textContent = iouSlider.value + '%';
});

// ──────────────────────────────────────
// FAQ Accordion
// ──────────────────────────────────────
document.querySelectorAll('.faq-question').forEach(btn => {
    btn.addEventListener('click', () => {
        const answerId = btn.id + '-answer';
        const answer   = document.getElementById(answerId);
        const icon     = btn.querySelector('.faq-icon');
        
        if (!answer) return;
        
        const isOpen = answer.classList.contains('open');
        
        // Tutup semua
        document.querySelectorAll('.faq-answer').forEach(a => a.classList.remove('open'));
        document.querySelectorAll('.faq-icon').forEach(i => i.classList.remove('open'));
        
        // Toggle yang diklik
        if (!isOpen) {
            answer.classList.add('open');
            icon.classList.add('open');
        }
    });
});

// ──────────────────────────────────────
// Deteksi — Main Function
// ──────────────────────────────────────
btnDetect.addEventListener('click', async () => {
    if (!selectedFile) {
        showToast('❌ Pilih gambar terlebih dahulu!');
        return;
    }

    // Show loading
    setDetecting(true);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('conf', confSlider.value / 100);
        formData.append('iou', iouSlider.value / 100);

        const response = await fetch('/detect', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }

        const data = await response.json();
        
        setDetecting(false);

        if (data.success) {
            lastResult = data;
            displayResult(data);
        } else {
            showToast(`❌ ${data.error || 'Terjadi kesalahan'}`);
            showPlaceholder();
        }

    } catch (err) {
        setDetecting(false);
        console.error('Detection error:', err);
        showToast('❌ Gagal terhubung ke server. Pastikan Flask berjalan.');
        showPlaceholder();
    }
});

// ──────────────────────────────────────
// Set Detecting State
// ──────────────────────────────────────
function setDetecting(loading) {
    if (loading) {
        loadingOverlay.style.display = 'flex';
        btnDetect.disabled = true;
        btnDetect.querySelector('.btn-text').style.display = 'none';
        btnDetect.querySelector('.btn-spinner').style.display = 'block';
    } else {
        loadingOverlay.style.display = 'none';
        btnDetect.disabled = false;
        btnDetect.querySelector('.btn-text').style.display = 'block';
        btnDetect.querySelector('.btn-spinner').style.display = 'none';
    }
}

// ──────────────────────────────────────
// Display Result
// ──────────────────────────────────────
function displayResult(data) {
    resultPlaceholder.style.display = 'none';
    resultContent.style.display    = 'flex';

    // Status utama
    const statusEl  = document.getElementById('resultStatus');
    const statusIcon = document.getElementById('statusIconLarge');
    const statusTitle = document.getElementById('statusTitle');
    const statusDesc  = document.getElementById('statusDesc');

    if (data.overall_status === 'normal') {
        statusEl.style.background    = 'rgba(34, 197, 94, 0.08)';
        statusEl.style.borderColor   = 'rgba(34, 197, 94, 0.3)';
        statusIcon.textContent       = '✅';
        statusTitle.textContent      = data.status_text;
        statusTitle.style.color      = '#22c55e';
    } else {
        statusEl.style.background    = 'rgba(239, 68, 68, 0.08)';
        statusEl.style.borderColor   = 'rgba(239, 68, 68, 0.3)';
        statusIcon.textContent       = '⚠️';
        statusTitle.textContent      = data.status_text;
        statusTitle.style.color      = '#ef4444';
    }
    statusDesc.textContent = data.status_desc;

    // Gambar
    const originalImg   = document.getElementById('originalImg');
    const detectionImg  = document.getElementById('detectionImg');
    
    if (data.original_image_url) {
        originalImg.src = data.original_image_url;
    }
    
    if (data.result_image) {
        detectionImg.src = 'data:image/jpeg;base64,' + data.result_image;
    }

    // Stats
    document.getElementById('numDetections').textContent = data.num_detections;
    document.getElementById('inferenceTime').textContent = data.inference_time;
    
    const deviceText = data.device_used || 'CPU';
    document.getElementById('deviceUsed').textContent = deviceText.includes('GPU') ? 'GPU' : 'CPU';

    // Detection List
    const listEl = document.getElementById('detectionList');
    listEl.innerHTML = '';

    if (data.detections && data.detections.length > 0) {
        const header = document.createElement('div');
        header.style.cssText = 'font-size:0.85rem;font-weight:600;color:var(--text-muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;';
        header.textContent = `Detail Deteksi (${data.detections.length} objek)`;
        listEl.appendChild(header);

        data.detections.forEach((det, i) => {
            const item = document.createElement('div');
            item.className = 'detection-item fade-in';
            item.style.animationDelay = `${i * 80}ms`;

            const confColor = det.confidence >= 80 ? '#ef4444' : 
                              det.confidence >= 50 ? '#f97316' : '#eab308';

            item.innerHTML = `
                <div class="det-info">
                    <div class="det-class" style="color:${confColor}">${det.class_name}</div>
                    <div class="det-bbox">BBox: [${det.bbox.join(', ')}]</div>
                </div>
                <div class="det-right">
                    <span class="det-conf-text" style="color:${confColor}">${det.confidence}%</span>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar" style="width:${det.confidence}%;background:${confColor}"></div>
                    </div>
                    <span class="severity-badge" style="background:${det.sev_color}22;color:${det.sev_color};border:1px solid ${det.sev_color}44;">
                        ${det.severity}
                    </span>
                </div>
            `;
            listEl.appendChild(item);
        });
    }

    // Rekomendasi
    const recEl = document.getElementById('recommendation');
    if (data.overall_status === 'normal') {
        recEl.style.background    = 'rgba(34, 197, 94, 0.06)';
        recEl.style.borderColor   = 'rgba(34, 197, 94, 0.25)';
        recEl.innerHTML = `
            <strong style="color:#22c55e;">✅ Rekomendasi:</strong><br>
            Tidak ditemukan indikasi katarak pada analisis ini. Namun, tetap lakukan 
            pemeriksaan mata rutin minimal <strong>1 tahun sekali</strong> untuk menjaga 
            kesehatan mata jangka panjang.
        `;
    } else {
        recEl.style.background    = 'rgba(239, 68, 68, 0.06)';
        recEl.style.borderColor   = 'rgba(239, 68, 68, 0.25)';
        recEl.innerHTML = `
            <strong style="color:#ef4444;">⚠️ Rekomendasi:</strong><br>
            Terdeteksi indikasi yang perlu diperhatikan. <strong>Segera konsultasikan 
            dengan dokter spesialis mata</strong> untuk pemeriksaan lebih lanjut dan 
            penanganan yang tepat. Jangan tunda — deteksi dini meningkatkan peluang 
            pemulihan secara signifikan.
        `;
    }

    // Scroll ke hasil
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    showToast('🎉 Deteksi selesai!');
}

// ──────────────────────────────────────
// Show Placeholder
// ──────────────────────────────────────
function showPlaceholder() {
    resultPlaceholder.style.display = 'flex';
    resultContent.style.display    = 'none';
}

// ──────────────────────────────────────
// Reset Button
// ──────────────────────────────────────
btnReset.addEventListener('click', () => {
    resetUpload();
    lastResult = null;
    showToast('🔄 Siap untuk deteksi baru!');
});

// ──────────────────────────────────────
// Download Result
// ──────────────────────────────────────
btnDownload.addEventListener('click', () => {
    if (!lastResult || !lastResult.result_image) {
        showToast('❌ Tidak ada hasil untuk diunduh');
        return;
    }

    const link = document.createElement('a');
    link.href  = 'data:image/jpeg;base64,' + lastResult.result_image;
    link.download = `sicasa_hasil_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('⬇️ Gambar hasil diunduh!');
});

// ──────────────────────────────────────
// Animate cards on scroll
// ──────────────────────────────────────
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.info-card, .step-item').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = `opacity 0.5s ease ${i * 80}ms, transform 0.5s ease ${i * 80}ms`;
    observer.observe(el);
});

// ──────────────────────────────────────
// Init
// ──────────────────────────────────────
console.log('%c 👁️ SiCASA — Sistem Deteksi Dini Katarak', 
    'color:#3b82f6;font-size:1rem;font-weight:bold;');
console.log('%c v1.0.0 — YOLOv8s + Flask + CUDA', 
    'color:#94a3b8;font-size:0.8rem;');
