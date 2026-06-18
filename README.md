# SiCASA — CataractScan: Pipeline Deteksi Katarak dengan YOLOv8

**Proyek:** Sistem Deteksi Dini Katarak (SiCASA)  
**Model:** YOLOv8 Object Detection  
**Dataset:** ODIR Cataract (Roboflow)  
**Kelas:** 2 — `Cataract` (0), `Normal` (1)  
**Author:** Brian  
**Environment:** `sicasa_gpu` (Anaconda + CUDA)  

Sistem Deteksi Dini Katarak (SiCASA) adalah sebuah proyek visi komputer yang dikembangkan untuk mendeteksi katarak secara otomatis dari citra medis dengan cepat dan akurat. Proyek ini mengimplementasikan model deteksi objek mutakhir YOLOv8 yang dilatih menggunakan dataset ODIR Cataract untuk membedakan antara kondisi mata normal dan yang terindikasi katarak. Seluruh tahapan dalam pipeline proyek ini telah dirancang secara sistematis mulai dari persiapan dan pembersihan data hingga ke tahap deployment. Hasil akhir dari proyek ini adalah sebuah aplikasi web interaktif yang memungkinkan pengguna untuk melakukan skrining deteksi dini secara mandiri.

---

### Progress dan Pipeline End-to-End

Berikut adalah rincian tahapan capaian dalam pengembangan proyek ini beserta deskripsi cara kerjanya:

| Tahap | Deskripsi & Cara Kerja |
|:-----:|:---------|
| **1** | **Install & Import Library**<br>Mengatur environment pengembangan dengan menginstal dependensi yang dibutuhkan seperti Ultralytics, OpenCV, dan Pandas. Tahap ini memastikan semua fungsi dan arsitektur model siap digunakan dalam pipeline. |
| **2** | **Download Dataset dari Roboflow**<br>Dataset citra katarak diunduh langsung dari platform Roboflow menggunakan API ke dalam penyimpanan lokal. Struktur direktori dataset awal otomatis terbuat untuk diproses lebih lanjut. |
| **3** | **Exploratory Data Analysis (EDA)**<br>Melakukan eksplorasi data untuk memahami distribusi kelas, dimensi ukuran citra, dan karakteristik bounding box. Analisis ini sangat krusial untuk menentukan perlakuan pra-pemrosesan data yang tepat. |
| **4** | **Data Cleaning & Validasi**<br>Memeriksa kelengkapan pasangan gambar dan label untuk menyaring data yang korup atau tidak valid. Proses ini menjamin model hanya akan dilatih menggunakan sampel data yang berkualitas tinggi. |
| **5** | **Stratified Split (70/20/10)**<br>Membagi dataset secara proporsional menjadi himpunan pelatihan, validasi, dan pengujian. Metode stratified digunakan agar rasio kelas katarak dan normal selalu seimbang di setiap partisinya. |
| **6** | **Restrukturisasi Folder YOLOv8**<br>Menyusun ulang struktur folder penyimpanan dataset sesuai dengan standar hirarki yang diwajibkan oleh framework YOLOv8. Gambar dan label didistribusikan ke direktori masing-masing secara terstruktur. |
| **7** | **Generate `data.yaml`**<br>Membuat file konfigurasi utama yang memuat jalur absolut menuju direktori dataset beserta definisi jumlah dan nama kelas. File ini akan dibaca secara otomatis oleh YOLOv8 saat memulai pelatihan. |
| **8** | **Training Model YOLOv8**<br>Melatih model dasar YOLOv8 menggunakan data latih untuk mengenali pola dan fitur penyakit katarak secara iteratif. Proses optimasi bobot model ini difasilitasi oleh akselerasi perangkat keras GPU. |
| **9** | **Evaluasi & Analisis Performa**<br>Mengukur kinerja model yang telah dilatih menggunakan data validasi dan data pengujian yang belum pernah dikenali sebelumnya. Model diuji agar tidak mengalami overfitting menggunakan metrik seperti Precision, Recall, dan mAP. |
| **10**| **Visualisasi Hasil Prediksi**<br>Menampilkan gambar sampel beserta prediksi bounding box yang dihasilkan oleh model secara langsung. Langkah ini berfungsi sebagai uji kualitatif untuk memastikan model melokalisasi area katarak dengan logis. |
| **11**| **Deploy ke Web App SiCASA**<br>Mengintegrasikan bobot model terbaik (`best.pt`) ke dalam antarmuka aplikasi web berbasis Python. Hal ini menghubungkan model ke ranah produksi agar pengguna bisa mengunggah gambar dan melihat prediksi deteksi secara langsung. |

---

> **Catatan Riset:**  
> Seluruh pipeline ini dirancang agar **reproducible** (`seed=42`).  
> Stratified split memastikan distribusi kelas proporsional di setiap partisi.  
> Early stopping (`patience=20`) digunakan untuk mencegah overfitting selama masa pelatihan.
