@echo off
title SiCASA — Sistem Deteksi Dini Katarak
color 0B

echo.
echo ============================================================
echo   ^|^|^|  SiCASA - Sistem Deteksi Dini Katarak
echo ============================================================
echo.
echo   Menggunakan environment: sicasa_gpu
echo   Model: YOLOv8s + CUDA (RTX 3050)
echo.

REM Cek apakah best.pt ada
if exist "app\best.pt" (
    echo   [OK] Model best.pt ditemukan!
) else (
    echo   [!] PERINGATAN: best.pt tidak ditemukan di folder app/
    echo   [!] Jalankan training di notebook terlebih dahulu,
    echo   [!] lalu salin best.pt ke folder app/
    echo.
)

echo   Starting Flask server...
echo   Buka browser: http://localhost:5000
echo.
echo   Tekan Ctrl+C untuk menghentikan server
echo ============================================================
echo.

C:\Users\brian\anaconda3\envs\sicasa_gpu\python.exe app\app.py

pause
