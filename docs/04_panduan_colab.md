# 04 — Panduan Menjalankan Notebook di Google Colab

Bagian ini dikerjakan **oleh kamu (user)**. Agent menulis notebook di VS Code; Colab menjalankannya dengan GPU.

## A. Jalankan notebook
1. Buka https://colab.research.google.com lalu pilih **File > Upload notebook**.
   Pilih file `submission/notebook.ipynb` dari folder proyek.
2. Pilih **Runtime > Change runtime type > T4 GPU > Save**.
   (Metadata notebook sudah meminta GPU, tapi tetap cek.)
3. Pilih **Runtime > Run all** (untuk menjalankan ulang dari nol: **Runtime > Restart session and run all**).
   Jika muncul peringatan "This notebook was not authored by Google", klik **Run anyway**.
4. Tunggu sampai sel terakhir selesai. Perkiraan waktu dengan T4 (bisa berbeda):
   - unduh dataset ±1–3 menit
   - cek resolusi semua gambar ±1–2 menit
   - training ±5–15 menit (berhenti otomatis oleh callback)
   - konversi TFJS (membuat venv terpisah) ±2–5 menit
   - inference < 1 menit
5. Saat sel konversi selesai, browser otomatis mengunduh **`hasil_colab_artifacts.zip`**
   (berisi `saved_model/`, `tflite/`, `tfjs_model/`, `requirements.txt`). Izinkan unduhan bila diminta.
   Jika tidak terunduh: buka panel **Files** (ikon folder di kiri), klik kanan file tersebut, lalu **Download**.

## B. Unduh hasil (setelah SEMUA sel selesai tanpa error)
6. **File > Download > Download .ipynb** (notebook beserta seluruh output).
7. **File > Download > Download .py**.
8. Pindahkan ketiga file ke folder **`hasil_colab/`** di proyek VS Code:
   ```
   hasil_colab/
   ├── notebook.ipynb              (nama boleh berbeda, yang terbaru dipakai)
   ├── notebook.py
   └── hasil_colab_artifacts.zip
   ```
9. Di panel Claude Code, ketik **`/finalisasi`**.

## C. Jika ada error
1. Salin **traceback lengkap**: mulai dari baris `Traceback (most recent call last)` atau nama error,
   sampai pesan terakhir. Sertakan juga potongan kode sel yang gagal bila perlu.
2. Di Claude Code ketik **`/perbaiki`** lalu tempel error tersebut.
3. Setelah agent memperbaiki notebook: di Colab buka lagi **File > Upload notebook** (pilih file yang sudah
   diperbaiki), lalu **Runtime > Restart session and run all**.
   - Jangan mengedit kode langsung di Colab. Semua perubahan lewat VS Code agar tetap sinkron.
   - Dataset dan venv TFJS tersimpan selama sesi Colab masih aktif, jadi run ulang lebih cepat.

## D. Catatan
- **Kuota GPU habis?** Notebook tetap bisa jalan di CPU (Runtime type: CPU), hanya lebih lama.
- **kagglehub meminta login?** Buat API token di kaggle.com (Settings > API > Create New Token).
  Di Colab: ikon kunci (Secrets) > tambahkan `KAGGLE_USERNAME` dan `KAGGLE_KEY` > aktifkan Notebook access >
  jalankan ulang.
- **Run final:** sebelum mengunduh versi final, pastikan notebook dijalankan sekali dari awal sampai akhir
  (Restart session and run all) agar nomor eksekusi sel berurutan dan tidak ada output lama.
