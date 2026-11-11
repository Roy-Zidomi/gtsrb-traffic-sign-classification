# Mulai di Sini — Paket Agent Submission Dicoding (GTSRB + Google Colab)

Paket ini menyiapkan **Claude Code di VS Code** untuk mengerjakan submission *Proyek Klasifikasi Gambar*
(kelas Belajar Fundamental Deep Learning) dengan target bintang 5.

## Isi paket
```
proyek-klasifikasi-gambar/
├── MULAI_DI_SINI.md          <- panduan ini
├── CLAUDE.md                 <- aturan proyek (otomatis dibaca Claude Code). Isi identitasmu di sini
├── PROMPT_EKSEKUSI.md        <- prompt utama berfase (A: bangun, B: perbaiki, C: finalisasi)
├── .claude/commands/         <- perintah cepat: /submission /perbaiki /finalisasi /cek-submission
├── docs/
│   ├── 01_kriteria_dicoding.md   kriteria wajib, saran bintang 5, alasan penolakan
│   ├── 02_template_notebook.md   struktur heading templat Dicoding (wajib persis)
│   ├── 03_panduan_teknis.md      keputusan teknis, jebakan, kode referensi teruji
│   ├── 04_panduan_colab.md       langkah kamu di Google Colab
│   └── 05_template_readme.md     template README submission
├── scripts/
│   ├── check_submission.py       pemeriksa otomatis (pre / final / zip)
│   └── pack_submission.py        impor hasil Colab, verify, buat notebook.py, buat zip
├── hasil_colab/                  <- taruh hasil unduhan Colab di sini
└── submission/
    └── notebook.ipynb            kerangka templat Dicoding (akan diisi agent)
```

## Prasyarat
- VS Code dengan ekstensi **Claude Code**.
- Python 3.8+ di komputer (hanya untuk script pemeriksa, tanpa library tambahan).
- Akun Google untuk Colab.

## Cara pakai
1. Ekstrak paket, lalu di VS Code pilih **File > Open Folder** > `proyek-klasifikasi-gambar`.
2. Isi identitas di `CLAUDE.md` (Nama, Email, ID Dicoding). Jika lupa, agent akan menanyakannya.
3. Buka panel Claude Code dan ketik **`/submission`**.
   Agent membaca semua dokumen, menulis notebook lengkap, menjalankan pre-check, lalu memberi langkah Colab.
   *(Alternatif: salin seluruh isi `PROMPT_EKSEKUSI.md` ke chat Claude Code.)*
4. Jalankan notebook di Colab sesuai `docs/04_panduan_colab.md`, lalu taruh 3 file hasilnya di `hasil_colab/`.
   - Jika error: ketik **`/perbaiki`** lalu tempel traceback-nya. Upload ulang notebook dan run lagi.
5. Ketik **`/finalisasi`**. Agent memverifikasi, menulis README dari angka nyata, dan membuat `submission.zip`.
6. Upload `submission.zip` di halaman submission Dicoding.

Kapan saja, **`/cek-submission`** menjalankan pemeriksa dan menjelaskan hasilnya.

## Catatan jujur
- Pipeline referensi sudah diuji end-to-end: training, SavedModel, TF-Lite, TFJS (dimuat di TensorFlow.js),
  dan checker. Pengujian memakai replika struktur dataset GTSRB, bukan dataset aslinya. Akurasi nyata baru
  diketahui saat kamu menjalankan "Run all" di Colab. Dari literatur, CNN seperti ini biasanya di atas 95%
  pada GTSRB, tetapi angkanya tetap harus dibuktikan oleh output notebook.
- Reviewer juga menilai kualitas kode. Baca penjelasan di notebook agar kamu paham setiap langkah,
  karena itulah inti pembelajarannya.
