# 02 — Template Notebook Dicoding (WAJIB diikuti)

Notebook submission harus mengikuti templat resmi Dicoding
(https://colab.research.google.com/drive/17_EuntmF8YjvKWkcU-CPoKREw67BhnZd).
File `submission/notebook.ipynb` di proyek ini sudah berisi kerangka templat tersebut.

## 1. Struktur heading (persis, berurutan)

```markdown
# Proyek Klasifikasi Gambar: [Input Nama Dataset]
- **Nama:** [Input Nama]
- **Email:** [Input Email]
- **ID Dicoding:** [Input Username]

## Import Semua Packages/Library yang Digunakan
## Data Preparation
### Data Loading
### Data Preprocessing
#### Split Dataset
## Modelling
## Evaluasi dan Visualisasi
## Konversi Model
## Inference (Optional)
```

Aturan:
- Teks heading dan urutannya **tidak boleh diubah, dihapus, atau disusun ulang**. Checker memeriksa ini.
- Boleh **menambah sel** (markdown penjelasan atau kode) di bawah heading mana pun.
- Jangan menambah heading baru di antara heading templat. Untuk sub-judul kecil, pakai teks tebal (`**...**`).
- Sel header pertama wajib terisi: judul dataset dan identitas tanpa sisa `[Input ...]`.
- Jika user menaruh file templat asli (unduhan dari link di atas) di proyek, utamakan file itu sebagai basis,
  dengan aturan yang sama.

Isi header untuk proyek ini:
```markdown
# Proyek Klasifikasi Gambar: German Traffic Sign Recognition Benchmark (GTSRB)
- **Nama:** <Nama dari CLAUDE.md>
- **Email:** <Email dari CLAUDE.md>
- **ID Dicoding:** <ID Dicoding dari CLAUDE.md>
```
Boleh menambahkan 2–3 kalimat deskripsi proyek di sel yang sama, di bawah identitas.

## 2. Isi setiap bagian

Pola untuk setiap bagian: **markdown pendek (apa dan kenapa) -> sel kode -> (opsional) markdown interpretasi**.
Kode referensi yang sudah diuji ada di `docs/03_panduan_teknis.md` (Lampiran).

| Bagian | Isi minimal | Output yang harus terlihat |
|---|---|---|
| Import Semua Packages/Library | Sel pemasangan `kagglehub` bila belum ada; semua import; info versi, GPU, lingkungan; sel konfigurasi (seed, ukuran gambar, batch, epoch, path) | Versi Python/TF/Keras dan status GPU |
| Data Loading | Unduh GTSRB via `kagglehub`; daftar isi folder; nama 43 kelas; baca `Train.csv`/`Test.csv`; validasi file; analisis resolusi asli + cek file korup; tabel per kelas; grafik distribusi kelas + scatter resolusi; contoh gambar per kelas | `TOTAL_IMAGES`, `NUM_CLASSES`, `UNIQUE_RESOLUTIONS`, tabel, 2–3 gambar |
| Data Preprocessing | Fungsi baca + resize; pipeline augmentasi (tanpa flip) + visualisasi contoh augmentasi | Gambar contoh augmentasi |
| Split Dataset | Split per-track (train/validation) + test resmi; cek tidak ada irisan; tabel jumlah & persentase; pipeline `tf.data` (augmentasi hanya train) | Tabel split, `SPLIT_COUNTS`, bentuk batch |
| Modelling | `build_model()` dengan `keras.Sequential` + `Conv2D` + `MaxPooling2D`; `model.summary()`; callback kustom + bawaan; `fit` dengan `validation_data`; memuat model terbaik | Ringkasan model, log per epoch (`verbose=2`) |
| Evaluasi dan Visualisasi | Plot akurasi & loss; `evaluate` train (tanpa augmentasi) / validation / test; classification report; confusion matrix | Plot, `TRAIN_ACCURACY`, `VAL_ACCURACY`, `TEST_ACCURACY`, report, heatmap |
| Konversi Model | SavedModel (`model.export`) + verifikasi; TF-Lite + `label.txt`; TFJS di venv terpisah; `requirements.txt`; daftar isi folder; unduh artefak (Colab) | Ukuran file tiap format, isi `requirements.txt`, pohon folder |
| Inference (Optional) | Interpreter TF-Lite; akurasi TF-Lite di seluruh test set; grid 12 gambar test (label asli vs prediksi + confidence) | `TFLITE_TEST_ACCURACY`, gambar grid |

## 3. Baris penanda (marker) yang wajib dicetak
Checker membaca angka dari output notebook. Cetak persis dengan format `NAMA: nilai`:

```
TOTAL_IMAGES: 51839
NUM_CLASSES: 43
UNIQUE_RESOLUTIONS: <angka>
SPLIT_COUNTS: train=<n>, val=<n>, test=<n>
TRAIN_ACCURACY: 0.9xxx
VAL_ACCURACY: 0.9xxx
TEST_ACCURACY: 0.9xxx
TFLITE_TEST_ACCURACY: 0.9xxx
```
(Angka di atas hanya contoh format, bukan hasil.)

## 4. Gaya penulisan markdown
- Bahasa Indonesia yang ringkas dan jelas. Satu paragraf pendek atau 2–4 butir per penjelasan.
- Jelaskan **alasan** keputusan, misalnya: kenapa split per-track, kenapa tanpa flip, kenapa `Rescaling` ada
  di dalam model, kenapa konverter TFJS dijalankan di venv terpisah.
- Saat menulis notebook di VS Code (sebelum dijalankan), markdown interpretasi **tidak boleh berisi angka hasil**.
  Gunakan kalimat netral ("lihat output di atas"). Angka nyata baru boleh ditulis di tahap finalisasi,
  diambil dari output yang benar-benar ada.
- Kesimpulan akhir boleh ditambahkan sebagai sel markdown terakhir (tanpa heading baru), misalnya
  diawali `**Kesimpulan.**`.
