# 01 — Kriteria Submission Dicoding: Proyek Klasifikasi Gambar

Kelas: **Belajar Fundamental Deep Learning** (Dicoding). Dokumen ini merangkum instruksi resmi dan
memetakannya ke cara proyek ini memenuhinya. Jika ada keraguan, instruksi resmi Dicoding yang berlaku.

## 1. Tugas
Membangun model ML untuk klasifikasi gambar, lalu mengonversi model ke format **SavedModel**, **TF-Lite**,
dan **TFJS**. Topik bebas, harus etis, tidak kontroversial, dan mematuhi lisensi sumber data.

Proyek ini memakai **GTSRB (German Traffic Sign Recognition Benchmark)**: 43 kelas rambu lalu lintas,
lebih dari 50.000 gambar, ukuran asli bervariasi 15x15 sampai 250x250 piksel dan tidak selalu persegi.
Data bebas dipakai dengan permintaan sitasi: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel,
"The German Traffic Sign Recognition Benchmark: A multi-class classification competition", IJCNN 2011.

## 2. Kriteria wajib (semua harus terpenuhi)

| # | Kriteria resmi | Cara proyek ini memenuhi | Bukti di notebook |
|---|---|---|---|
| 1 | Dataset bebas, minimal 1.000 gambar | GTSRB versi Kaggle (±51.839 gambar) diunduh dengan `kagglehub` | Output `TOTAL_IMAGES` |
| 2 | Tidak boleh dataset yang sudah dipakai di kelas (Rock-Paper-Scissors, X-Ray) | GTSRB bukan keduanya | Nama dataset di judul dan sel Data Loading |
| 3 | Dataset dibagi menjadi train, test, dan validation (X% / Y% / Z%) | Folder `Train` dibagi train/validation per kelas berbasis *track* (tanpa kebocoran). Folder `Test` resmi + `Test.csv` menjadi test set | Tabel split + `SPLIT_COUNTS` |
| 4 | Model menggunakan **Sequential, Conv2D, Pooling Layer** | `keras.Sequential` berisi blok `Conv2D` + `MaxPooling2D` | `model.summary()` |
| 5 | Akurasi training dan testing minimal **85%** | Target internal >= 95% (lihat saran) | `TRAIN_ACCURACY`, `TEST_ACCURACY` |
| 6 | Plot akurasi dan loss model | Dua grafik: akurasi dan loss (train vs validation) per epoch | Gambar di bagian Evaluasi |
| 7 | Simpan model ke SavedModel, TF-Lite, dan TFJS | `model.export()`, `TFLiteConverter`, `tensorflowjs_converter` | Folder `saved_model/`, `tflite/`, `tfjs_model/` |

## 3. Saran untuk nilai tinggi (target: SEMUA terpenuhi = bintang 5)

| # | Saran resmi | Cara proyek ini memenuhi | Bukti |
|---|---|---|---|
| S1 | Mengimplementasikan Callback | `TargetAccuracyStop` (callback kustom) + `ModelCheckpoint` + `EarlyStopping` + `ReduceLROnPlateau` | Argumen `callbacks=` pada `fit` dan log training |
| S2 | Gambar pada dataset asli beresolusi tidak seragam (tanpa preprocessing) | Resolusi asli GTSRB 15–250 px; dicek dengan PIL untuk setiap gambar | `UNIQUE_RESOLUTIONS` + scatter plot resolusi |
| S3 | Dataset minimal 10.000 gambar | ±51.839 gambar | `TOTAL_IMAGES` |
| S4 | Akurasi training dan testing minimal **95%** | CNN + BatchNorm + Dropout + augmentasi | `TRAIN_ACCURACY`, `TEST_ACCURACY` |
| S5 | Memiliki 3 kelas atau lebih | 43 kelas | `NUM_CLASSES` |
| S6 | Inference memakai salah satu model (TF-Lite/TFJS/SavedModel) dengan bukti screenshot atau output notebook | Inference TF-Lite pada seluruh test set + grid 12 gambar dengan label prediksi | `TFLITE_TEST_ACCURACY` + gambar di bagian Inference |

## 4. Skala penilaian resmi

- Bintang 1: semua kriteria utama terpenuhi, tetapi penulisan kode masih perlu banyak diperbaiki atau terindikasi plagiat.
- Bintang 2: semua kriteria utama terpenuhi, tetapi penulisan kode masih perlu diperbaiki.
- Bintang 3: semua kriteria utama terpenuhi, tetapi tidak ada saran yang terpenuhi.
- Bintang 4: semua kriteria utama terpenuhi dan minimal 3 saran diterapkan.
- Bintang 5: semua kriteria utama terpenuhi dan semua saran diterapkan.

Artinya **kualitas penulisan kode ikut menentukan bintang**. Kode harus rapi, terstruktur, diberi penjelasan,
dan tidak menyalin pekerjaan orang lain.

## 5. Ketentuan berkas
- Bahasa pemrograman Python.
- Kumpulkan: notebook `.ipynb`, model (SavedModel, TFJS, TF-Lite), `requirements.txt`.
- Notebook `.ipynb` **sudah dijalankan** sehingga output terlihat tanpa reviewer menjalankan ulang.
- Kirim dalam **satu folder yang di-zip**.

Struktur yang disarankan Dicoding, ditambah `notebook.py` (lihat catatan interpretasi):

```
submission
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
├───notebook.ipynb
├───notebook.py          <- wajib menurut daftar penolakan
├───README.md
└───requirements.txt
```

## 6. Submission yang ditolak (daftar resmi)
- Akurasi training dan testing di bawah 85%.
- Tidak melampirkan **kedua** file `.py` dan `.ipynb`.
- Tidak menyimpan model dalam format SavedModel, TFJS, dan TF-Lite.
- File Jupyter Notebook belum dijalankan.
- Tidak menerapkan seluruh kriteria wajib.
- File submission tidak bisa di-load oleh platform Dicoding.
- Tidak dikirim sebagai archive `.zip` atau `.rar`.
- "Menambahkan file requirements.txt" (lihat catatan interpretasi).

### Catatan interpretasi (keputusan proyek)
1. **`.py` wajib.** Struktur yang disarankan tidak memuat `.py`, tetapi daftar penolakan menyebut kedua file
   `.py` dan `.ipynb` harus dilampirkan, dan tips resmi menyuruh mengunduh `.ipynb` serta `.py` dari Colab.
   Maka `submission/notebook.py` selalu disertakan.
2. **`requirements.txt` wajib ada.** Poin penolakan "Menambahkan file requirements.txt" bertentangan dengan
   ketentuan berkas yang mewajibkannya. Kemungkinan besar maksudnya "tidak menambahkan". Maka file ini disertakan.
3. **Dataset tidak ikut di-zip.** Yang dikirim hanya notebook, model, dan dokumen pendukung.
4. **"Tanpa preprocessing" pada saran S2** berarti dataset aslinya belum diseragamkan ukurannya. Notebook tetap
   boleh (dan harus) melakukan resize di pipeline data. Buktinya adalah output analisis resolusi asli.
5. **Augmentasi hanya pada data training** (tips resmi no. 4) agar validasi dan test tetap konsisten.

## 7. Tips resmi yang relevan
1. Gunakan templat proyek resmi (struktur heading: `docs/02_template_notebook.md`).
2. Eksplorasi data dulu untuk menentukan preprocessing dan augmentasi.
3. Gunakan Google Colab dengan GPU T4.
4. Augmentasi hanya untuk data training.
5. Cek jumlah gambar dan resolusi. Fungsi contoh dari Dicoding mencetak semua resolusi unik per kelas, yang
   terlalu panjang untuk GTSRB, sehingga notebook memakai versi ringkas (tabel per kelas + jumlah resolusi unik).
6. `requirements.txt` bisa dibuat dengan `pip freeze` atau `pipreqs`. Proyek ini menuliskannya dari versi
   library yang benar-benar dipakai saat notebook dijalankan.
7. Ekspor dari Colab: File > Download > Download .ipynb dan Download .py.

## 8. Proses review
- Review paling lambat 3 hari kerja (tidak termasuk Sabtu, Minggu, libur nasional).
- Jangan submit berkali-kali karena memperlama antrean penilaian. Pastikan pemeriksa otomatis lulus dulu.
- Forum diskusi: https://www.dicoding.com/academies/185/discussions
