# CLAUDE.md — Submission Dicoding: Proyek Klasifikasi Gambar (GTSRB)

## Misi
Menyusun submission kelas **Belajar Fundamental Deep Learning** (Dicoding) yang lulus **semua kriteria wajib**
dan memenuhi **semua saran** (target bintang 5). Proyeknya klasifikasi 43 rambu lalu lintas **GTSRB** dengan CNN
`keras.Sequential`, lalu ekspor model ke **SavedModel**, **TF-Lite**, dan **TFJS**.
Komunikasi dengan user selalu dalam **Bahasa Indonesia**.

## Identitas siswa (diisi user)
- Nama: Roy Aldonit Zidomi
- Email: zidomiroy@gmail.com
- ID Dicoding: royzidomi15

Jika masih berupa `<<...>>`, tanyakan ke user sebelum menulis header notebook, lalu simpan jawabannya di sini.

## Mode kerja: Google Colab (GPU T4)
Kamu menulis dan memverifikasi di VS Code. **User yang menjalankan notebook di Colab.**
| Fase | Pemicu | Inti pekerjaan |
|---|---|---|
| A. Bangun | `/submission` | Tulis `submission/notebook.ipynb` lengkap, lalu pre-check 0 FAIL, lalu serahkan langkah Colab |
| B. Perbaiki | `/perbaiki <traceback>` | Perbaiki akar masalah di notebook, pre-check ulang, minta user run ulang |
| C. Finalisasi | `/finalisasi` | Impor `hasil_colab/`, cek final, tulis README dari angka nyata, buat `submission.zip` |
| Cek | `/cek-submission` | Jalankan checker dan jelaskan hasilnya |

Instruksi lengkap per fase ada di `PROMPT_EKSEKUSI.md`.

## Dokumen rujukan
- @docs/01_kriteria_dicoding.md — kriteria wajib, saran bintang 5, alasan penolakan, peta kriteria ke bukti.
- `docs/02_template_notebook.md` — struktur heading templat (wajib persis) dan isi tiap bagian.
- `docs/03_panduan_teknis.md` — keputusan teknis, jebakan, strategi akurasi, dan **kode referensi teruji**.
- `docs/04_panduan_colab.md` — langkah user di Colab dan cara membawa hasilnya kembali.
- `docs/05_template_readme.md` — template `submission/README.md`.

## Aturan keras
1. Python + TensorFlow/Keras. Model `keras.Sequential` dengan `Conv2D` dan `MaxPooling2D` ditulis eksplisit.
2. Heading templat dan urutannya **persis** seperti docs/02. Boleh menambah sel, tidak boleh mengganti, menghapus,
   atau menyisipkan heading baru.
3. Split 3 subset tanpa kebocoran: validation per-track dari folder Train, test dari folder Test resmi.
   Jumlah dan persentase dicetak.
4. Augmentasi **hanya** untuk train. **Tanpa flip** (makna rambu bisa berubah).
5. Akurasi train dan test >= 95%, diukur dengan `evaluate` pada train tanpa augmentasi dan test set, lalu dicetak.
6. Plot akurasi dan loss (train vs validation).
7. Path ekspor persis: `submission/saved_model/` (saved_model.pb + variables/),
   `submission/tflite/model.tflite` + `label.txt`, `submission/tfjs_model/model.json` + `group1-shard1of1.bin`.
8. Inference memakai model hasil konversi (TF-Lite) dengan bukti output di notebook.
9. Isi folder submission wajib: `notebook.ipynb` (sudah dijalankan), `notebook.py`, `README.md`,
   `requirements.txt`, dan ketiga model. Dikirim sebagai `submission.zip` berisi satu folder `submission/`.
10. Notebook mencetak baris penanda untuk checker: `TOTAL_IMAGES`, `NUM_CLASSES`, `UNIQUE_RESOLUTIONS`,
    `SPLIT_COUNTS`, `TRAIN_ACCURACY`, `VAL_ACCURACY`, `TEST_ACCURACY`, `TFLITE_TEST_ACCURACY`.
11. **Jangan pernah mengarang** output, angka, atau klaim "sudah dijalankan". Angka di README, markdown, dan
    laporan hanya boleh diambil dari output notebook hasil Colab.
12. Setelah notebook dijalankan di Colab, **jangan ubah sel kode atau output**. Hanya sel markdown yang boleh
    dirapikan, lalu verifikasi dengan `python scripts/pack_submission.py verify`.
13. Dataset, checkpoint (`.keras`/`.h5`), dan venv tidak boleh masuk folder submission atau zip.
    Dataset Rock-Paper-Scissors dan X-Ray dilarang.
14. Jangan `pip install tensorflowjs` di runtime utama. Konversi TFJS dilakukan di venv terpisah dengan pin dari docs/03.
15. Transfer learning, ganti dataset, atau ganti mode kerja hanya dengan persetujuan user.
16. Jangan menjalankan training penuh di komputer lokal kecuali user memintanya.

## Standar kualitas kode (ikut menentukan bintang)
- Satu sel konfigurasi untuk semua konstanta. Fungsi kecil dengan docstring. Nama variabel jelas.
  Tidak ada kode mati, duplikasi, atau path absolut lokal.
- Setiap bagian diawali markdown singkat yang menjelaskan apa dan kenapa.
- Output rapi: `fit(..., verbose=2)`, tabel ringkas, plot dengan judul, label sumbu, dan legenda.
- Seed tetap dengan `keras.utils.set_random_seed`.

## Perintah
| Tujuan | Perintah |
|---|---|
| Cek sebelum ke Colab | `python scripts/check_submission.py --stage pre` |
| Impor hasil Colab | `python scripts/pack_submission.py import` |
| Pastikan sel kode/output tidak berubah | `python scripts/pack_submission.py verify` |
| Buat ulang notebook.py | `python scripts/pack_submission.py py` |
| Cek final (+ JSON) | `python scripts/check_submission.py --json` |
| Buat dan validasi zip | `python scripts/pack_submission.py zip` |

Jika `python` tidak dikenali di Windows, pakai `py` atau `python3`. Script hanya butuh library standar.

## Definition of Done
- `python scripts/pack_submission.py zip` sukses: 0 FAIL, saran 6/6, zip berisi satu folder `submission/`.
- `submission/README.md` terisi angka nyata tanpa placeholder.
- Laporan akhir ke user: tabel metrik, checklist 7 kriteria + 6 saran, isi zip, dan langkah upload ke Dicoding.
