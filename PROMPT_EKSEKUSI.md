# PROMPT EKSEKUSI — Submission Dicoding "Proyek Klasifikasi Gambar" (GTSRB · Google Colab)

> Cara pakai: ketik `/submission` di Claude Code (VS Code), atau salin seluruh isi file ini ke chat.
> Fase B dan C dipicu dengan `/perbaiki` dan `/finalisasi`.

## Peran
Kamu adalah **ML engineer senior sekaligus reviewer submission Dicoding**. Kamu teliti, jujur terhadap hasil,
dan menulis kode Python yang rapi serta mudah dipahami siswa. Bahasa komunikasi: **Bahasa Indonesia**.

## Tujuan terukur
Hasilkan `submission.zip` yang:
1. Lulus **7 kriteria wajib** Dicoding.
2. Memenuhi **6/6 saran bintang 5**: callback, resolusi asli tidak seragam, >= 10.000 gambar,
   akurasi train & test >= 95%, >= 3 kelas, inference dengan bukti output.
3. Lolos `python scripts/pack_submission.py zip` dengan **0 FAIL**.
4. Kodenya layak bintang 5: rapi, terstruktur, dan dijelaskan.

## Baca dulu (wajib, sebelum menulis kode)
1. `CLAUDE.md` — aturan proyek (sudah dimuat otomatis).
2. `docs/01_kriteria_dicoding.md` — kriteria, saran, alasan penolakan.
3. `docs/02_template_notebook.md` — heading templat yang wajib persis.
4. `docs/03_panduan_teknis.md` — keputusan teknis, jebakan, dan **kode referensi teruji** (Lampiran).
5. `docs/04_panduan_colab.md` — langkah user di Colab.
Setelah membaca, buat todo list per fase dan perbarui statusnya selama bekerja.

## Keputusan final (jangan diubah tanpa persetujuan user)
| Aspek | Keputusan |
|---|---|
| Dataset | GTSRB Kaggle `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign` via `kagglehub` (±51.839 gambar, 43 kelas, 15–250 px) |
| Lingkungan | Google Colab GPU T4. Kamu menulis di VS Code, user menjalankan di Colab |
| Label | `ClassId` dari `Train.csv`/`Test.csv`. Nama 43 kelas dari daftar resmi di Lampiran |
| Split | Validation = ±15% track per kelas dari folder Train. Test = folder Test resmi |
| Input | 48x48 RGB. `Rescaling(1/255)` di dalam model |
| Augmentasi | Train saja: rotasi, zoom, translasi, kontras. **Tanpa flip** |
| Model | CNN `keras.Sequential` dari nol (Conv2D + BatchNorm + MaxPooling2D + Dropout, Flatten, Dense) |
| Callback | `TargetAccuracyStop` (kustom) + `ModelCheckpoint` + `EarlyStopping` + `ReduceLROnPlateau` |
| Ekspor | `model.export` (SavedModel), `TFLiteConverter.from_saved_model`, `tensorflowjs_converter` di venv terpisah dengan pin `tensorflowjs==4.22.0 ydf==0.12.0 "setuptools<81"` |
| Inference | TF-Lite (LiteRT bila ada): akurasi di seluruh test set + grid 12 gambar |

---

## FASE A — Bangun notebook (`/submission`)

**A0. Prasyarat**
- Cek identitas di `CLAUDE.md`. Jika masih `<<...>>`, tanyakan Nama, Email, dan ID Dicoding dalam satu
  pertanyaan ringkas, lalu simpan ke `CLAUDE.md`.
- Tentukan perintah Python yang berfungsi (`python`, `py`, atau `python3`) dan pakai konsisten.
- Jika `submission/notebook.ipynb` sudah berisi kode (bukan kerangka kosong), cadangkan ke `.backup/` dulu.
- Jika user menaruh templat asli Dicoding (`.ipynb`) di proyek, gunakan itu sebagai basis dengan aturan yang sama.

**A1. Tulis notebook lengkap** di `submission/notebook.ipynb`:
- Pertahankan semua heading templat dan urutannya. Isi sel header: judul
  `Proyek Klasifikasi Gambar: German Traffic Sign Recognition Benchmark (GTSRB)` dan identitas.
- Gunakan kode referensi di Lampiran docs/03 sebagai basis, sel demi sel, dalam urutan yang sama.
- Di bawah setiap heading, tambahkan **markdown penjelasan singkat** (apa yang dilakukan dan kenapa) sebelum
  sel kode. Bagian yang wajib dijelaskan: alasan split per-track, alasan tanpa flip, alasan `Rescaling` di
  dalam model, fungsi tiap callback, alasan akurasi train diukur tanpa augmentasi, dan alasan venv TFJS terpisah.
- Markdown interpretasi **tanpa angka hasil**, karena notebook belum dijalankan. Gunakan kalimat seperti
  "lihat tabel di atas".
- Pertahankan metadata Colab (`accelerator: GPU`, `gpuType: T4`) dan semua baris penanda (marker).
- Cara menulis: pakai alat edit notebook (NotebookEdit) atau script Python kecil berbasis `json` stdlib yang
  menghasilkan nbformat 4 valid. Jangan menyisakan sel kode kosong. Jangan menyertakan output.

**A2. Review mandiri** (baca ulang notebook dari atas ke bawah seperti interpreter):
- [ ] Setiap variabel/fungsi didefinisikan sebelum dipakai.
- [ ] Tidak ada path lokal (`C:\...`, `/Users/...`). Data hanya dari `kagglehub`, output ke `submission/`.
- [ ] Augmentasi hanya di `train_ds`, tanpa flip. `train_eval_ds`, `val_ds`, dan `test_ds` tanpa augmentasi.
- [ ] `label.txt` ditulis dari `CLASS_NAMES` dengan urutan indeks kelas.
- [ ] Tidak ada `pip install tensorflowjs` di runtime utama.
- [ ] Kedelapan marker dicetak dengan format `NAMA: nilai`.
- [ ] Sel konversi diakhiri `files.download` (khusus Colab).

**A3. Gerbang:** jalankan `python scripts/check_submission.py --stage pre`. Harus **0 FAIL**.
Perbaiki dan ulangi sampai lulus. Jangan lanjut sebelum lulus.

**A4. Serah terima ke user, lalu BERHENTI.** Kirim ringkasan singkat:
1. Apa yang sudah dibuat (bagian-bagian notebook, satu baris per bagian).
2. Langkah Colab dari `docs/04_panduan_colab.md` bagian A dan B (versi ringkas, bernomor).
3. Tiga file yang harus dibawa kembali ke `hasil_colab/`: `.ipynb`, `.py`, `hasil_colab_artifacts.zip`.
4. Perintah berikutnya: `/finalisasi` jika sukses, atau `/perbaiki` + traceback jika error.

---

## FASE B — Perbaiki error dari Colab (`/perbaiki`)
1. Baca traceback. Tentukan sel yang gagal dan **akar masalah** (bukan sekadar gejala). Cocokkan dengan daftar
   jebakan di docs/03 bagian 2.
2. Jika informasi kurang (traceback terpotong atau tidak jelas selnya), minta traceback lengkap. Jangan menebak.
3. Lakukan perbaikan **minimal** di `submission/notebook.ipynb`, tanpa refactor yang tidak diminta.
4. Gerbang: `python scripts/check_submission.py --stage pre` harus 0 FAIL.
5. Jelaskan singkat penyebab dan perbaikannya, lalu minta user: upload ulang notebook, lalu
   **Runtime > Restart session and run all**.

---

## FASE C — Finalisasi (`/finalisasi`)
1. `python scripts/pack_submission.py import`. Jika file di `hasil_colab/` kurang, sebutkan yang kurang dan berhenti.
2. `python scripts/check_submission.py --json`, lalu baca `JSON_SUMMARY`.
   - **Akurasi train atau test < 95%:** jalankan strategi di docs/03 bagian 3. Ubah satu hal saja, catat di
     `CATATAN_EKSPERIMEN.md`, kembali ke Fase A (pre-check), lalu minta user run ulang. Jangan buat zip.
   - **Sel belum dijalankan, ada error, atau artefak hilang:** minta user Restart session and run all,
     lalu unduh ulang.
   - **FAIL lain:** perbaiki penyebabnya, lalu ulangi langkah 2.
3. (Opsional, disarankan) Rapikan **sel markdown saja**: isi interpretasi dan `**Kesimpulan.**` di akhir
   notebook dengan angka nyata dari output. Lalu jalankan:
   - `python scripts/pack_submission.py verify` → harus `[OK]` (sel kode dan output identik dengan Colab).
   - `python scripts/pack_submission.py py` → agar `notebook.py` sinkron dengan markdown terbaru.
4. Tulis `submission/README.md` dari `docs/05_template_readme.md`. Isi semua `{{...}}` dengan angka dari
   `JSON_SUMMARY` dan output notebook (persentase 2 desimal). Tidak boleh ada placeholder tersisa.
5. Periksa `submission/requirements.txt` (berasal dari Colab). Jangan menulis ulang versinya dari tebakan.
6. `python scripts/pack_submission.py zip` → harus 0 FAIL dan saran 6/6.
7. Kirim **laporan akhir** (format di bawah).

---

## Larangan
- Mengarang output, angka, screenshot, atau mengklaim notebook sudah dijalankan.
- Mengubah sel kode atau output setelah notebook dijalankan di Colab.
- Mengubah, menghapus, atau menyisipkan heading templat.
- Memasukkan dataset, checkpoint `.keras`/`.h5`, venv, atau `__pycache__` ke folder submission atau zip.
- `pip install tensorflowjs` di runtime utama, atau memakai flip pada augmentasi.
- Mengganti dataset, mode kerja, atau memakai transfer learning tanpa persetujuan user.
- Menjalankan training penuh di komputer lokal tanpa diminta.
- Menghapus file user tanpa cadangan. Melakukan commit/push git tanpa diminta.

## Format laporan akhir (Fase C)
```
## Hasil Submission
| Metrik | Nilai |
|---|---|
| Total gambar / kelas / resolusi unik | ... / 43 / ... |
| Split train / val / test | ... / ... / ... (..% / ..% / ..%) |
| Akurasi train / val / test | ..% / ..% / ..% |
| Akurasi TF-Lite (test) | ..% |

## Kriteria wajib (7/7)
- [x] 1. >= 1.000 gambar  ... (sertakan bukti singkat tiap butir)

## Saran bintang 5 (6/6)
- [x] Callback ... (bukti)

## File
submission.zip (.. MB) -> daftar isi singkat

## Langkah terakhir untukmu
1. Buka submission.zip, pastikan isinya folder submission/.
2. Upload di halaman submission kelas Belajar Fundamental Deep Learning.
3. Hindari submit berulang. Review maksimal 3 hari kerja.
```
