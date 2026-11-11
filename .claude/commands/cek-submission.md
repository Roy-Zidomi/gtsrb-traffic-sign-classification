---
description: Jalankan pemeriksa otomatis submission lalu jelaskan hasilnya (tanpa mengubah file)
argument-hint: "[pre|final]"
---
Tahap yang diminta: $ARGUMENTS (jika kosong: `final`).

1. Jalankan `python scripts/check_submission.py --stage <tahap> --json`. Untuk tahap final, tambahkan
   `--zip submission.zip` jika file itu ada. Pakai `py` atau `python3` bila `python` tidak dikenali.
2. Jelaskan hasilnya dalam Bahasa Indonesia:
   - setiap FAIL: penyebab dan langkah perbaikan konkret;
   - WARN yang relevan;
   - metrik utama (jika ada) dan estimasi bintang.
3. Jangan mengubah file apa pun kecuali user memintanya.
