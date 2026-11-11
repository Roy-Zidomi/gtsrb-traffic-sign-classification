# 05 — Template README.md Submission

Agent menyalin template di bawah ke `submission/README.md` pada tahap finalisasi, lalu mengganti setiap
`{{...}}` dengan **angka nyata dari output notebook** (lihat `JSON_SUMMARY` dari
`python scripts/check_submission.py --json`) dan **nilai konfigurasi yang benar-benar dipakai** di notebook
(IMG_SIZE, BATCH_SIZE, EPOCHS, arsitektur). Jangan biarkan `{{` tersisa, karena checker akan memberi FAIL.
Jika arsitektur atau augmentasi diubah dari kode referensi, sesuaikan juga kalimat deskripsinya.

---

````markdown
# Klasifikasi Rambu Lalu Lintas (GTSRB) dengan CNN

Submission **Proyek Klasifikasi Gambar** untuk kelas *Belajar Fundamental Deep Learning* (Dicoding).

- **Nama:** {{NAMA}}
- **Email:** {{EMAIL}}
- **ID Dicoding:** {{ID_DICODING}}

## Ringkasan
Model CNN `Sequential` (Conv2D + MaxPooling2D) mengklasifikasikan **43 jenis rambu lalu lintas Jerman**
dari dataset GTSRB. Model diekspor ke SavedModel, TF-Lite, dan TensorFlow.js.

| Metrik | Nilai |
|---|---|
| Akurasi train (tanpa augmentasi) | {{TRAIN_ACCURACY_%}} |
| Akurasi validation | {{VAL_ACCURACY_%}} |
| Akurasi test | {{TEST_ACCURACY_%}} |
| Akurasi TF-Lite pada test set | {{TFLITE_TEST_ACCURACY_%}} |

## Dataset
- **Sumber:** German Traffic Sign Recognition Benchmark (GTSRB). Diunduh dari Kaggle
  (`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`). Situs resmi: https://benchmark.ini.rub.de/gtsrb_dataset.html
- **Jumlah:** {{TOTAL_IMAGES}} gambar, {{NUM_CLASSES}} kelas.
- **Resolusi asli tidak seragam:** {{UNIQUE_RESOLUTIONS}} resolusi unik ({{RENTANG_RESOLUSI}}).
- **Lisensi/sitasi:** data bebas digunakan. Sitasi: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel,
  "The German Traffic Sign Recognition Benchmark: A multi-class classification competition", IJCNN 2011.

## Pembagian Data
| Subset | Jumlah | Persentase | Sumber |
|---|---|---|---|
| Train | {{N_TRAIN}} | {{P_TRAIN}} | Folder `Train` (track terpilih) |
| Validation | {{N_VAL}} | {{P_VAL}} | Folder `Train` (±15% track per kelas, tidak beririsan dengan train) |
| Test | {{N_TEST}} | {{P_TEST}} | Folder `Test` resmi + `Test.csv` |

## Pipeline dan Model
- **Preprocessing:** baca gambar sebagai RGB, resize ke {{IMG_SIZE}}x{{IMG_SIZE}}. Normalisasi `Rescaling(1/255)` ada di dalam model.
- **Augmentasi (train saja):** rotasi ±18°, zoom ±10%, translasi ±10%, kontras ±20%. Tanpa flip karena
  flip dapat mengubah makna rambu.
- **Arsitektur:** Conv2D 32-32 > MaxPooling2D > Conv2D 64-64 > MaxPooling2D > Conv2D 128 > MaxPooling2D >
  Flatten > Dense 256 > Softmax 43, dengan BatchNormalization dan Dropout. ({{TOTAL_PARAMS}} parameter)
- **Training:** Adam (lr 1e-3), sparse categorical crossentropy, batch {{BATCH_SIZE}}, maksimal {{EPOCHS}} epoch.
  Berhenti pada epoch {{EPOCH_BERHENTI}}.
- **Callback:** `TargetAccuracyStop` (kustom), `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`.

## Struktur Folder
```
submission
├── tfjs_model/        model.json + group1-shard1of1.bin (TensorFlow.js graph model)
├── tflite/            model.tflite + label.txt (43 label, urut indeks kelas)
├── saved_model/       saved_model.pb + variables/
├── notebook.ipynb     notebook lengkap beserta output
├── notebook.py        versi skrip dari notebook
├── README.md
└── requirements.txt
```

## Cara Memakai Model
Semua format menerima input **float32 `[N, {{IMG_SIZE}}, {{IMG_SIZE}}, 3]`, RGB, rentang piksel 0–255** (normalisasi ada di
dalam model). Output berupa probabilitas 43 kelas, dan urutan labelnya ada di `tflite/label.txt`.

**SavedModel (Python)**
```python
import tensorflow as tf
model = tf.saved_model.load("saved_model")
probs = model.serve(images)            # images: float32 [N, {{IMG_SIZE}}, {{IMG_SIZE}}, 3]
```

**TF-Lite (Python)**
```python
from ai_edge_litert.interpreter import Interpreter   # atau: tf.lite.Interpreter
interpreter = Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
interpreter.set_tensor(inp["index"], image_batch)      # float32 [1, {{IMG_SIZE}}, {{IMG_SIZE}}, 3]
interpreter.invoke()
probs = interpreter.get_tensor(out["index"])
```

**TensorFlow.js (browser)**
```javascript
const model = await tf.loadGraphModel("tfjs_model/model.json");
const input = tf.image.resizeBilinear(tf.browser.fromPixels(imgElement), [{{IMG_SIZE}}, {{IMG_SIZE}}]).toFloat().expandDims(0);
const probs = model.predict(input);
```

## Menjalankan Ulang
Buka `notebook.ipynb` di Google Colab, pilih Runtime T4 GPU, lalu Run all. Dataset diunduh otomatis lewat `kagglehub`.

## Dependensi
Lihat `requirements.txt`. Konversi TFJS dijalankan di virtual environment terpisah (`tensorflowjs==4.22.0`,
`ydf==0.12.0`, `setuptools<81`) karena tensorflowjs mengunci `tensorflow==2.19.0`.
````
