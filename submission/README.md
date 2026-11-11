# Klasifikasi Rambu Lalu Lintas (GTSRB) dengan CNN

Submission **Proyek Klasifikasi Gambar** untuk kelas *Belajar Fundamental Deep Learning* (Dicoding).

- **Nama:** Roy Aldonit Zidomi
- **Email:** zidomiroy@gmail.com
- **ID Dicoding:** royzidomi15

## Ringkasan
Model CNN `Sequential` (Conv2D + MaxPooling2D) mengklasifikasikan **43 jenis rambu lalu lintas Jerman**
dari dataset GTSRB. Model diekspor ke SavedModel, TF-Lite, dan TensorFlow.js.

| Metrik | Nilai |
|---|---|
| Akurasi train (tanpa augmentasi) | 99,98% |
| Akurasi validation | 99,62% |
| Akurasi test | 98,37% |
| Akurasi TF-Lite pada test set | 98,37% |

## Dataset
- **Sumber:** German Traffic Sign Recognition Benchmark (GTSRB). Diunduh dari Kaggle
  (`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign`). Situs resmi: https://benchmark.ini.rub.de/gtsrb_dataset.html
- **Jumlah:** 51.839 gambar, 43 kelas.
- **Resolusi asli tidak seragam:** 3.338 resolusi unik (lebar 25–266 px, tinggi 25–232 px).
- **Lisensi/sitasi:** data bebas digunakan. Sitasi: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel,
  "The German Traffic Sign Recognition Benchmark: A multi-class classification competition", IJCNN 2011.

## Pembagian Data
| Subset | Jumlah | Persentase | Sumber |
|---|---|---|---|
| Train | 33.359 | 64,35% | Folder `Train` (track terpilih) |
| Validation | 5.850 | 11,28% | Folder `Train` (±15% track per kelas, tidak beririsan dengan train) |
| Test | 12.630 | 24,36% | Folder `Test` resmi + `Test.csv` |

## Pipeline dan Model
- **Preprocessing:** baca gambar sebagai RGB, resize ke 48x48. Normalisasi `Rescaling(1/255)` ada di dalam model.
- **Augmentasi (train saja):** rotasi ±18°, zoom ±10%, translasi ±10%, kontras ±20%. Tanpa flip karena
  flip dapat mengubah makna rambu.
- **Arsitektur:** Conv2D 32-32 > MaxPooling2D > Conv2D 64-64 > MaxPooling2D > Conv2D 128 > MaxPooling2D >
  Flatten > Dense 256 > Softmax 43, dengan BatchNormalization dan Dropout. (1.332.683 parameter)
- **Training:** Adam (lr 1e-3), sparse categorical crossentropy, batch 64, maksimal 30 epoch.
  Berhenti pada epoch 14 oleh `TargetAccuracyStop`, saat akurasi train dan validation sama-sama melewati 99%.
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
Semua format menerima input **float32 `[N, 48, 48, 3]`, RGB, rentang piksel 0–255** (normalisasi ada di
dalam model). Output berupa probabilitas 43 kelas, dan urutan labelnya ada di `tflite/label.txt`.

**SavedModel (Python)**
```python
import tensorflow as tf
model = tf.saved_model.load("saved_model")
probs = model.serve(images)            # images: float32 [N, 48, 48, 3]
```

**TF-Lite (Python)**
```python
from ai_edge_litert.interpreter import Interpreter   # atau: tf.lite.Interpreter
interpreter = Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
interpreter.set_tensor(inp["index"], image_batch)      # float32 [1, 48, 48, 3]
interpreter.invoke()
probs = interpreter.get_tensor(out["index"])
```

**TensorFlow.js (browser)**
```javascript
const model = await tf.loadGraphModel("tfjs_model/model.json");
const input = tf.image.resizeBilinear(tf.browser.fromPixels(imgElement), [48, 48]).toFloat().expandDims(0);
const probs = model.predict(input);
```

## Menjalankan Ulang
Buka `notebook.ipynb` di Google Colab, pilih Runtime T4 GPU, lalu Run all. Dataset diunduh otomatis lewat `kagglehub`.
Hasil di atas diperoleh di Google Colab (GPU T4) dengan Python 3.13.16, TensorFlow 2.21.0, dan Keras 3.13.2.

## Dependensi
Lihat `requirements.txt`. Konversi TFJS dijalankan di virtual environment terpisah (`tensorflowjs==4.22.0`,
`ydf==0.12.0`, `setuptools<81`) karena tensorflowjs mengunci `tensorflow==2.19.0`.
