# 03 — Panduan Teknis & Kode Referensi Teruji

## 0. Status pengujian
Pipeline di Lampiran (bagian 4) sudah diuji end-to-end pada 11 November 2026 (Linux, Python 3.12):

- **Runtime utama:** TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.5.4, pandas 3.1.1, ai-edge-litert 2.3.0, kagglehub 1.0.3.
- **Venv konverter TFJS:** tensorflowjs 4.22.0, yang menarik TensorFlow 2.19.0 dan tensorflow-decision-forests 1.12.0,
  ditambah pin `ydf==0.12.0` dan `setuptools<81`.
- **Data uji:** replika struktur dataset Kaggle GTSRB (`Train/<ClassId>/...`, `Train.csv`, `Test/`, `Test.csv`,
  folder duplikat) berisi 3.600 gambar sintetis berukuran acak 15–127 px. Kaggle tidak dapat diakses dari
  lingkungan uji, jadi data asli tidak ikut diuji.
- **Hasil uji:** semua sel berjalan tanpa error. Prediksi SavedModel sama dengan Keras. Prediksi TF-Lite 100% sama
  dengan Keras. Model TFJS (graph model) berhasil dimuat di `@tensorflow/tfjs` 4.22.0 (Node.js) dengan
  prediksi identik (selisih probabilitas maksimum 2e-7). Checker final 0 FAIL dan zip valid.
- **Belum diuji:** dataset GTSRB asli dan runtime Colab. Struktur Kaggle (kolom CSV
  `Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path`, contoh path `Train/13/00013_00001_00009.png`)
  dikonfirmasi dari sumber publik. Akurasi nyata baru diketahui setelah "Run all" di Colab.

## 1. Keputusan teknis dan alasannya

| Aspek | Keputusan | Alasan |
|---|---|---|
| Sumber data | `kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")` | Dataset publik, tanpa login, cepat di Colab |
| Label | Dari kolom `ClassId` di `Train.csv`/`Test.csv` | Menghindari salah urut folder `"0","1","10",...` |
| Ukuran input | 48x48 RGB | Rambu terkecil 15 px. 48 px cukup detail dan cepat dilatih |
| Normalisasi | `layers.Rescaling(1/255)` di dalam model | Model ekspor menerima piksel mentah 0–255, jadi inference TF-Lite/TFJS sederhana |
| Split | Validation = ±15% *track* per kelas dari folder Train. Test = folder Test resmi (12.630 gambar) | 1 track = ±30 foto beruntun dari rambu fisik yang sama. Split acak per gambar membocorkan data |
| Augmentasi | Train saja: rotasi ±18°, zoom ±10%, translasi ±10%, kontras ±20%. **Tanpa flip** | Flip mengubah makna rambu (kiri <-> kanan) |
| Model | `Sequential`: Conv2D 32-32-64-64-128 + BatchNorm + 3x MaxPooling2D + Dropout, Flatten, Dense 256, softmax | ±1,3 juta parameter (±5 MB per format) |
| Callback | `TargetAccuracyStop` (kustom, berhenti bila accuracy & val_accuracy >= 99%), `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` | Memenuhi saran callback dan mencegah overfitting |
| Akurasi train | `evaluate` pada `train_eval_ds` (tanpa augmentasi & shuffle) | Akurasi log epoch dihitung saat dropout & augmentasi aktif, jadi lebih rendah |
| SavedModel | `best_model.export(...)` (Keras 3) | `model.save("folder")` tidak didukung Keras 3 |
| TF-Lite | `TFLiteConverter.from_saved_model(...)` + `label.txt` urut indeks kelas | Konversi stabil dari SavedModel |
| TFJS | `tensorflowjs_converter` (tf_saved_model -> tfjs_graph_model) di **venv terpisah**, `--weight_shard_size_bytes=67108864` | Menghindari konflik versi TF dan menghasilkan satu shard `group1-shard1of1.bin` |
| Inference | LiteRT (`ai_edge_litert`) bila ada, fallback `tf.lite.Interpreter` | `tf.lite.Interpreter` sudah deprecated |
| requirements.txt | Ditulis dari `importlib.metadata` saat notebook berjalan | Versi persis sama dengan lingkungan Colab |

## 2. Jebakan teknis (gejala -> penyebab -> solusi)
1. **`pip install tensorflowjs` di runtime Colab** -> versi TensorFlow berubah atau muncul error import aneh ->
   tensorflowjs 4.22.0 menarik tensorflow-decision-forests 1.12.0 yang mengunci `tensorflow==2.19.0` (wheel hanya
   Linux/macOS, Python <= 3.12) -> jalankan konverter di venv terpisah (`uv venv --python 3.12`), seperti di Lampiran.
2. **`VersionError: Detected incompatible Protobuf Gencode/Runtime versions ... yggdrasil_decision_forests`** ->
   `ydf` terbaru dibangun dengan protobuf 6 sedangkan TF 2.19 memakai protobuf 5 -> pin `ydf==0.12.0`.
3. **`ModuleNotFoundError: No module named 'pkg_resources'`** (dari `tensorflow_hub`) -> venv buatan `uv` tidak berisi
   setuptools, dan setuptools terbaru (84.x) sudah tidak menyertakan `pkg_resources` -> pasang `setuptools<81`
   (teruji: 80.10.2).
4. **`ValueError` pada `model.save("folder")`** -> Keras 3 hanya menyimpan `.keras`/`.h5` -> pakai `model.export(path)`.
5. **Peringatan `tf.lite.Interpreter is deprecated`** -> pakai `ai_edge_litert.interpreter.Interpreter` bila ada
   (kode referensi sudah fallback otomatis).
6. **Label tertukar dengan `image_dataset_from_directory`** pada folder `0..42` -> urutan string ≠ ClassId -> pakai
   `ClassId` dari CSV (kode referensi), atau petakan lewat `class_names`.
7. **`val_accuracy` rendah di 1–3 epoch pertama lalu naik** -> statistik BatchNormalization belum stabil. Ini normal.
   Jika tertahan di level tebakan (±2% untuk 43 kelas) lebih dari 5 epoch, curigai bug pipeline (label/preprocessing).
8. **`UserWarning: shuffle=True was passed, but will be ignored ...`** -> aman diabaikan, data sudah diacak di `tf.data`.
9. **Flip pada augmentasi** -> makna rambu berubah (Turn left <-> Turn right, Keep left <-> Keep right) -> jangan dipakai.
10. **Folder duplikat** di dataset Kaggle (`train-1/`, `test-1/`, `meta-1/` atau huruf kecil) dan entri non-gambar
    di `Test/` -> abaikan, selalu pakai path dari `Train.csv`/`Test.csv`.
11. **`kagglehub` meminta login / error 401/403** -> buat API token di kaggle.com (Settings > API > Create New Token).
    Di Colab, tambahkan Secrets `KAGGLE_USERNAME` dan `KAGGLE_KEY` (ikon kunci di sidebar, aktifkan Notebook access),
    lalu jalankan ulang.
12. **Unduhan `hasil_colab_artifacts.zip` tidak muncul** -> izinkan unduhan di browser, atau unduh manual dari panel
    Files (ikon folder) > klik kanan file > Download.
13. **Colab tanpa GPU (kuota habis)** -> notebook tetap jalan di CPU, hanya lebih lambat. Kode tidak perlu diubah.
14. **Windows: `python` tidak dikenali** -> coba `py` atau `python3`. Script `scripts/` hanya butuh library standar.
15. **Mengedit kode langsung di Colab** -> versi tidak sinkron dengan VS Code. Semua perubahan kode dilakukan di
    VS Code, lalu notebook di-upload ulang.
16. **Notebook berisi output dari run lama/parsial** -> sebelum unduh final pakai Runtime > Restart session and run all
    agar `execution_count` berurutan 1..N.

## 3. Strategi jika akurasi train/test < 95%
Ubah **satu hal per iterasi** dan catat di `CATATAN_EKSPERIMEN.md` (root proyek, bukan di folder submission).
0. Pastikan bukan bug: loss train turun? label cocok di grid contoh? augmentasi tidak masuk ke val/test?
   akurasi train dihitung dengan `train_eval_ds`?
1. Tambah epoch: `EPOCHS = 50`, `EarlyStopping(patience=10)`.
2. Naikkan resolusi: `IMG_SIZE = 64`.
3. Underfitting (train < 95%): kurangi augmentasi/dropout, tambah kapasitas (Conv2D 128 kedua atau Dense 512).
4. Overfitting (train tinggi, test rendah): augmentasi sedikit lebih kuat, Dropout lebih besar, atau `class_weight`.
5. Learning rate awal 5e-4 atau `BATCH_SIZE = 32`.
6. Jalan terakhir, **hanya dengan persetujuan user**: transfer learning (mis. MobileNetV2) di dalam `Sequential`,
   ditambah `Conv2D` + `MaxPooling2D` milik sendiri agar kriteria 4 tetap jelas terpenuhi.

## 4. Lampiran — kode referensi teruji (urut sesuai templat)
Gunakan sebagai basis notebook. Boleh menambah markdown penjelasan dan merapikan, tetapi bagian kritis berikut
jangan diubah tanpa alasan kuat: pembacaan CSV & `ClassId`, split per-track, `train_eval_ds`, `Rescaling` di dalam
model, ekspor `model.export`, konversi TFJS di venv terpisah (beserta pin), dan baris penanda (marker).

Format: setiap blok adalah satu sel notebook, berurutan dari atas ke bawah.

**Sel 1 — markdown**

```markdown
# Proyek Klasifikasi Gambar: German Traffic Sign Recognition Benchmark (GTSRB)
- **Nama:** <Nama dari CLAUDE.md>
- **Email:** <Email dari CLAUDE.md>
- **ID Dicoding:** <ID Dicoding dari CLAUDE.md>
```

**Sel 2 — markdown**

```markdown
## Import Semua Packages/Library yang Digunakan
```

**Sel 3 — kode**

```python
# kagglehub biasanya sudah terpasang di Colab; sel ini hanya memasangnya bila belum ada
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("kagglehub") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
```

**Sel 4 — kode**

```python
# ====== Library standar ======
import os
import shutil
import importlib.metadata as importlib_metadata
from pathlib import Path

# ====== Analisis data & visualisasi ======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# ====== Deep learning ======
import tensorflow as tf
import keras
from keras import layers, callbacks

# ====== Sumber dataset ======
import kagglehub

IN_COLAB = "google.colab" in sys.modules

print(f"Python      : {sys.version.split()[0]}")
print(f"TensorFlow  : {tf.__version__}")
print(f"Keras       : {keras.__version__}")
print(f"GPU         : {tf.config.list_physical_devices('GPU') or 'tidak ada (CPU)'}")
print(f"Lingkungan  : {'Google Colab' if IN_COLAB else 'Lokal'}")
```

**Sel 5 — kode**

```python
# ====== Konfigurasi global ======
SEED = 42
IMG_SIZE = 48            # resolusi asli 15-250 px diseragamkan menjadi 48x48 di pipeline data
BATCH_SIZE = 64
EPOCHS = 30              # batas atas; callback menghentikan training lebih awal
VAL_FRACTION = 0.15      # porsi track dari folder Train yang dijadikan validation set
TARGET_ACCURACY = 0.95   # target akurasi train & test (saran bintang 5)

SUBMISSION_DIR = Path("submission")
SAVED_MODEL_DIR = SUBMISSION_DIR / "saved_model"
TFLITE_DIR = SUBMISSION_DIR / "tflite"
TFJS_DIR = SUBMISSION_DIR / "tfjs_model"
CHECKPOINT_PATH = "best_model.keras"

keras.utils.set_random_seed(SEED)   # seed untuk random, NumPy, dan TensorFlow
```

**Sel 6 — markdown**

```markdown
## Data Preparation
```

**Sel 7 — markdown**

```markdown
### Data Loading
```

**Sel 8 — kode**

```python
# Unduh dataset GTSRB dari Kaggle (dataset publik, tidak perlu login)
DATASET_HANDLE = "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
dataset_root = Path(kagglehub.dataset_download(DATASET_HANDLE))

print("Lokasi dataset:", dataset_root)
for item in sorted(dataset_root.iterdir()):
    info = f"{sum(1 for _ in item.iterdir())} item" if item.is_dir() else f"{item.stat().st_size / 1e6:.2f} MB"
    print(f"  - {item.name:<10} {info}")
```

**Sel 9 — kode**

```python
GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]
```

**Sel 10 — kode**

```python
def resolve_path(root: Path, name: str) -> Path:
    """Cari file/folder `name` di root: nama persis dulu, lalu tanpa membedakan huruf besar/kecil."""
    exact = root / name
    if exact.exists():
        return exact
    for candidate in root.rglob("*"):
        if candidate.name.lower() == name.lower():
            return candidate
    raise FileNotFoundError(f"'{name}' tidak ditemukan di {root}")


def load_annotation(csv_name: str) -> pd.DataFrame:
    """Baca Train.csv / Test.csv dan ubah kolom Path menjadi path absolut."""
    csv_path = resolve_path(dataset_root, csv_name)
    table = pd.read_csv(csv_path)
    table["path"] = [str(csv_path.parent / p) for p in table["Path"]]
    table["label"] = table["ClassId"].astype(int)
    return table


train_full_df = load_annotation("Train.csv")
test_df = load_annotation("Test.csv")

# Nama file Train: <kelas>_<track>_<frame>.png -> token kedua dari belakang adalah nomor track.
# Satu track = foto beruntun dari rambu fisik yang sama (dipakai saat split agar tidak bocor).
train_full_df["track"] = [int(Path(p).stem.split("_")[-2]) for p in train_full_df["path"]]

missing = [p for p in pd.concat([train_full_df["path"], test_df["path"]]) if not os.path.exists(p)]
assert not missing, f"{len(missing)} file tidak ditemukan, contoh: {missing[:3]}"

NUM_CLASSES = train_full_df["label"].nunique()
assert sorted(train_full_df["label"].unique()) == list(range(NUM_CLASSES)), "ClassId harus 0..N-1"
CLASS_NAMES = GTSRB_CLASS_NAMES[:NUM_CLASSES]

print(f"Gambar folder Train: {len(train_full_df):,} | Gambar folder Test: {len(test_df):,} | Kelas: {NUM_CLASSES}")
print(f"Jumlah track per kelas: {train_full_df.groupby('label')['track'].nunique().min()} - "
      f"{train_full_df.groupby('label')['track'].nunique().max()}")
train_full_df[["path", "label", "track", "Width", "Height"]].head()
```

**Sel 11 — kode**

```python
def inspect_images(paths) -> pd.DataFrame:
    """Baca resolusi (lebar, tinggi) tiap gambar dan cek apakah file bisa dibuka (tidak korup)."""
    records = []
    for path in paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                img.verify()
            records.append((width, height, True))
        except Exception:
            records.append((None, None, False))
    return pd.DataFrame(records, columns=["width", "height", "valid"]).astype(
        {"width": "Int64", "height": "Int64", "valid": bool})


image_cols = ["width", "height", "valid"]
train_full_df = train_full_df.drop(columns=image_cols, errors="ignore").join(inspect_images(train_full_df["path"]))
test_df = test_df.drop(columns=image_cols, errors="ignore").join(inspect_images(test_df["path"]))
all_images = pd.concat([train_full_df, test_df], ignore_index=True)
resolutions = all_images.loc[all_images["valid"], ["width", "height"]].astype(int)

print("=== RINGKASAN DATASET ===")
print(f"TOTAL_IMAGES: {len(all_images)}")
print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"UNIQUE_RESOLUTIONS: {len(resolutions.drop_duplicates())}")
print(f"Lebar  : {resolutions['width'].min()} - {resolutions['width'].max()} px")
print(f"Tinggi : {resolutions['height'].min()} - {resolutions['height'].max()} px")
print(f"Gambar korup/tidak terbaca: {int((~all_images['valid']).sum())}")

per_class = (train_full_df.assign(resolusi=train_full_df["width"].astype(str) + "x" + train_full_df["height"].astype(str))
             .groupby("label")
             .agg(jumlah_train=("path", "size"), resolusi_unik=("resolusi", "nunique"),
                  lebar_min=("width", "min"), lebar_max=("width", "max")))
per_class["jumlah_test"] = test_df["label"].value_counts().sort_index()
per_class.insert(0, "nama_kelas", CLASS_NAMES)
per_class
```

**Sel 12 — kode**

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})

per_class["jumlah_train"].plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("Jumlah gambar per kelas (folder Train)")
axes[0].set_xlabel("ClassId")
axes[0].set_ylabel("Jumlah gambar")

axes[1].scatter(resolutions["width"], resolutions["height"], s=3, alpha=0.25)
axes[1].set_title("Sebaran resolusi asli (tidak seragam)")
axes[1].set_xlabel("Lebar (px)")
axes[1].set_ylabel("Tinggi (px)")
plt.tight_layout()
plt.show()

samples = train_full_df.groupby("label").sample(1, random_state=SEED)
n_cols = 9
n_rows = int(np.ceil(len(samples) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 2.0))
for ax in np.ravel(axes):
    ax.axis("off")
for ax, (_, row) in zip(np.ravel(axes), samples.iterrows()):
    with Image.open(row["path"]) as img:
        ax.imshow(img)
        ax.set_title(f"{row['label']} | {img.size[0]}x{img.size[1]}", fontsize=8)
plt.suptitle("Satu contoh per kelas: ClassId | resolusi asli")
plt.tight_layout()
plt.show()
```

**Sel 13 — markdown**

```markdown
### Data Preprocessing
```

**Sel 14 — kode**

```python
AUTOTUNE = tf.data.AUTOTUNE


def load_and_resize(path: tf.Tensor) -> tf.Tensor:
    """Baca file -> RGB -> resize ke IMG_SIZE x IMG_SIZE (uint8, rentang 0-255)."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


# Augmentasi HANYA untuk data latih. Tidak memakai flip karena membalik rambu
# dapat mengubah maknanya (mis. "Turn left ahead" <-> "Turn right ahead").
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.05, seed=SEED),          # +-18 derajat
    layers.RandomZoom(0.10, seed=SEED),
    layers.RandomTranslation(0.10, 0.10, seed=SEED),
    layers.RandomContrast(0.20, seed=SEED),
], name="data_augmentation")

example = load_and_resize(tf.constant(samples.iloc[0]["path"]))
fig, axes = plt.subplots(1, 8, figsize=(14, 2.2))
for i, ax in enumerate(axes):
    shown = example if i == 0 else data_augmentation(tf.cast(example, tf.float32)[None, ...], training=True)[0]
    ax.imshow(tf.cast(shown, tf.uint8))
    ax.set_title("asli (48x48)" if i == 0 else f"augmentasi {i}", fontsize=8)
    ax.axis("off")
plt.show()
```

**Sel 15 — markdown**

```markdown
#### Split Dataset
```

**Sel 16 — kode**

```python
def split_by_track(df: pd.DataFrame, val_fraction: float, seed: int = SEED):
    """Bagi data folder Train menjadi train/validation per kelas berdasarkan track.

    Satu track berisi foto beruntun dari rambu fisik yang sama, jadi satu track utuh
    dimasukkan ke satu subset saja untuk mencegah kebocoran data (data leakage).
    """
    rng = np.random.default_rng(seed)
    groups = df["label"].astype(str) + "_" + df["track"].astype(str)
    val_groups = set()
    for label, group in df.groupby("label"):
        tracks = np.sort(group["track"].unique())
        rng.shuffle(tracks)
        n_val = max(1, int(round(len(tracks) * val_fraction)))
        val_groups.update(f"{label}_{t}" for t in tracks[:n_val])
    is_val = groups.isin(val_groups).values
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


train_df, val_df = split_by_track(train_full_df, VAL_FRACTION)

# Pastikan tidak ada gambar yang sama di dua subset
assert not set(train_df["path"]) & set(val_df["path"])
assert not set(train_df["path"]) & set(test_df["path"])

total = len(train_df) + len(val_df) + len(test_df)
split_summary = pd.DataFrame({"jumlah_gambar": [len(train_df), len(val_df), len(test_df)]},
                             index=["train", "validation", "test"])
split_summary["persentase"] = (split_summary["jumlah_gambar"] / total * 100).round(2)
print(split_summary.to_string())
print(f"SPLIT_COUNTS: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
```

**Sel 17 — kode**

```python
def make_dataset(df: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
    """Pipeline tf.data: baca -> resize -> cache -> (shuffle + augmentasi) -> batch -> prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values.astype("int32")))
    ds = ds.map(lambda p, y: (load_and_resize(p), y), num_parallel_calls=AUTOTUNE).cache()
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(tf.cast(x, tf.float32), training=True), y),
                    num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)
train_eval_ds = make_dataset(train_df)   # tanpa augmentasi: untuk mengukur akurasi training akhir

images, labels = next(iter(train_ds))
print("Bentuk batch :", images.shape, labels.shape)
print("Rentang piksel:", float(tf.reduce_min(images)), "-", float(tf.reduce_max(images)))
```

**Sel 18 — markdown**

```markdown
## Modelling
```

**Sel 19 — kode**

```python
def build_model(num_classes: int) -> keras.Sequential:
    """CNN Sequential: blok Conv2D + MaxPooling2D, lalu Dense untuk klasifikasi."""
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image"),
        layers.Rescaling(1.0 / 255),              # normalisasi di dalam model -> input mentah 0-255

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="gtsrb_cnn")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = build_model(NUM_CLASSES)
model.summary()
```

**Sel 20 — kode**

```python
class TargetAccuracyStop(callbacks.Callback):
    """Callback kustom: hentikan training saat akurasi train dan validasi sudah >= target."""

    def __init__(self, target: float = 0.99):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc, val_acc = logs.get("accuracy", 0.0), logs.get("val_accuracy", 0.0)
        if acc >= self.target and val_acc >= self.target:
            print(f"\n[TargetAccuracyStop] epoch {epoch + 1}: accuracy={acc:.4f}, "
                  f"val_accuracy={val_acc:.4f} >= {self.target:.0%} -> training dihentikan")
            self.model.stop_training = True


training_callbacks = [
    TargetAccuracyStop(target=0.99),
    callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=training_callbacks,
    verbose=2,
)

best_model = keras.models.load_model(CHECKPOINT_PATH)   # bobot dengan val_accuracy terbaik
print("Model terbaik dimuat dari", CHECKPOINT_PATH)
```

**Sel 21 — markdown**

```markdown
## Evaluasi dan Visualisasi
```

**Sel 22 — kode**

```python
hist = pd.DataFrame(history.history)
hist.index = np.arange(1, len(hist) + 1)
best_epoch = int(hist["val_accuracy"].idxmax())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in [(axes[0], "accuracy", "Akurasi"), (axes[1], "loss", "Loss")]:
    ax.plot(hist.index, hist[metric], marker="o", label=f"train {metric}")
    ax.plot(hist.index, hist[f"val_{metric}"], marker="o", label=f"validation {metric}")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"epoch terbaik ({best_epoch})")
    ax.set_title(f"{title} per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.show()
```

**Sel 23 — kode**

```python
results = {}
for name, ds in [("train", train_eval_ds), ("validation", val_ds), ("test", test_ds)]:
    loss, acc = best_model.evaluate(ds, verbose=0)
    results[name] = {"loss": loss, "accuracy": acc}
print(pd.DataFrame(results).T.round(4).to_string())

print(f"\nTRAIN_ACCURACY: {results['train']['accuracy']:.4f}")
print(f"VAL_ACCURACY: {results['validation']['accuracy']:.4f}")
print(f"TEST_ACCURACY: {results['test']['accuracy']:.4f}")
target_met = all(results[s]["accuracy"] >= TARGET_ACCURACY for s in ("train", "test"))
print(f"Target >= {TARGET_ACCURACY:.0%} pada train & test: {'TERCAPAI' if target_met else 'BELUM TERCAPAI'}")
```

**Sel 24 — kode**

```python
y_true = test_df["label"].values
y_prob = best_model.predict(test_ds, verbose=0)
y_pred = y_prob.argmax(axis=1)

print(classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                            target_names=CLASS_NAMES, digits=4, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)), normalize="true")
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(cm, cmap="Blues", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "proporsi"})
ax.set_title("Confusion Matrix Test Set (dinormalisasi per label asli)")
ax.set_xlabel("Prediksi (ClassId)")
ax.set_ylabel("Label asli (ClassId)")
plt.tight_layout()
plt.show()
```

**Sel 25 — markdown**

```markdown
## Konversi Model
```

**Sel 26 — kode**

```python
for directory in (SAVED_MODEL_DIR, TFLITE_DIR, TFJS_DIR):
    shutil.rmtree(directory, ignore_errors=True)
SUBMISSION_DIR.mkdir(exist_ok=True)

# 1) SavedModel (Keras 3: model.export, bukan model.save)
best_model.export(str(SAVED_MODEL_DIR))

reloaded = tf.saved_model.load(str(SAVED_MODEL_DIR))
sample_batch, _ = next(iter(test_ds))
same = np.allclose(reloaded.serve(sample_batch).numpy(), best_model.predict(sample_batch, verbose=0), atol=1e-4)
print("Prediksi SavedModel sama dengan model Keras:", same)
```

**Sel 27 — kode**

```python
# 2) TF-Lite + label.txt
converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
tflite_model = converter.convert()

TFLITE_DIR.mkdir(parents=True, exist_ok=True)
(TFLITE_DIR / "model.tflite").write_bytes(tflite_model)
(TFLITE_DIR / "label.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
print(f"model.tflite: {len(tflite_model) / 1e6:.2f} MB | label.txt: {len(CLASS_NAMES)} kelas")
```

**Sel 28 — kode**

```python
# 3) TensorFlow.js
# tensorflowjs 4.22 menarik tensorflow-decision-forests yang mengunci tensorflow==2.19. Menginstalnya
# langsung di runtime ini bisa mengganti versi TensorFlow yang sedang dipakai, jadi konverter
# dijalankan di virtual environment terpisah. Dua pin tambahan diperlukan agar konverter bisa jalan:
#   - ydf==0.12.0     : ydf terbaru butuh protobuf 6, sedangkan tensorflow 2.19 memakai protobuf 5
#   - setuptools<81   : tensorflow_hub masih mengimpor pkg_resources
def run_command(cmd, env=None) -> str:
    """Jalankan perintah shell; tampilkan output bila gagal lalu hentikan notebook."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print((result.stdout + result.stderr)[-5000:])
        raise RuntimeError(f"Perintah gagal (exit code {result.returncode})")
    return result.stdout


TFJS_ENV = Path("/content/tfjs_env") if IN_COLAB else Path(".tfjs_env").resolve()
tfjs_converter = TFJS_ENV / "bin" / "tensorflowjs_converter"

if not tfjs_converter.exists():
    if shutil.which("uv"):
        uv = [shutil.which("uv")]
    else:
        run_command([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = [sys.executable, "-m", "uv"]
    shutil.rmtree(TFJS_ENV, ignore_errors=True)
    run_command(uv + ["venv", "--python", "3.12", TFJS_ENV])
    run_command(uv + ["pip", "install", "--python", TFJS_ENV / "bin" / "python",
                      "tensorflowjs==4.22.0", "ydf==0.12.0", "setuptools<81"])

converter_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "3"}
run_command([tfjs_converter,
             "--input_format=tf_saved_model",
             "--output_format=tfjs_graph_model",
             "--signature_name=serving_default",
             "--saved_model_tags=serve",
             "--weight_shard_size_bytes=67108864",   # satu file shard: group1-shard1of1.bin
             SAVED_MODEL_DIR, TFJS_DIR], env=converter_env)

for f in sorted(TFJS_DIR.iterdir()):
    print(f"  {f.name:<26} {f.stat().st_size / 1e6:.2f} MB")
```

**Sel 29 — kode**

```python
# 4) requirements.txt berisi versi library yang benar-benar dipakai saat notebook dijalankan
packages = ["tensorflow", "keras", "numpy", "pandas", "matplotlib", "seaborn",
            "scikit-learn", "pillow", "kagglehub", "ai-edge-litert"]
lines = []
for package in packages:
    try:
        lines.append(f"{package}=={importlib_metadata.version(package)}")
    except importlib_metadata.PackageNotFoundError:
        pass
lines += ["", "# Konversi TFJS dijalankan di virtual environment terpisah (lihat bagian Konversi Model):",
          "# tensorflowjs==4.22.0 ydf==0.12.0 setuptools<81  (otomatis memakai tensorflow==2.19.0)"]
(SUBMISSION_DIR / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((SUBMISSION_DIR / "requirements.txt").read_text())

print("Isi folder submission:")
for path in sorted(SUBMISSION_DIR.rglob("*")):
    depth = len(path.relative_to(SUBMISSION_DIR).parts) - 1
    size = f"({path.stat().st_size / 1e6:.2f} MB)" if path.is_file() else ""
    print(f"{'    ' * depth}{path.name}{'/' if path.is_dir() else ''} {size}")

if IN_COLAB:   # unduh semua artefak model sekaligus untuk dibawa ke folder proyek lokal
    from google.colab import files
    archive = shutil.make_archive("hasil_colab_artifacts", "zip", root_dir=".", base_dir=SUBMISSION_DIR.name)
    files.download(archive)
```

**Sel 30 — markdown**

```markdown
## Inference (Optional)
```

**Sel 31 — kode**

```python
try:
    from ai_edge_litert.interpreter import Interpreter   # LiteRT, penerus tf.lite.Interpreter
except ImportError:
    Interpreter = tf.lite.Interpreter

interpreter = Interpreter(model_path=str(TFLITE_DIR / "model.tflite"))
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]
labels = (TFLITE_DIR / "label.txt").read_text(encoding="utf-8").splitlines()
print("Input :", input_detail["shape"], input_detail["dtype"].__name__)
print("Output:", output_detail["shape"], output_detail["dtype"].__name__)


def predict_tflite(batch: np.ndarray) -> np.ndarray:
    """Prediksi satu batch gambar (float32, 0-255, IMG_SIZE x IMG_SIZE) dengan model TF-Lite."""
    interpreter.resize_tensor_input(input_detail["index"], batch.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_detail["index"], batch.astype(np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(output_detail["index"])


tflite_pred = np.concatenate([predict_tflite(x.numpy()).argmax(axis=1) for x, _ in test_ds])
print(f"TFLITE_TEST_ACCURACY: {(tflite_pred == y_true).mean():.4f}")
print(f"Prediksi TF-Lite sama dengan model Keras: {(tflite_pred == y_pred).mean():.2%}")
```

**Sel 32 — kode**

```python
def predict_image_file(path: str):
    """Prediksi satu file gambar mentah (resolusi berapa pun) dengan model TF-Lite."""
    image = load_and_resize(tf.constant(path)).numpy().astype(np.float32)[None, ...]
    probs = predict_tflite(image)[0]
    return int(probs.argmax()), float(probs.max())


demo = test_df.sample(12, random_state=SEED)
fig, axes = plt.subplots(3, 4, figsize=(15, 11))
for ax, (_, row) in zip(axes.flat, demo.iterrows()):
    pred, conf = predict_image_file(row["path"])
    with Image.open(row["path"]) as img:
        ax.imshow(img)
        size = img.size
    ax.set_title(f"Asli: {labels[row['label']]}\nPrediksi: {labels[pred]} ({conf:.1%})\n{size[0]}x{size[1]} px",
                 color="green" if pred == row["label"] else "red", fontsize=8)
    ax.axis("off")
plt.suptitle("Inference TF-Lite pada gambar test (hijau = benar, merah = salah)")
plt.tight_layout()
plt.show()
```
