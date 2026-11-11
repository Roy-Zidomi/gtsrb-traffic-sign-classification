#!/usr/bin/env python3
"""Pemeriksa otomatis submission Dicoding "Proyek Klasifikasi Gambar" (Belajar Fundamental Deep Learning).

Pemakaian (jalankan dari root proyek):
    python scripts/check_submission.py --stage pre          # sebelum notebook dijalankan di Colab
    python scripts/check_submission.py                      # final: folder submission/ lengkap
    python scripts/check_submission.py --zip submission.zip # final + validasi isi zip
    python scripts/check_submission.py --json               # tambahan ringkasan JSON di akhir

Hanya memakai library standar Python 3.8+. Exit code: 0 = tanpa FAIL, 1 = ada FAIL, 2 = salah pemakaian.

Notebook diharapkan mencetak baris penanda berikut (dibaca dari output sel):
    TOTAL_IMAGES, NUM_CLASSES, UNIQUE_RESOLUTIONS, SPLIT_COUNTS,
    TRAIN_ACCURACY, VAL_ACCURACY, TEST_ACCURACY, TFLITE_TEST_ACCURACY
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

try:  # konsol Windows kadang bukan UTF-8
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

# --------------------------------------------------------------------------- konfigurasi
REQUIRED_HEADINGS = [
    ("judul", r"^proyek klasifikasi gambar"),
    ("import", r"^import semua packages/library yang digunakan"),
    ("data_preparation", r"^data preparation"),
    ("data_loading", r"^data loading"),
    ("data_preprocessing", r"^data preprocessing"),
    ("split_dataset", r"^split dataset"),
    ("modelling", r"^modell?ing"),
    ("evaluasi", r"^evaluasi dan visualisasi"),
    ("konversi", r"^konversi model"),
    ("inference", r"^inference"),
]
SECTIONS_NEED_CODE = ["import", "data_loading", "split_dataset", "modelling", "evaluasi", "konversi", "inference"]
MARKERS = ["TOTAL_IMAGES", "NUM_CLASSES", "UNIQUE_RESOLUTIONS", "SPLIT_COUNTS",
           "TRAIN_ACCURACY", "VAL_ACCURACY", "TEST_ACCURACY", "TFLITE_TEST_ACCURACY"]
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
JUNK_NAMES = {".ipynb_checkpoints", "__pycache__", ".DS_Store", "Thumbs.db", ".backup", ".tfjs_env"}
JUNK_SUFFIX = {".keras", ".h5", ".zip", ".pyc", ".bak"}
ALLOWED_TOP = {"saved_model", "tflite", "tfjs_model", "notebook.ipynb", "notebook.py", "README.md", "requirements.txt"}
FORBIDDEN_DATASETS = re.compile(r"rock[-_ ]?paper[-_ ]?scissors|rockpaperscissors|chest[-_ ]?x[-_ ]?ray|x[-_]?ray", re.I)
PLACEHOLDER = re.compile(r"\[input[^\]]*\]|<<\s*isi|dari claude\.md|\{\{", re.I)
MIN_ACC, STAR_ACC = 0.85, 0.95


# --------------------------------------------------------------------------- pencatat hasil
class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.metrics: dict = {}
        self.suggestions: dict[str, bool] = {}

    def add(self, level: str, group: str, msg: str) -> None:
        self.items.append({"level": level, "group": group, "msg": msg})

    def ok(self, group: str, msg: str) -> None:
        self.add("PASS", group, msg)

    def fail(self, group: str, msg: str) -> None:
        self.add("FAIL", group, msg)

    def warn(self, group: str, msg: str) -> None:
        self.add("WARN", group, msg)

    def info(self, group: str, msg: str) -> None:
        self.add("INFO", group, msg)

    def check(self, cond: bool, group: str, ok_msg: str, bad_msg: str, level: str = "FAIL") -> bool:
        self.add("PASS" if cond else level, group, ok_msg if cond else bad_msg)
        return cond

    def count(self, level: str) -> int:
        return sum(1 for i in self.items if i["level"] == level)


# --------------------------------------------------------------------------- utilitas notebook
def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def normalize_heading(line: str) -> str:
    text = re.sub(r"^#+\s*", "", line.strip())
    text = re.sub(r"[*_`]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def output_texts(cell: dict) -> list[str]:
    texts = []
    for out in cell.get("outputs", []) or []:
        kind = out.get("output_type")
        if kind == "stream":
            t = out.get("text", "")
            texts.append("".join(t) if isinstance(t, list) else str(t))
        elif kind in ("execute_result", "display_data"):
            t = out.get("data", {}).get("text/plain", "")
            texts.append("".join(t) if isinstance(t, list) else str(t))
        elif kind == "error":
            texts.append("\n".join(out.get("traceback", [])))
    return texts


def has_image(cell: dict) -> bool:
    return any("image/png" in (o.get("data") or {}) or "image/jpeg" in (o.get("data") or {})
               for o in cell.get("outputs", []) or [])


def strip_magics(src: str) -> str:
    """Ganti baris IPython magic (!cmd, %magic) dengan `pass` agar bisa di-compile."""
    if src.lstrip().startswith("%%"):
        return ""
    lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            lines.append(line[: len(line) - len(stripped)] + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def last_float(pattern: str, text: str):
    hits = re.findall(pattern, text)
    if not hits:
        return None
    value = float(hits[-1].replace(",", "."))
    return value / 100 if value > 1.0 else value


def last_int(pattern: str, text: str):
    hits = re.findall(pattern, text)
    return int(re.sub(r"[^\d]", "", hits[-1])) if hits else None


def assign_sections(cells: list[dict]):
    """Petakan setiap sel ke bagian template terakhir yang ditemui di atasnya."""
    current, mapping, found = None, [], []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            for line in cell_source(cell).splitlines():
                if not line.lstrip().startswith("#"):
                    continue
                norm = normalize_heading(line)
                for key, pattern in REQUIRED_HEADINGS:
                    if re.match(pattern, norm):
                        current = key
                        found.append((key, idx, norm))
                        break
        mapping.append(current)
    return mapping, found


# --------------------------------------------------------------------------- pemeriksaan notebook
def check_notebook(nb_path: Path, rep: Report, stage: str) -> None:
    G = "Notebook"
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        rep.fail(G, f"{nb_path.name} tidak bisa dibaca sebagai JSON notebook: {exc}")
        return
    cells = nb.get("cells", [])
    rep.check(nb.get("nbformat", 0) >= 4 and bool(cells), G, f"{nb_path.name} valid (nbformat {nb.get('nbformat')}, {len(cells)} sel)",
              f"{nb_path.name} bukan notebook nbformat 4 atau tidak punya sel")

    sections, found = assign_sections(cells)
    # 1) heading template berurutan
    order_ok, pos = True, -1
    missing = []
    for key, _ in REQUIRED_HEADINGS:
        idxs = [i for k, i, _ in found if k == key and i > pos]
        if not idxs:
            missing.append(key)
            order_ok = False
            continue
        pos = idxs[0]
    rep.check(order_ok, G, "Semua heading template Dicoding ada dan berurutan",
              f"Heading template hilang/tidak berurutan: {', '.join(missing) or 'urutan salah'} (lihat docs/02_template_notebook.md)")

    # 2) identitas & judul
    header = next((cell_source(c) for c in cells if c.get("cell_type") == "markdown"), "")
    fields = {k: re.search(rf"\*\*{k}:?\*\*:?\s*(.+)", header) for k in ("Nama", "Email", "ID Dicoding")}
    empty = [k for k, m in fields.items() if not m or not m.group(1).strip() or PLACEHOLDER.search(m.group(1))]
    title_ok = re.search(r"proyek klasifikasi gambar\s*:\s*(\S.+)", header, re.I)
    ok_header = not empty and title_ok and not PLACEHOLDER.search(header)
    rep.check(bool(ok_header), G, "Judul dataset & identitas (Nama, Email, ID Dicoding) sudah diisi",
              f"Header notebook masih placeholder/kosong: {', '.join(empty) or 'judul dataset'}")

    # 3) setiap bagian punya sel kode
    code_by_section: dict[str, list[int]] = {}
    for idx, (cell, sec) in enumerate(zip(cells, sections)):
        if cell.get("cell_type") == "code" and cell_source(cell).strip():
            code_by_section.setdefault(sec, []).append(idx)
    if "split_dataset" not in code_by_section and "data_preprocessing" in code_by_section:
        code_by_section["split_dataset"] = code_by_section["data_preprocessing"]
    empty_sections = [s for s in SECTIONS_NEED_CODE if not code_by_section.get(s)]
    rep.check(not empty_sections, G, "Setiap bagian template berisi sel kode",
              f"Bagian tanpa sel kode: {', '.join(empty_sections)}")

    # 4) kode bisa di-compile
    code_cells = [(i, c) for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    syntax_errors = []
    for i, c in code_cells:
        try:
            compile(strip_magics(cell_source(c)), f"sel_{i}", "exec")
        except SyntaxError as exc:
            syntax_errors.append(f"sel #{i} baris {exc.lineno}: {exc.msg}")
    rep.check(not syntax_errors, G, "Semua sel kode lolos pemeriksaan sintaks",
              "Syntax error: " + "; ".join(syntax_errors[:5]))

    code = "\n".join(cell_source(c) for _, c in code_cells)
    outputs_all = "\n".join(t for _, c in code_cells for t in output_texts(c))

    # 5) eksekusi (hanya final)
    if stage == "final":
        nonempty = [(i, c) for i, c in code_cells if cell_source(c).strip()]
        not_run = [i for i, c in nonempty if c.get("execution_count") is None]
        rep.check(not not_run and bool(nonempty), G, "Semua sel kode sudah dijalankan (punya execution_count)",
                  f"Sel kode belum dijalankan: {not_run[:10]} -> jalankan ulang 'Run all' di Colab")
        counts = [c.get("execution_count") for _, c in nonempty if c.get("execution_count") is not None]
        rep.check(counts == sorted(counts), G, "Urutan eksekusi sel berurutan (bersih dari atas ke bawah)",
                  "Urutan eksekusi tidak berurutan -> sebaiknya Restart & Run all sekali lagi", level="WARN")
        errors = [i for i, c in code_cells for o in c.get("outputs", []) or [] if o.get("output_type") == "error"]
        rep.check(not errors, G, "Tidak ada output error", f"Ada output error di sel: {errors[:10]}")
        tb = [i for i, c in code_cells if any("Traceback (most recent call last)" in t for t in output_texts(c))]
        rep.check(not tb, G, "Tidak ada traceback di output", f"Ada traceback di output sel: {tb[:10]}", level="WARN")
        with_output = sum(1 for _, c in nonempty if c.get("outputs"))
        rep.check(with_output >= max(1, len(nonempty) // 2), G, f"{with_output}/{len(nonempty)} sel kode memiliki output",
                  f"Hanya {with_output}/{len(nonempty)} sel kode punya output -> notebook belum dijalankan penuh")
    else:
        if any(c.get("outputs") for _, c in code_cells):
            rep.info(G, "Notebook sudah berisi output (tidak wajib pada tahap pre)")
        accel = nb.get("metadata", {}).get("accelerator")
        rep.check(accel == "GPU", G, "Metadata Colab meminta GPU (accelerator=GPU)",
                  "Metadata accelerator bukan GPU -> set Runtime > Change runtime type > T4 GPU di Colab", level="WARN")

    # ------------------------------------------------------------ kriteria wajib (kode)
    K = "Kriteria wajib"
    rep.check(not FORBIDDEN_DATASETS.search(code), K, "Bukan dataset terlarang (Rock-Paper-Scissors / X-Ray)",
              "Terdeteksi dataset terlarang (Rock-Paper-Scissors / X-Ray) di kode")
    rep.check(bool(re.search(r"\bSequential\s*\(", code)), K, "Model memakai Sequential", "Tidak ditemukan `Sequential(` di kode")
    rep.check(bool(re.search(r"\bConv2D\s*\(", code)), K, "Model memakai Conv2D", "Tidak ditemukan `Conv2D(` di kode")
    spatial_pool = re.search(r"\b(MaxPooling2D|MaxPool2D|AveragePooling2D|AvgPool2D)\s*\(", code)
    global_pool = re.search(r"\bGlobal(Average|Max)Pooling2D\s*\(", code)
    if spatial_pool:
        rep.ok(K, f"Model memakai pooling layer ({spatial_pool.group(1)})")
    elif global_pool:
        rep.warn(K, "Hanya ada Global pooling -> tambahkan MaxPooling2D agar kriteria pooling layer tidak diragukan")
    else:
        rep.fail(K, "Tidak ditemukan pooling layer (MaxPooling2D/AveragePooling2D)")
    rep.check(bool(re.search(r"\.export\s*\(|saved_model\.save\s*\(", code)), K, "Kode ekspor SavedModel ada",
              "Tidak ditemukan ekspor SavedModel (model.export(...) / tf.saved_model.save(...))")
    rep.check("TFLiteConverter" in code, K, "Kode konversi TF-Lite ada", "Tidak ditemukan TFLiteConverter")
    rep.check("tensorflowjs_converter" in code or "tensorflowjs" in code, K, "Kode konversi TFJS ada",
              "Tidak ditemukan konversi TFJS (tensorflowjs_converter)")
    rep.check("label.txt" in code, K, "Kode menulis tflite/label.txt", "Tidak ditemukan penulisan label.txt")
    rep.check(bool(re.search(r"\bfit\s*\(", code)) and bool(re.search(r"validation_data\s*=", code)), K,
              "Training memakai validation_data", "fit(...) tanpa validation_data -> plot val tidak tersedia")

    missing_markers = [m for m in MARKERS if m not in code]
    rep.check(not missing_markers, K, "Semua baris penanda (marker) dicetak oleh kode",
              f"Marker belum dicetak di kode: {', '.join(missing_markers)} (wajib untuk verifikasi)")

    bad_tfjs = [ln.strip() for ln in code.splitlines()
                if re.search(r"pip[\"'\s,\]]+install", ln) and "tensorflowjs" in ln
                and not re.search(r"--python|tfjs_env|TFJS_ENV|venv", ln)]
    rep.check(not bad_tfjs, "Teknis", "tensorflowjs tidak di-install di runtime utama",
              "tensorflowjs di-install di runtime utama (bisa menurunkan versi TensorFlow) -> pakai venv terpisah (docs/03)",
              level="WARN")
    if re.search(r"RandomFlip\s*\(", code):
        rep.warn("Teknis", "Ada RandomFlip: untuk rambu lalu lintas flip dapat mengubah makna kelas (kiri<->kanan)")
    if "files.download" not in code:
        rep.warn("Teknis", "Tidak ada files.download(...) -> user harus mengunduh artefak Colab secara manual")

    # ------------------------------------------------------------ saran (kode)
    S = "Saran bintang 5"
    has_cb = bool(re.search(r"callbacks\s*=", code)) and bool(
        re.search(r"EarlyStopping|ModelCheckpoint|ReduceLROnPlateau|\(\s*(keras\.|tf\.keras\.)?callbacks\.Callback\s*\)|\(\s*Callback\s*\)", code))
    rep.suggestions["callback"] = has_cb
    rep.check(has_cb, S, "Callback diterapkan pada training", "Belum ada callback di model.fit(callbacks=...)", level="WARN")
    inf_code = "\n".join(cell_source(c) for (i, c), sec in zip(enumerate(cells), sections)
                         if sec == "inference" and c.get("cell_type") == "code")
    has_inf_code = bool(re.search(r"Interpreter\s*\(|saved_model\.load\s*\(|TFSMLayer|tensorflowjs|tfjs", inf_code))
    if stage == "pre":
        rep.suggestions["inference"] = has_inf_code
        rep.check(has_inf_code, S, "Bagian Inference memuat model hasil konversi (TF-Lite/SavedModel/TFJS)",
                  "Bagian Inference belum memakai model hasil konversi", level="WARN")
        return

    # ------------------------------------------------------------ angka dari output (final)
    m = rep.metrics
    m["total_images"] = last_int(r"TOTAL_IMAGES\s*[:=]\s*([\d.,_]+)", outputs_all)
    m["num_classes"] = last_int(r"NUM_CLASSES\s*[:=]\s*(\d+)", outputs_all)
    m["unique_resolutions"] = last_int(r"UNIQUE_RESOLUTIONS\s*[:=]\s*([\d.,_]+)", outputs_all)
    split = re.findall(r"SPLIT_COUNTS\s*[:=]\s*train\s*=\s*(\d+)\s*,\s*val\s*=\s*(\d+)\s*,\s*test\s*=\s*(\d+)", outputs_all)
    m["split"] = dict(zip(("train", "val", "test"), map(int, split[-1]))) if split else None
    m["train_accuracy"] = last_float(r"TRAIN_ACCURACY\s*[:=]\s*([0-9.,]+)", outputs_all)
    m["val_accuracy"] = last_float(r"VAL_ACCURACY\s*[:=]\s*([0-9.,]+)", outputs_all)
    m["test_accuracy"] = last_float(r"TEST_ACCURACY\s*[:=]\s*([0-9.,]+)", outputs_all)
    m["tflite_test_accuracy"] = last_float(r"TFLITE_TEST_ACCURACY\s*[:=]\s*([0-9.,]+)", outputs_all)
    m["last_epoch_accuracy"] = last_float(r"(?<![\w])accuracy: ([0-9.]+)", outputs_all)
    m["last_epoch_val_accuracy"] = last_float(r"val_accuracy: ([0-9.]+)", outputs_all)

    def need(key: str, label: str) -> bool:
        if m.get(key) is None:
            rep.fail(K, f"Penanda {label} tidak ditemukan di output notebook")
            return False
        return True

    if need("total_images", "TOTAL_IMAGES"):
        rep.check(m["total_images"] >= 1000, K, f"Jumlah gambar {m['total_images']:,} (>= 1.000)",
                  f"Jumlah gambar hanya {m['total_images']:,} (< 1.000)")
    if need("split", "SPLIT_COUNTS"):
        sp = m["split"]
        tot = sum(sp.values()) or 1
        rep.check(all(v > 0 for v in sp.values()), K,
                  "Split train/val/test = {train}/{val}/{test} ({p1:.1f}% / {p2:.1f}% / {p3:.1f}%)".format(
                      **sp, p1=sp["train"] / tot * 100, p2=sp["val"] / tot * 100, p3=sp["test"] / tot * 100),
                  f"Split tidak lengkap: {sp}")
    acc_ok = need("train_accuracy", "TRAIN_ACCURACY") & need("test_accuracy", "TEST_ACCURACY")
    if acc_ok:
        tr, te = m["train_accuracy"], m["test_accuracy"]
        rep.check(tr >= MIN_ACC and te >= MIN_ACC, K, f"Akurasi train {tr:.2%} & test {te:.2%} (>= 85%)",
                  f"Akurasi di bawah 85%: train {tr:.2%}, test {te:.2%} -> submission akan DITOLAK")
    if m["last_epoch_accuracy"] is not None:
        lvl = "INFO" if m["last_epoch_accuracy"] >= STAR_ACC else "WARN"
        rep.add(lvl, K, f"Akurasi log epoch terakhir: accuracy={m['last_epoch_accuracy']:.4f}"
                        + (f", val_accuracy={m['last_epoch_val_accuracy']:.4f}" if m["last_epoch_val_accuracy"] is not None else ""))

    eval_imgs = sum(1 for (i, c), sec in zip(enumerate(cells), sections) if sec == "evaluasi" and has_image(c))
    plot_code = re.search(r"history", code) and re.search(r"['\"]val_(accuracy|loss)['\"]|val_\{", code)
    rep.check(eval_imgs > 0 and bool(plot_code), K, f"Plot akurasi & loss ada ({eval_imgs} gambar di bagian Evaluasi)",
              "Plot akurasi/loss tidak ditemukan di output bagian Evaluasi dan Visualisasi")

    # ------------------------------------------------------------ saran (output)
    rep.suggestions["resolusi_tidak_seragam"] = bool(m["unique_resolutions"] and m["unique_resolutions"] > 1)
    rep.check(rep.suggestions["resolusi_tidak_seragam"], S,
              f"Resolusi asli tidak seragam ({m['unique_resolutions']} resolusi unik)",
              "Penanda UNIQUE_RESOLUTIONS tidak ada atau resolusi seragam", level="WARN")
    rep.suggestions["min_10000_gambar"] = bool(m["total_images"] and m["total_images"] >= 10000)
    rep.check(rep.suggestions["min_10000_gambar"], S, "Dataset >= 10.000 gambar", "Dataset < 10.000 gambar", level="WARN")
    tr, te = m.get("train_accuracy"), m.get("test_accuracy")
    rep.suggestions["akurasi_95"] = bool(tr is not None and te is not None and tr >= STAR_ACC and te >= STAR_ACC)
    rep.check(rep.suggestions["akurasi_95"], S, "Akurasi train & test >= 95%",
              f"Akurasi belum >= 95% (train={tr}, test={te})", level="WARN")
    rep.suggestions["min_3_kelas"] = bool(m["num_classes"] and m["num_classes"] >= 3)
    rep.check(rep.suggestions["min_3_kelas"], S, f"Jumlah kelas {m['num_classes']} (>= 3)",
              "Penanda NUM_CLASSES tidak ada atau kelas < 3", level="WARN")
    inf_out = any((sec == "inference") and (c.get("outputs")) for (i, c), sec in zip(enumerate(cells), sections))
    rep.suggestions["inference"] = has_inf_code and inf_out and m["tflite_test_accuracy"] is not None
    rep.check(rep.suggestions["inference"], S,
              f"Inference model hasil konversi dengan bukti output (TF-Lite test acc {m['tflite_test_accuracy']})",
              "Bukti inference (output + TFLITE_TEST_ACCURACY) belum ada di bagian Inference", level="WARN")


# --------------------------------------------------------------------------- pemeriksaan folder
def check_folder(sub: Path, rep: Report) -> None:
    G = "Struktur"
    if not rep.check(sub.is_dir(), G, f"Folder {sub.name}/ ada", f"Folder {sub} tidak ditemukan"):
        return
    nb = sub / "notebook.ipynb"
    if not nb.exists():
        others = sorted(p.name for p in sub.glob("*.ipynb"))
        rep.fail(G, "notebook.ipynb tidak ada" + (f" (ditemukan: {', '.join(others)} -> ganti nama)" if others else ""))
    else:
        rep.ok(G, "notebook.ipynb ada")
    py = sub / "notebook.py"
    rep.check(py.exists() and py.stat().st_size > 100, G, "notebook.py ada",
              "notebook.py tidak ada/kosong -> Dicoding menolak jika .py & .ipynb tidak dilampirkan (pakai pack_submission.py py)")
    readme = sub / "README.md"
    if rep.check(readme.exists() and readme.stat().st_size > 300, G, "README.md ada", "README.md tidak ada atau terlalu pendek"):
        text = readme.read_text(encoding="utf-8", errors="replace")
        rep.check(not re.search(r"<<|TODO|\[isi|XX\.X|\{\{", text, re.I), G, "README.md tanpa placeholder",
                  "README.md masih berisi placeholder ({{...}}, <<, TODO, [isi, XX.X) -> isi dengan angka nyata")
    req = sub / "requirements.txt"
    if rep.check(req.exists() and req.stat().st_size > 0, G, "requirements.txt ada", "requirements.txt tidak ada/kosong"):
        lines = [ln.strip() for ln in req.read_text(encoding="utf-8", errors="replace").splitlines()
                 if ln.strip() and not ln.strip().startswith("#")]
        rep.check(any(re.match(r"tensorflow\b", ln, re.I) for ln in lines), G,
                  f"requirements.txt memuat tensorflow ({len(lines)} paket)", "requirements.txt tidak memuat tensorflow", level="WARN")

    sm = sub / "saved_model"
    var = sm / "variables"
    sm_ok = (sm / "saved_model.pb").exists() and (var / "variables.index").exists() and any(var.glob("variables.data-*"))
    rep.check(sm_ok, G, "SavedModel lengkap (saved_model.pb + variables/)", "SavedModel tidak lengkap di submission/saved_model/")

    tfl = sub / "tflite" / "model.tflite"
    if rep.check(tfl.exists(), G, "tflite/model.tflite ada", "tflite/model.tflite tidak ada"):
        head = tfl.read_bytes()[:8]
        rep.check(head[4:8] == b"TFL3" and tfl.stat().st_size > 10_000, G,
                  f"model.tflite valid ({tfl.stat().st_size / 1e6:.2f} MB)", "model.tflite rusak/bukan flatbuffer TFLite")
    lab = sub / "tflite" / "label.txt"
    if rep.check(lab.exists(), G, "tflite/label.txt ada", "tflite/label.txt tidak ada"):
        labels = [ln.strip() for ln in lab.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        n = rep.metrics.get("num_classes")
        cond = len(labels) >= 3 and len(set(labels)) == len(labels) and (n is None or n == len(labels))
        rep.check(cond, G, f"label.txt berisi {len(labels)} label unik" + (" (= NUM_CLASSES)" if n else ""),
                  f"label.txt bermasalah: {len(labels)} baris, unik={len(set(labels))}, NUM_CLASSES={n}")

    tj = sub / "tfjs_model"
    mj = tj / "model.json"
    if rep.check(mj.exists(), G, "tfjs_model/model.json ada", "tfjs_model/model.json tidak ada"):
        try:
            meta = json.loads(mj.read_text(encoding="utf-8"))
            paths = [p for grp in meta.get("weightsManifest", []) for p in grp.get("paths", [])]
            missing = [p for p in paths if not (tj / p).exists()]
            rep.check(bool(paths) and not missing, G,
                      f"TFJS {meta.get('format', '?')}: {len(paths)} shard ({', '.join(paths[:3])})",
                      f"Shard TFJS hilang: {missing or 'weightsManifest kosong'}")
        except Exception as exc:  # noqa: BLE001
            rep.fail(G, f"model.json tidak valid: {exc}")

    files = [p for p in sub.rglob("*") if p.is_file()]
    n_img = sum(1 for p in files if p.suffix.lower() in IMAGE_EXT)
    rep.check(n_img <= 30, G, f"Tidak ada dataset di folder submission ({n_img} file gambar)",
              f"{n_img} file gambar di folder submission -> dataset tidak boleh ikut")
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    rep.check(size_mb <= 100, G, f"Ukuran folder submission {size_mb:.1f} MB", f"Ukuran folder {size_mb:.1f} MB (> 100 MB), perkecil",
              level="WARN")
    junk = sorted({p.name for p in sub.rglob("*") if p.name in JUNK_NAMES or p.suffix.lower() in JUNK_SUFFIX})
    if junk:
        rep.warn(G, f"File/folder sampah akan dikecualikan saat zip: {', '.join(junk[:8])}")
    extra = sorted(p.name for p in sub.iterdir() if p.name not in ALLOWED_TOP and p.name not in JUNK_NAMES)
    if extra:
        rep.info(G, f"Item tambahan di submission/: {', '.join(extra)}")


def check_zip(zip_path: Path, rep: Report) -> None:
    G = "Zip"
    if not rep.check(zip_path.exists(), G, f"{zip_path.name} ada", f"{zip_path} tidak ditemukan"):
        return
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            names = [n for n in zf.namelist() if not n.endswith("/")]
    except zipfile.BadZipFile as exc:
        rep.fail(G, f"Zip rusak: {exc}")
        return
    rep.check(bad is None, G, "Integritas zip OK", f"File rusak di zip: {bad}")
    roots = {n.split("/")[0] for n in names}
    rep.check(roots == {"submission"}, G, "Zip berisi satu folder 'submission/'", f"Root zip harus 'submission/', ditemukan: {sorted(roots)}")
    required = ["submission/notebook.ipynb", "submission/notebook.py", "submission/README.md", "submission/requirements.txt",
                "submission/saved_model/saved_model.pb", "submission/saved_model/variables/variables.index",
                "submission/tflite/model.tflite", "submission/tflite/label.txt", "submission/tfjs_model/model.json"]
    missing = [r for r in required if r not in names]
    has_bin = any(re.match(r"submission/tfjs_model/group\d+-shard\d+of\d+\.bin$", n) for n in names)
    rep.check(not missing and has_bin, G, f"Zip memuat semua file wajib ({len(names)} file)",
              f"File wajib tidak ada di zip: {missing + ([] if has_bin else ['tfjs_model/group*-shard*.bin'])}")
    n_img = sum(1 for n in names if Path(n).suffix.lower() in IMAGE_EXT)
    rep.check(n_img <= 30, G, "Zip tanpa dataset", f"Zip berisi {n_img} file gambar -> dataset tidak boleh ikut")
    junk = [n for n in names if any(part in JUNK_NAMES for part in n.split("/")) or Path(n).suffix.lower() in JUNK_SUFFIX]
    rep.check(not junk, G, "Zip bersih dari file sampah", f"Zip berisi file sampah: {junk[:5]}", level="WARN")
    size_mb = zip_path.stat().st_size / 1e6
    rep.check(size_mb <= 100, G, f"Ukuran zip {size_mb:.1f} MB", f"Ukuran zip {size_mb:.1f} MB (> 100 MB)", level="WARN")


# --------------------------------------------------------------------------- ringkasan
def summarize(rep: Report, stage: str) -> str:
    groups: dict[str, list[dict]] = {}
    for item in rep.items:
        groups.setdefault(item["group"], []).append(item)
    lines = []
    for group, items in groups.items():
        lines.append(f"\n== {group} ==")
        lines += [f"[{i['level']}] {i['msg']}" for i in items]
    n_fail, n_warn, n_pass = rep.count("FAIL"), rep.count("WARN"), rep.count("PASS")
    lines.append("\n== Ringkasan ==")
    lines.append(f"FAIL: {n_fail} | WARN: {n_warn} | PASS: {n_pass}")
    if stage == "final":
        met = sum(rep.suggestions.values())
        total = 6
        if n_fail:
            verdict = "BELUM SIAP KIRIM - perbaiki semua FAIL terlebih dahulu"
        elif met == total:
            verdict = "Siap kirim. Semua saran terpenuhi -> berpotensi bintang 5 (kualitas kode tetap dinilai reviewer)"
        elif met >= 3:
            verdict = f"Siap kirim. {met}/{total} saran -> berpotensi bintang 4"
        else:
            verdict = f"Siap kirim. {met}/{total} saran -> berpotensi bintang 3"
        lines.append(f"Saran terpenuhi: {met}/{total} {sorted(k for k, v in rep.suggestions.items() if v)}")
        lines.append(f"Status: {verdict}")
    else:
        lines.append("Status: " + ("SIAP dijalankan di Colab" if not n_fail else "BELUM SIAP - perbaiki FAIL sebelum ke Colab"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pemeriksa submission Dicoding klasifikasi gambar")
    parser.add_argument("--dir", default="submission", help="folder submission (default: submission)")
    parser.add_argument("--stage", choices=["pre", "final"], default="final")
    parser.add_argument("--zip", help="path zip yang akan divalidasi (opsional)")
    parser.add_argument("--json", action="store_true", help="cetak ringkasan JSON di akhir")
    args = parser.parse_args(argv)

    sub = Path(args.dir)
    rep = Report()
    nb_path = sub / "notebook.ipynb"
    if nb_path.exists():
        check_notebook(nb_path, rep, args.stage)
    else:
        rep.fail("Notebook", f"{nb_path} tidak ditemukan")
    if args.stage == "final":
        check_folder(sub, rep)
    if args.zip:
        check_zip(Path(args.zip), rep)

    print(f"Pemeriksaan submission (tahap: {args.stage}) -> {sub.resolve()}")
    print(summarize(rep, args.stage))
    if args.json:
        print("\nJSON_SUMMARY: " + json.dumps({
            "stage": args.stage, "fail": rep.count("FAIL"), "warn": rep.count("WARN"), "pass": rep.count("PASS"),
            "suggestions": rep.suggestions, "metrics": rep.metrics,
            "failures": [i["msg"] for i in rep.items if i["level"] == "FAIL"],
        }, ensure_ascii=True))
    return 1 if rep.count("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
