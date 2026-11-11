#!/usr/bin/env python3
"""Utilitas finalisasi submission Dicoding (hanya library standar Python 3.8+).

Subperintah (jalankan dari root proyek):
    python scripts/pack_submission.py import [--src hasil_colab]   # bawa hasil Colab ke submission/
    python scripts/pack_submission.py verify                       # sel kode/output masih identik dgn hasil Colab?
    python scripts/pack_submission.py py                           # buat ulang notebook.py dari notebook.ipynb
    python scripts/pack_submission.py zip [--force]                # cek final lalu buat submission.zip

`import` mencari di folder --src:
    - file .ipynb hasil "Download .ipynb" dari Colab  -> submission/notebook.ipynb
    - file .py hasil "Download .py" dari Colab        -> submission/notebook.py (dibuat otomatis bila tidak ada)
    - hasil_colab_artifacts.zip (atau folder saved_model/, tflite/, tfjs_model/ + requirements.txt)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

try:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

ARTIFACT_DIRS = ("saved_model", "tflite", "tfjs_model")
ARTIFACT_FILES = ("requirements.txt",)
EXCLUDE_NAMES = {".ipynb_checkpoints", "__pycache__", ".DS_Store", "Thumbs.db", ".backup", ".tfjs_env"}
EXCLUDE_SUFFIX = {".pyc", ".bak", ".keras", ".h5", ".zip"}


def newest(paths):
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def notebook_to_py(nb_path: Path, py_path: Path) -> None:
    """Konversi .ipynb -> .py (format sel '# %%'), magic IPython dijadikan komentar."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    out = ["# -*- coding: utf-8 -*-", f'"""{nb_path.name}', "", "File ini dibuat otomatis dari notebook.ipynb.", '"""', ""]
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            out.append("# %% [markdown]")
            out += [("# " + ln) if ln.strip() else "#" for ln in src.splitlines()]
        elif cell.get("cell_type") == "code":
            out.append("# %%")
            for ln in src.splitlines():
                stripped = ln.lstrip()
                out.append(ln[: len(ln) - len(stripped)] + "# " + stripped if stripped.startswith(("!", "%")) else ln)
        out.append("")
    py_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> list[str]:
    """Ekstrak hanya artefak model (saved_model/, tflite/, tfjs_model/, requirements.txt) dengan aman."""
    written = []
    for info in zf.infolist():
        parts = [p for p in PurePosixPath(info.filename).parts if p not in ("", ".")]
        if parts and parts[0] in ("submission", "content"):
            parts = parts[1:]
        if parts and parts[0] == "submission":
            parts = parts[1:]
        if not parts or ".." in parts or info.is_dir():
            continue
        if parts[0] not in ARTIFACT_DIRS and not (len(parts) == 1 and parts[0] in ARTIFACT_FILES):
            continue
        target = dest.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        written.append("/".join(parts))
    return written


def cmd_import(args) -> int:
    src, dest = Path(args.src), Path(args.dest)
    if not src.is_dir():
        print(f"[FAIL] Folder {src}/ tidak ditemukan. Taruh hasil unduhan Colab di sana.")
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    nb = newest(list(src.glob("*.ipynb")) or list(src.rglob("*.ipynb")))
    if nb is None:
        print(f"[FAIL] Tidak ada file .ipynb di {src}/ (Colab: File > Download > Download .ipynb)")
        return 1
    data = json.loads(nb.read_text(encoding="utf-8"))
    code_cells = [c for c in data.get("cells", []) if c.get("cell_type") == "code" and "".join(c.get("source", "")).strip()]
    executed = sum(1 for c in code_cells if c.get("execution_count") is not None)
    if executed < len(code_cells):
        print(f"[WARN] {nb.name}: baru {executed}/{len(code_cells)} sel kode yang dijalankan. Pastikan unduh SETELAH 'Run all' selesai.")
    target_nb = dest / "notebook.ipynb"
    if target_nb.exists():
        backup = Path(".backup")
        backup.mkdir(exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(target_nb, backup / f"notebook-{stamp}.ipynb")
        print(f"[INFO] notebook lama dicadangkan ke .backup/notebook-{stamp}.ipynb")
    shutil.copy2(nb, target_nb)
    print(f"[OK]   {nb} -> {target_nb}")

    py = newest([p for p in src.glob("*.py")])
    if py is not None:
        shutil.copy2(py, dest / "notebook.py")
        print(f"[OK]   {py} -> {dest / 'notebook.py'}")
    else:
        notebook_to_py(target_nb, dest / "notebook.py")
        print(f"[OK]   notebook.py dibuat otomatis dari notebook.ipynb (tidak ada .py di {src}/)")

    archive = newest(list(src.glob("*.zip")))
    if archive is not None:
        for d in ARTIFACT_DIRS:
            shutil.rmtree(dest / d, ignore_errors=True)
        with zipfile.ZipFile(archive) as zf:
            written = safe_extract(zf, dest)
        print(f"[OK]   {archive.name}: {len(written)} file artefak diekstrak ke {dest}/")
        if not written:
            print("[WARN] Zip tidak berisi saved_model/, tflite/, tfjs_model/ atau requirements.txt")
    else:
        copied = 0
        for d in ARTIFACT_DIRS:
            if (src / d).is_dir():
                shutil.rmtree(dest / d, ignore_errors=True)
                shutil.copytree(src / d, dest / d)
                copied += 1
        for f in ARTIFACT_FILES:
            if (src / f).is_file():
                shutil.copy2(src / f, dest / f)
                copied += 1
        level = "[OK]  " if copied else "[WARN]"
        print(f"{level} {copied} artefak disalin dari folder {src}/" + ("" if copied else " -> artefak model belum ada!"))
    print("\nLangkah berikut: python scripts/check_submission.py")
    return 0


def cmd_py(args) -> int:
    dest = Path(args.dest)
    nb = dest / "notebook.ipynb"
    if not nb.exists():
        print(f"[FAIL] {nb} tidak ada")
        return 1
    notebook_to_py(nb, dest / "notebook.py")
    print(f"[OK]   {dest / 'notebook.py'} dibuat ulang dari {nb}")
    return 0


def _code_signature(nb_path: Path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    sig, n_md = [], 0
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if cell.get("cell_type") == "code":
            outputs = json.dumps(cell.get("outputs", []), sort_keys=True)
            sig.append((src.strip(), outputs, cell.get("execution_count")))
        else:
            n_md += 1
    return sig, n_md


def cmd_verify(args) -> int:
    """Pastikan sel kode, output, dan nomor eksekusi identik dengan notebook asli dari Colab."""
    dest, src = Path(args.dest), Path(args.src)
    final_nb = dest / "notebook.ipynb"
    colab_nb = newest(list(src.glob("*.ipynb")) or list(src.rglob("*.ipynb")))
    if not final_nb.exists() or colab_nb is None:
        print(f"[FAIL] Butuh {final_nb} dan file .ipynb asli di {src}/")
        return 1
    a, md_a = _code_signature(colab_nb)
    b, md_b = _code_signature(final_nb)
    problems = []
    if len(a) != len(b):
        problems.append(f"jumlah sel kode berbeda: Colab={len(a)}, final={len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        for label, xa, yb in (("kode", x[0], y[0]), ("output", x[1], y[1]), ("execution_count", x[2], y[2])):
            if xa != yb:
                problems.append(f"sel kode ke-{i + 1}: {label} berubah")
    if problems:
        print("[FAIL] Notebook final tidak identik dengan hasil Colab pada sel kode/output:")
        for p in problems[:15]:
            print("   -", p)
        print("Kembalikan perubahan (salinan asli ada di hasil_colab/ dan .backup/) atau jalankan ulang di Colab.")
        return 1
    print(f"[OK]   {len(b)} sel kode + output identik dengan {colab_nb.name}. Sel markdown: {md_a} -> {md_b}.")
    return 0


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDE_NAMES for part in path.parts) or path.suffix.lower() in EXCLUDE_SUFFIX


def cmd_zip(args) -> int:
    dest, out = Path(args.dest), Path(args.out)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_submission  # noqa: E402

    if not args.force:
        print("Menjalankan pemeriksaan final sebelum membuat zip...\n")
        if check_submission.main(["--dir", str(dest)]) != 0:
            print("\n[FAIL] Zip TIDAK dibuat karena masih ada FAIL. Perbaiki dulu (atau --force untuk memaksa).")
            return 1
    files = sorted(p for p in dest.rglob("*") if p.is_file() and not is_excluded(p.relative_to(dest)))
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            zf.write(p, arcname=(PurePosixPath("submission") / p.relative_to(dest).as_posix()).as_posix())
    print(f"\n[OK]   {out} dibuat: {len(files)} file, {out.stat().st_size / 1e6:.2f} MB")
    print("Validasi isi zip:\n")
    return check_submission.main(["--dir", str(dest), "--zip", str(out)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Finalisasi submission Dicoding")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_imp = sub.add_parser("import", help="bawa hasil Colab ke folder submission/")
    p_imp.add_argument("--src", default="hasil_colab")
    p_imp.add_argument("--dest", default="submission")
    p_py = sub.add_parser("py", help="buat ulang notebook.py dari notebook.ipynb")
    p_py.add_argument("--dest", default="submission")
    p_ver = sub.add_parser("verify", help="pastikan sel kode/output sama dengan notebook asli Colab")
    p_ver.add_argument("--src", default="hasil_colab")
    p_ver.add_argument("--dest", default="submission")
    p_zip = sub.add_parser("zip", help="cek final lalu buat submission.zip")
    p_zip.add_argument("--dest", default="submission")
    p_zip.add_argument("--out", default="submission.zip")
    p_zip.add_argument("--force", action="store_true", help="buat zip walau masih ada FAIL")
    args = parser.parse_args(argv)
    return {"import": cmd_import, "py": cmd_py, "verify": cmd_verify, "zip": cmd_zip}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
