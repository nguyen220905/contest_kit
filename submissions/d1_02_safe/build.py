"""Build submissions/d1_02_safe.zip (Windows-friendly replacement of 02_rag/build.sh).

Usage: .venv\Scripts\python submissions/d1_02_safe/build.py [--force]
Downloads Docling layout model + MiniLM next to main.py (skipped if present),
then zips main.py (at zip root) + docling_models/ + embed_model/.
"""
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZIP_PATH = HERE.parent / f"{HERE.name}.zip"
LIMIT = 1_000_000_000
FORCE = "--force" in sys.argv


def download_docling():
    out = HERE / "docling_models"
    if out.exists() and any(out.iterdir()) and not FORCE:
        print(f"[build] skip docling_models (exists)")
        return
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    exe = Path(sys.executable).parent / "docling-tools.exe"
    if not exe.exists():
        exe = Path(sys.executable).parent / "docling-tools"
    if exe.exists():
        subprocess.run([str(exe), "models", "download", "layout", "-o", str(out)], check=True, env=env)
    else:
        os.environ.update(env)
        from docling.utils.model_downloader import download_models
        download_models(output_dir=out, with_layout=True, with_tableformer=False, with_code_formula=False,
                        with_picture_classifier=False, with_easyocr=False, progress=True)


def download_embed():
    out = HERE / "embed_model"
    if out.exists() and any(out.iterdir()) and not FORCE:
        print("[build] skip embed_model (exists)")
        return
    shutil.rmtree(out, ignore_errors=True)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").save(str(out))


def collect():
    files = [(HERE / "main.py", "main.py")]
    for d in ("docling_models", "embed_model"):
        for p in sorted((HERE / d).rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            files.append((p, p.relative_to(HERE).as_posix()))
    return files


def main():
    download_docling()
    download_embed()
    files = collect()
    ZIP_PATH.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files:
            z.write(p, arc)

    with zipfile.ZipFile(ZIP_PATH) as z:
        infos = z.infolist()
    names = [i.filename for i in infos]
    total = sum(i.file_size for i in infos)
    print("[build] 15 largest files (uncompressed bytes):")
    for i in sorted(infos, key=lambda i: -i.file_size)[:15]:
        print(f"  {i.file_size:>12,}  {i.filename}")
    print(f"[build] files={len(infos)} uncompressed={total:,} bytes, zip={ZIP_PATH.stat().st_size:,} bytes -> {ZIP_PATH}")

    errors = []
    if "main.py" not in names:
        errors.append("main.py missing at zip root")
    if total >= LIMIT:
        errors.append(f"uncompressed size {total:,} >= {LIMIT:,}")
    for n in names:
        if n.startswith("eval/") or "dev_shim" in n:
            errors.append(f"forbidden file in zip: {n}")
    if errors:
        print("[build] ERROR: " + "; ".join(errors), file=sys.stderr)
        sys.exit(1)
    print("[build] OK")


if __name__ == "__main__":
    main()
