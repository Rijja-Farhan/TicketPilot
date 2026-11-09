"""Stage 1, load: text files directly, PDFs through pdftotext. Unreadable files are quarantined, not dropped."""
import shutil
import subprocess
from pathlib import Path

from ticketpilot.ingest.model import Quarantined, RawFile

TEXT_SUFFIXES = {".md", ".txt"}


def load_file(path: Path) -> RawFile | Quarantined:
    name = path.name
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(errors="replace")
    elif path.suffix.lower() == ".pdf":
        if shutil.which("pdftotext") is None:
            return Quarantined(name, "pdftotext_missing")
        out = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, text=True, check=False)
        text = out.stdout
        if out.returncode == 0 and not text.replace("\f", "").strip():
            return Quarantined(name, "needs_ocr")  # image-only: the text exists only as pixels
    else:
        return Quarantined(name, "unsupported_type")
    if not text.strip():
        return Quarantined(name, "empty")
    return RawFile(name, text)


def load_dir(kb_dir: Path) -> tuple[list[RawFile], list[Quarantined]]:
    loaded, quarantined = [], []
    for p in sorted(kb_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            r = load_file(p)
            (loaded if isinstance(r, RawFile) else quarantined).append(r)
    return loaded, quarantined
