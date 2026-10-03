"""DOCX -> PDF with LibreOffice headless (plan section 6.2 step 3).

DOCX has no fixed pages, so we convert it to PDF and cite that PDF's pages. If LibreOffice
is not installed, the caller falls back to paragraph anchors and tells the user.
"""

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

WINDOWS_PATHS = [r"C:\Program Files\LibreOffice\program\soffice.exe",
                 r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]
MAC_PATH = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
TIMEOUT_S = 180


@lru_cache(maxsize=1)
def soffice_path() -> str | None:
    configured = os.environ.get("SAKSHYA_LIBREOFFICE_CMD")
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else None
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for path in WINDOWS_PATHS + [MAC_PATH]:
        if Path(path).exists():
            return path
    return None


def status() -> dict:
    path = soffice_path()
    return {"available": path is not None, "path": path}


def docx_to_pdf(src: Path, dest: Path) -> None:
    """Convert src (.docx) to dest (.pdf). Raises RuntimeError on failure."""
    exe = soffice_path()
    if exe is None:
        raise RuntimeError("LibreOffice is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        source = work / "input.docx"
        shutil.copyfile(src, source)
        # A private profile directory lets several conversions run without clashing with an
        # open LibreOffice window.
        profile = (work / "profile").as_uri()
        cmd = [exe, f"-env:UserInstallation={profile}", "--headless", "--norestore",
               "--convert-to", "pdf", "--outdir", str(work), str(source)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        out = work / "input.pdf"
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"LibreOffice could not convert the file: {(proc.stderr or proc.stdout).strip()[:300]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), dest)
