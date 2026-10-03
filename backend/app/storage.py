"""Uploaded originals and their viewer PDFs, under DATA_DIR/uploads/<doc id>/.

The original is stored as <sha256>.<ext>, so a second upload of the same bytes is
recognised by its hash. DOCX files and images also get a viewer.pdf: the PDF the
viewer opens and the citations point into.
"""

import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import config

CHUNK = 1024 * 1024


@dataclass
class Received:
    path: Path
    sha256: str
    size: int
    too_large: bool


def receive(stream, max_bytes: int) -> Received:
    """Copy an upload to a temporary file, hashing it on the way. Stops past max_bytes."""
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    tmp = tempfile.NamedTemporaryFile(dir=config.UPLOAD_DIR, prefix=".upload-", delete=False)
    with tmp:
        while block := stream.read(CHUNK):
            size += len(block)
            if size > max_bytes:
                break
            digest.update(block)
            tmp.write(block)
    return Received(Path(tmp.name), digest.hexdigest(), size, size > max_bytes)


def doc_dir(doc_id: str) -> Path:
    return config.UPLOAD_DIR / doc_id


def keep(received: Received, doc_id: str, ext: str) -> str:
    """Move a received upload into the document's folder; returns its path relative to UPLOAD_DIR."""
    folder = doc_dir(doc_id)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{received.sha256}.{ext}" if ext else received.sha256
    shutil.move(str(received.path), folder / name)
    return f"{doc_id}/{name}"


def resolve(relative: str) -> Path:
    path = (config.UPLOAD_DIR / relative).resolve()
    if config.UPLOAD_DIR.resolve() not in path.parents:
        raise ValueError("Path outside the upload folder")
    return path


def viewer_relpath(doc_id: str) -> str:
    return f"{doc_id}/viewer.pdf"


def remove(doc_id: str) -> None:
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)


def discard(received: Received) -> None:
    received.path.unlink(missing_ok=True)
