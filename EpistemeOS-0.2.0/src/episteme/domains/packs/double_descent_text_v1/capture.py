"""Read one bundle.json. This is the pack's only filesystem hook."""

from pathlib import Path
import hashlib
import json
import stat

from episteme.domains import api


PACK_ID = "double_descent_text_v1"
PACK_VERSION = "1"
MAX_FILE_BYTES = 64 * 1024 * 1024
NAME = "bundle.json"


def _unsafe(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def read_directory(source):
    root = Path(source).absolute()
    if not root.is_dir() or _unsafe(root):
        raise ValueError("capture source must be a regular directory")
    for ancestor in root.parents:
        if _unsafe(ancestor):
            raise ValueError("capture path traverses a link or reparse point")
    root = root.resolve()
    names = []
    for item in root.iterdir():
        if _unsafe(item):
            raise ValueError(f"capture source contains a link: {item.name}")
        if not item.is_file():
            raise ValueError(f"capture source contains a non-file: {item.name}")
        names.append(item.name)
    if names != [NAME]:
        raise ValueError("capture source must contain exactly bundle.json")
    path = root / NAME
    if _unsafe(path) or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("bundle.json must be a regular file")
    with path.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES or len(data) > api.MAX_BLOB_BYTES:
        raise ValueError("bundle.json exceeds the capture limit")
    document = json.loads(data.decode("utf-8"))
    origin = document.get("origin")
    if origin not in ("synthetic_fixture", "full_pipeline"):
        raise ValueError("bundle origin is not recognised")
    digest = hashlib.sha256(data).hexdigest()
    return api.CaptureBundle(
        pack_id=PACK_ID, pack_version=PACK_VERSION,
        source_label=f"{origin}-{digest[:12]}", files={NAME: data},
        audit={"origin": origin, "bytes": len(data),
               "modelwise_rows": len(document.get("modelwise", [])),
               "epochwise_rows": len(document.get("epochwise", [])),
               "test_rows": len(document.get("y_test", []))})
