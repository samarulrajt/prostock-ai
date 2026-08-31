"""Replace pro_model.h5 / pro_scaler.pkl on disk. Next predict() loads the new files."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib

import research_policy as rp

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
MAX_MODEL_BYTES = 80 * 1024 * 1024


def _archive_dir() -> Path:
    return rp.ROOT / "models" / "archive"


def model_status() -> dict:
    def info(path: Path):
        if not path.is_file():
            return {"present": False, "bytes": 0, "mtime": None}
        st = path.stat()
        return {
            "present": True,
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    return {
        "model": info(rp.MODEL_PATH),
        "scaler": info(rp.SCALER_PATH),
        "metrics": info(rp.METRICS_PATH),
    }


def _archive_current() -> None:
    archive = _archive_dir()
    archive.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for src in (rp.MODEL_PATH, rp.SCALER_PATH, rp.METRICS_PATH):
        if src.is_file():
            shutil.copy2(src, archive / f"{src.name}.{stamp}")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".uploading")
    tmp.write_bytes(data)
    tmp.replace(path)


def _validate_h5(data: bytes) -> None:
    if len(data) < 64:
        raise ValueError("Model file is too small to be a Keras HDF5 weights file.")
    if len(data) > MAX_MODEL_BYTES:
        raise ValueError("Model file is larger than 80 MB.")
    if not data.startswith(HDF5_MAGIC):
        raise ValueError("Not an HDF5 file. Upload the .h5 Keras save (pro_model.h5).")


def _validate_scaler(data: bytes) -> None:
    if len(data) < 16:
        raise ValueError("Scaler file is too small.")
    tmp = rp.ROOT / ".scaler_upload_check.pkl"
    try:
        tmp.write_bytes(data)
        assets = joblib.load(tmp)
    except Exception as exc:
        raise ValueError(f"Could not unpickle scaler: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    if not isinstance(assets, dict) or "scaler" not in assets:
        raise ValueError("Scaler pickle must be a dict with a 'scaler' key (pro_scaler.pkl).")


def install_weights(model_bytes: bytes, scaler_bytes: bytes, metrics_bytes: bytes | None = None) -> dict:
    _validate_h5(model_bytes)
    _validate_scaler(scaler_bytes)
    if metrics_bytes:
        try:
            json.loads(metrics_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("metrics must be UTF-8 JSON (model_metrics.json).") from exc
    _archive_current()
    _atomic_write(rp.MODEL_PATH, model_bytes)
    _atomic_write(rp.SCALER_PATH, scaler_bytes)
    if metrics_bytes:
        _atomic_write(rp.METRICS_PATH, metrics_bytes)
    return model_status()
