"""Rust deep-filter binary backend — alternative to Python torch DeepFilterNet3.

Uses `deep-filter` CLI (from Rikorose/DeepFilterNet Rust implementation, tract ONNX runtime).
No torch dependency, ~15-30MB binary + ~10MB model, CPU-only, low RAM, built-in streaming.

Installation:
  cargo install deep_filter --features cli
  or download release from https://github.com/Rikorose/DeepFilterNet/releases

Usage: deep-filter checks for model in ~/.cache/deep_filter or similar, downloads on first run.
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional


def _find_binary() -> Optional[str]:
    for name in ("deep-filter", "deep_filter"):
        path = shutil.which(name)
        if path:
            return path
    return None


def rust_enhance(wav_in: str, wav_out: str, model: str = "DeepFilterNet3") -> None:
    """Enhance wav using Rust deep-filter binary."""
    binary = _find_binary()
    if binary is None:
        raise FileNotFoundError("deep-filter binary not found in PATH — install via cargo install deep_filter --features cli or download release")

    # deep-filter expects wav files, output dir
    # Usage: deep-filter [OPTIONS] [FILES]...
    # We'll use temp out dir then move
    out_dir = os.path.dirname(os.path.abspath(wav_out))
    os.makedirs(out_dir, exist_ok=True)

    # Create temp file for output to ensure atomicity similar to embed
    fd, tmp_out = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)

    # deep-filter writes to out dir with same basename, so we need to place input in temp? Simpler: use input directly and let it output to tmp dir, then move
    # Actually deep-filter: deep-filter --output-dir <dir> <input> -> creates <dir>/<basename>.wav
    # We'll use a dedicated temp out dir
    with tempfile.TemporaryDirectory(prefix="deepfilter_") as tmp_out_dir:
        # Default model is DeepFilterNet3, don't pass --model unless custom tar.gz path
        if model and model != "DeepFilterNet3" and os.path.isfile(model):
            cmd = [binary, "--model", model, "--output-dir", tmp_out_dir, wav_in]
        else:
            cmd = [binary, "--output-dir", tmp_out_dir, wav_in]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"deep-filter failed: {result.stderr[:1000]}")

        # Find output file — deep-filter preserves basename
        basename = os.path.basename(wav_in)
        # It may output with .wav extension regardless? Check
        candidates = [os.path.join(tmp_out_dir, basename), os.path.join(tmp_out_dir, os.path.splitext(basename)[0] + ".wav")]
        found = None
        for cand in candidates:
            if os.path.exists(cand):
                found = cand
                break
        # Also list dir
        if found is None:
            files = os.listdir(tmp_out_dir)
            if files:
                found = os.path.join(tmp_out_dir, files[0])

        if found is None or not os.path.exists(found):
            raise RuntimeError(f"deep-filter did not produce output in {tmp_out_dir}")

        # Validate and move to tmp_out then atomic replace to final wav_out
        if os.path.getsize(found) < 1024:
            raise RuntimeError(f"deep-filter produced invalid output size {os.path.getsize(found)}")

        # If wav_out is already temp file we created, replace it
        try:
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)
        except OSError:
            pass
        shutil.move(found, tmp_out)

    # Now atomic replace to final destination
    os.replace(tmp_out, wav_out)
