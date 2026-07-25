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
    # Also check common local paths
    for p in [os.path.expanduser("~/.cargo/bin/deep-filter"), os.path.expanduser("~/.local/bin/deep-filter")]:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def ensure_rust_binary(version: str = "v0.5.6") -> str:
    """Auto-install Rust deep-filter binary if missing, like ffmpeg guidance.

    Downloads musl static binary for Linux x86_64 from GitHub releases to ~/.local/bin/deep-filter.
    Returns path to binary.
    """
    existing = _find_binary()
    if existing:
        return existing

    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Only auto-install for Linux x86_64 for now, else error with instructions
    if system != "linux" or machine not in ("x86_64", "amd64"):
        raise FileNotFoundError(
            f"Auto-install only supports Linux x86_64, detected {system} {machine}. "
            "Install manually: cargo install deep_filter or download from "
            "https://github.com/Rikorose/DeepFilterNet/releases"
        )

    # Use musl static which works everywhere
    url = f"https://github.com/Rikorose/DeepFilterNet/releases/download/{version}/deep-filter-{version}-x86_64-unknown-linux-musl"
    dest_dir = os.path.expanduser("~/.local/bin")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "deep-filter")

    print(f"  downloading deep-filter {version} from {url} to {dest_path}...")

    # Download via curl or python requests
    try:
        import requests
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception:
        # Fallback to curl via subprocess
        result = subprocess.run(["curl", "-L", "-o", dest_path, url], capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not os.path.exists(dest_path):
            raise FileNotFoundError(f"Failed to download deep-filter from {url}: {result.stderr[:500]}")

    os.chmod(dest_path, 0o755)

    # Ensure dest_dir in PATH for future which checks — add to PATH env for current process
    os.environ["PATH"] = dest_dir + os.pathsep + os.environ.get("PATH", "")

    # Verify
    if not os.path.isfile(dest_path):
        raise FileNotFoundError(f"Auto-install failed, {dest_path} not found")

    # Test run --help to ensure binary works and model will be downloaded later
    try:
        subprocess.run([dest_path, "--help"], capture_output=True, timeout=10)
    except Exception:
        pass

    return dest_path


def rust_enhance(wav_in: str, wav_out: str, model: str = "DeepFilterNet3",
                 atten_lim_db: float = 100.0, pf: bool = False, pf_beta: float = 0.02,
                 compensate_delay: bool = False) -> None:
    """Enhance wav using Rust deep-filter binary.

    Knobs:
      atten_lim_db: 0-100 dB, 0=no reduction, 100=full. For audiobooks 20-30dB keeps natural room tone.
      pf: enable post-filter (over-attenuates very noisy)
      pf_beta: 0.02 default, higher = stronger post-filter
      compensate_delay: add padding to compensate STFT lookahead delay
    """
    binary = _find_binary()
    if binary is None:
        raise FileNotFoundError("deep-filter binary not found in PATH — install via cargo install deep_filter --features cli or download release")

    out_dir = os.path.dirname(os.path.abspath(wav_out))
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_out = tempfile.mkstemp(suffix=".wav", dir=out_dir)
    os.close(fd)

    with tempfile.TemporaryDirectory(prefix="deepfilter_") as tmp_out_dir:
        cmd = [binary, "--output-dir", tmp_out_dir, "--atten-lim-db", str(atten_lim_db)]
        if pf:
            cmd.append("--pf")
            if pf_beta != 0.02:
                cmd.extend(["--pf-beta", str(pf_beta)])
        if compensate_delay:
            cmd.append("--compensate-delay")
        if model and model != "DeepFilterNet3" and os.path.isfile(model):
            cmd.extend(["--model", model])
        # Input file last
        cmd.append(wav_in)

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
