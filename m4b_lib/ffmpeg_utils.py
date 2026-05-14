"""ffmpeg/ffprobe subprocess wrappers + chapter and noise-floor helpers."""
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional


def get_duration(file_path: str) -> float:
    """Return the audio duration in seconds, or 0.0 on failure."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def transcode_mp3_to_m4a(mp3_path: str, out_path: str, bitrate: str = "64k") -> None:
    """Re-encode an mp3 to AAC m4a. Drops video/cover streams."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", mp3_path,
            "-vn", "-c:a", "aac", "-b:a", bitrate,
            out_path,
        ],
        check=True, capture_output=True,
    )


def parallel_transcode(mp3_files: Iterable[str], output_dir: str, max_workers: Optional[int] = None) -> list[str]:
    """Transcode many mp3s to m4a in parallel. Returns sorted list of m4a paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for mp3 in mp3_files:
            base = os.path.splitext(os.path.basename(mp3))[0]
            out = os.path.join(output_dir, base + ".m4a")
            paths.append(out)
            futures[ex.submit(transcode_mp3_to_m4a, mp3, out)] = out
        for f in as_completed(futures):
            f.result()  # raises on subprocess failure
    return sorted(paths)


def concat_audio_to_m4b(audio_files: list[str], output_m4b: str) -> None:
    """Concatenate ordered audio files into a single m4b via ffmpeg concat demuxer."""
    listfile = output_m4b + ".concat.txt"
    with open(listfile, "w") as f:
        for p in audio_files:
            # Escape apostrophes per ffmpeg concat demuxer spec
            safe_p = p.replace("'", r"'\''")
            f.write(f"file '{safe_p}'\n")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", listfile,
                "-c", "copy", output_m4b,
            ],
            check=True, capture_output=True,
        )
    finally:
        try:
            os.unlink(listfile)
        except OSError:
            pass


def extract_chapters(m4b_path: str, out_ini_path: str) -> None:
    """Extract chapter markers from an m4b into an ffmetadata file."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", m4b_path,
            "-f", "ffmetadata", out_ini_path,
        ],
        check=True, capture_output=True,
    )


def embed_chapters_and_meta(
    audio_in: str, chapters_ini: Optional[str], cover_bytes: Optional[bytes],
    output_m4b: str, title: str = "", author: str = "", bitrate: str = "64k",
) -> None:
    """Re-encode audio_in to m4b, attaching chapters from ffmetadata file + optional cover."""
    cmd = ["ffmpeg", "-y", "-i", audio_in]
    if chapters_ini:
        cmd += ["-i", chapters_ini, "-map_metadata", "1"]
    cover_path = None
    if cover_bytes:
        cover_path = output_m4b + ".cover.jpg"
        with open(cover_path, "wb") as f:
            f.write(cover_bytes)
        cmd += ["-i", cover_path, "-map", "0:a", "-map", f"{2 if chapters_ini else 1}:v", "-disposition:v:0", "attached_pic"]
    else:
        cmd += ["-map", "0:a"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if author:
        cmd += ["-metadata", f"artist={author}"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, output_m4b]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if cover_path and os.path.exists(cover_path):
            os.unlink(cover_path)


def probe_noise_floor(audio_path: str) -> float:
    """Estimate noise floor (dBFS) using ffmpeg volumedetect on a 5s window.

    Returns mean_volume in dB. Higher = louder file overall.
    Used by cleanup pre-flight to skip already-clean inputs.
    """
    r = subprocess.run(
        [
            "ffmpeg", "-i", audio_path, "-t", "5",
            "-af", "volumedetect", "-vn", "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", r.stderr)
    return float(m.group(1)) if m else 0.0
