"""m4b restoration pipeline.

Two modes:
  basic — sox noiseprof + noisered + highpass + acompressor + loudnorm
  ml    — DeepFilterNet 3 inference (see Task 9)

Common flow:
  1. Decode m4b -> wav
  2. Extract chapters + cover art from m4b
  3. Process wav with chosen pipeline
  4. Re-encode wav -> m4b with chapters + cover re-attached
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile
from typing import Iterable

from m4b_lib import ffmpeg_utils


SILENCE_DB_THRESH = -40       # dB below peak considered "silence"
SILENCE_MIN_DUR = 1.0         # seconds — minimum silence length for noise sampling
NOISE_AGGRESSION = 0.21       # sox noisered aggression (0.0-1.0; conservative)


def iter_targets(input_arg: str, pattern: str) -> Iterable[str]:
    """Yield m4b paths from either a single file or a directory + glob."""
    if os.path.isfile(input_arg):
        yield input_arg
    elif os.path.isdir(input_arg):
        for p in sorted(glob.glob(os.path.join(input_arg, pattern))):
            if p.lower().endswith(".m4b"):
                yield p
    else:
        raise FileNotFoundError(input_arg)


def _decode_to_wav(m4b_path: str, wav_path: str) -> None:
    """Decode any audio container into a 16-bit stereo wav."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", m4b_path,
            "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
            wav_path,
        ],
        check=True, capture_output=True,
    )


def _find_silence_window(wav_path: str) -> tuple[float, float] | None:
    """Return (start_sec, end_sec) of the first silence span >= SILENCE_MIN_DUR, or None."""
    r = subprocess.run(
        [
            "ffmpeg", "-i", wav_path,
            "-af", f"silencedetect=noise={SILENCE_DB_THRESH}dB:d={SILENCE_MIN_DUR}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True,
    )
    starts = re.findall(r"silence_start:\s*([\d.]+)", r.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", r.stderr)
    if starts and ends:
        return float(starts[0]), float(ends[0])
    return None


def _sox_pipeline(wav_in: str, wav_out: str, tmpdir: str) -> None:
    """Run the basic DSP cleanup chain via sox."""
    win = _find_silence_window(wav_in)
    if win is None:
        # No silence found — skip noise reduction, do gentle highpass + loudnorm only.
        print(f"  [warn] no silence window found in {wav_in}; skipping noise reduction")
        subprocess.run(
            [
                "sox", wav_in, wav_out,
                "highpass", "80",
                "lowpass", "14000",
                "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2",
            ],
            check=True, capture_output=True,
        )
        return

    start, end = win
    duration = max(0.5, min(2.0, end - start - 0.1))  # 0.5-2.0s sample
    noise_sample = os.path.join(tmpdir, "noise.wav")
    noise_profile = os.path.join(tmpdir, "noise.prof")

    subprocess.run(
        ["sox", wav_in, noise_sample, "trim", f"{start + 0.05}", f"{duration}"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["sox", noise_sample, "-n", "noiseprof", noise_profile],
        check=True, capture_output=True,
    )
    subprocess.run(
        [
            "sox", wav_in, wav_out,
            "noisered", noise_profile, str(NOISE_AGGRESSION),
            "highpass", "80",
            "lowpass", "14000",
            "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2",
        ],
        check=True, capture_output=True,
    )


def _loudness_normalize(wav_in: str, wav_out: str) -> None:
    """EBU R128 loudness normalize to -18 LUFS (audiobook convention)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_in,
            "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
            wav_out,
        ],
        check=True, capture_output=True,
    )


def clean_one(m4b_path: str, mode: str = "basic", keep_original: bool = False,
              skip_threshold_db: float = -35.0) -> None:
    """Run the cleanup pipeline on one m4b in place."""
    print(f"\n=== Cleaning: {m4b_path} (mode={mode}) ===")

    noise = ffmpeg_utils.probe_noise_floor(m4b_path)
    if noise < skip_threshold_db:
        print(f"  noise floor {noise:.1f}dB below threshold; skipping")
        return

    with tempfile.TemporaryDirectory(prefix="m4b_clean_") as tmp:
        wav_decoded = os.path.join(tmp, "decoded.wav")
        wav_cleaned = os.path.join(tmp, "cleaned.wav")
        wav_normalized = os.path.join(tmp, "normalized.wav")
        chapters_ini = os.path.join(tmp, "chapters.ini")
        cleaned_m4b = os.path.join(tmp, "cleaned.m4b")

        _decode_to_wav(m4b_path, wav_decoded)
        ffmpeg_utils.extract_chapters(m4b_path, chapters_ini)

        if mode == "basic":
            _sox_pipeline(wav_decoded, wav_cleaned, tmp)
        elif mode == "ml":
            from m4b_lib.cleanup_ml import df3_enhance
            df3_enhance(wav_decoded, wav_cleaned)
        else:
            raise ValueError(f"unknown mode: {mode}")

        _loudness_normalize(wav_cleaned, wav_normalized)
        ffmpeg_utils.embed_chapters_and_meta(
            wav_normalized, chapters_ini=chapters_ini, cover_bytes=None,
            output_m4b=cleaned_m4b,
        )

        if keep_original:
            backup = os.path.splitext(m4b_path)[0] + ".orig.m4b"
            shutil.move(m4b_path, backup)
            print(f"  original kept at: {backup}")
        shutil.move(cleaned_m4b, m4b_path)
        print(f"  done: {m4b_path}")
