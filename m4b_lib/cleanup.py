"""m4b restoration pipeline.

Two modes:
  basic — sox noiseprof + noisered + highpass + acompressor + loudnorm
  ml    — DeepFilterNet 3 inference (chunked for long files)

Common flow:
  1. Decode m4b -> wav
  2. Extract chapters + cover from m4b
  3. Process wav with chosen pipeline
  4. Re-encode wav -> m4b with chapters + cover preserved
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from m4b_lib import ffmpeg_utils


SILENCE_DB_THRESH = -40       # dB below peak considered "silence"
SILENCE_MIN_DUR = 1.0         # seconds — minimum silence length for noise sampling
NOISE_AGGRESSION = 0.27       # sox noisered aggression (0.0-1.0; 0.21 old conservative, 0.27-0.30 better for hiss)
TARGET_I = -19                # LUFS integrated loudness target (ACX -19 typical, -19 mono == -16 stereo)
TARGET_TP = -2.0              # dB true peak, -3.0 for strict ACX, -2.0 safe for AAC 64k
TARGET_LRA = 7                # loudness range, 7 LU pro for narration (was 11 too wide)
HP_FREQ = 70                  # highpass Hz, 70 better than 80 for male warmth (was 80)
LP_FREQ = 16000               # lowpass Hz, 16k preserves air vs 14k dull (AAC 64k already ~15-16k)


def iter_targets(input_arg: str, pattern: str) -> Iterable[str]:
    """Yield m4b paths from either a single file or a directory + glob.

    - File branch: yields only .m4b files (strict filter). Non-m4b files are ignored with warning.
    - Dir branch: glob with recursive support for ** patterns, sorted, filters .m4b, skips broken symlinks.
    - Raises FileNotFoundError for missing path or broken symlink pointing to missing target.
    """
    p = Path(input_arg)
    if p.is_file() or (os.path.lexists(input_arg) and not p.exists()):
        if not p.exists() and os.path.lexists(input_arg):
            raise FileNotFoundError(f"Broken symlink or inaccessible: {input_arg}")
        if p.suffix.lower() != ".m4b":
            print(f"  [warn] skipping non-m4b file: {input_arg}")
            return
        yield str(p)
        return
    elif p.is_dir():
        # Use Path.rglob with follow_symlinks=False to avoid symlink loops
        # For pattern without **, use glob; for **, use rglob
        yielded = 0
        try:
            if "**" in pattern:
                # rglob ignores pattern prefix, so we need to handle **/*.m4b -> *.m4b recursive
                # Extract suffix after **
                suffix = pattern.split("**")[-1].lstrip("/\\")
                if not suffix:
                    suffix = "*.m4b"
                # Use rglob with follow_symlinks=False
                for fp_path in sorted(p.rglob(suffix)):
                    # Skip symlink dirs already (rglob with follow_symlinks=False does, but double-check)
                    try:
                        if fp_path.is_symlink() and fp_path.is_dir():
                            continue
                    except OSError:
                        continue
                    fp = str(fp_path)
                    if fp.lower().endswith(".m4b") and os.path.isfile(fp):
                        yield fp
                        yielded += 1
            else:
                # Non-recursive: use glob but still check is_file and not symlink dir loop
                matched = glob.glob(os.path.join(input_arg, pattern), recursive=False)
                for fp in sorted(matched):
                    # Avoid following symlink dirs
                    try:
                        if os.path.islink(fp) and os.path.isdir(fp):
                            continue
                    except OSError:
                        continue
                    if fp.lower().endswith(".m4b") and os.path.isfile(fp):
                        yield fp
                        yielded += 1
        except Exception as e:
            print(f"  [warn] glob failed for {input_arg} pattern {pattern}: {e}")
        return
    else:
        if os.path.lexists(input_arg):
            raise FileNotFoundError(f"Broken symlink or inaccessible: {input_arg}")
        raise FileNotFoundError(input_arg)


def _decode_to_wav(m4b_path: str, wav_path: str, channels: int = 2, sample_rate: int = 44100) -> None:
    """Decode any audio container into wav."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", m4b_path,
            "-vn", "-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            wav_path,
        ],
        check=True, capture_output=True, timeout=300,
    )


def _find_silence_window(wav_path: str) -> tuple[float, float] | None:
    """Return (start_sec, end_sec) of the first silence span >= SILENCE_MIN_DUR, or None."""
    r = subprocess.run(
        [
            "ffmpeg", "-i", wav_path,
            "-af", f"silencedetect=noise={SILENCE_DB_THRESH}dB:d={SILENCE_MIN_DUR}",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        # ffmpeg failed (corrupt wav) - treat as no silence, caller will use fallback
        print(f"  [warn] silencedetect failed (rc={r.returncode}): {r.stderr[:300]}")
        return None
    starts = re.findall(r"silence_start:\s*([\d.]+)", r.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", r.stderr)
    if starts and ends:
        try:
            return float(starts[0]), float(ends[0])
        except ValueError:
            return None
    return None


def _sox_pipeline(wav_in: str, wav_out: str, tmpdir: str) -> None:
    """Run the basic DSP cleanup chain via sox — pro-tuned: HP 70Hz, LP 16k, gate-downward compand."""
    if shutil.which("sox") is None:
        raise FileNotFoundError("sox not found in PATH — required for basic cleanup mode")

    win = _find_silence_window(wav_in)
    if win is None:
        print(f"  [warn] no silence window found in {wav_in}; skipping noise reduction")
        try:
            subprocess.run(
                [
                    "sox", wav_in, wav_out,
                    "highpass", str(HP_FREQ),
                    "lowpass", str(LP_FREQ),
                    "compand", "0.05,0.2", "-60,-90,-40,-40,-20,-10,0,-5", "0", "-90", "0.1",
                ],
                check=True, capture_output=True, text=True, timeout=120,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"sox fallback pipeline failed: {e.stderr}") from e
        return

    start, end = win
    # Guard against silence window at tail where trim would go past EOF
    try:
        file_dur = ffmpeg_utils.get_duration(wav_in)
    except Exception:
        file_dur = None

    # Clamp duration to 0.5-2.0s and ensure it doesn't go past EOF
    raw_dur = end - start - 0.1
    duration = max(0.5, min(2.0, raw_dur))
    if file_dur is not None and file_dur > 0:
        remaining = file_dur - (start + 0.05)
        if remaining <= 0.2:
            print(f"  [warn] silence window near EOF (start={start:.2f}, file_dur={file_dur:.2f}); skipping noise reduction")
            subprocess.run(
                [
                    "sox", wav_in, wav_out,
                    "highpass", str(HP_FREQ),
                    "lowpass", str(LP_FREQ),
                    "compand", "0.05,0.2", "-60,-90,-40,-40,-20,-10,0,-5", "0", "-90", "0.1",
                ],
                check=True, capture_output=True, text=True, timeout=120,
            )
            return
        duration = min(duration, remaining - 0.05)
        duration = max(0.3, duration)

    noise_sample = os.path.join(tmpdir, "noise.wav")
    noise_profile = os.path.join(tmpdir, "noise.prof")

    try:
        subprocess.run(
            ["sox", wav_in, noise_sample, "trim", f"{start + 0.05}", f"{duration}"],
            check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["sox", noise_sample, "-n", "noiseprof", noise_profile],
            check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            [
                "sox", wav_in, wav_out,
                "noisered", noise_profile, str(NOISE_AGGRESSION),
                "highpass", str(HP_FREQ),
                "lowpass", str(LP_FREQ),
                "compand", "0.05,0.2", "-60,-90,-40,-40,-20,-10,0,-5", "0", "-90", "0.1",
            ],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"sox pipeline failed: {e.stderr or e}") from e


def _loudness_normalize(wav_in: str, wav_out: str, channels: int = 2) -> None:
    """Two-pass EBU R128 loudness normalize to TARGET_I LUFS with dual_mono for mono.

    Pro standard: I=-19 TP=-2 LRA=7 dual_mono true for mono, linear=true second pass
    to avoid limiter pumping. Uses soxr high-precision resampler.
    """
    import json

    dual_mono = "true" if channels == 1 else "false"

    # Pass 1: measure
    result = subprocess.run(
        [
            "ffmpeg", "-i", wav_in,
            "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:dual_mono={dual_mono}:print_format=json",
            "-f", "null", "-",
        ],
        capture_output=True, text=True, timeout=600,
    )
    # Parse JSON from stderr (ffmpeg prints json to stderr)
    try:
        # ffmpeg loudnorm first pass prints a JSON block with input_i, input_tp, input_lra, input_thresh, target_offset
        # Find the JSON containing input_i
        json_match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', result.stderr, re.DOTALL)
        if not json_match:
            # Fallback: try any JSON blob
            json_match = re.search(r'\{.*\}', result.stderr, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in loudnorm output")
        stats = json.loads(json_match.group(0))
    except Exception as e:
        print(f"  [warn] loudnorm first pass JSON parse failed ({e}), falling back to single-pass")
        # Fallback single-pass
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_in,
                "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:dual_mono={dual_mono}",
                "-ar", "44100", "-ac", str(channels), "-c:a", "pcm_s16le",
                wav_out,
            ],
            check=True, capture_output=True, timeout=120,
        )
        return

    # Extract measured values from first pass (input_* becomes measured_* for second pass)
    try:
        measured_i = stats.get("input_i")
        measured_tp = stats.get("input_tp")
        measured_lra = stats.get("input_lra")
        measured_thresh = stats.get("input_thresh")
        target_offset = stats.get("target_offset", 0)
        if None in (measured_i, measured_tp, measured_lra, measured_thresh):
            raise ValueError(f"Incomplete stats: {stats}")
    except Exception as e:
        print(f"  [warn] loudnorm stats parse incomplete ({e}), falling back to single-pass")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_in,
                "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:dual_mono={dual_mono}",
                "-ar", "44100", "-ac", str(channels), "-c:a", "pcm_s16le",
                wav_out,
            ],
            check=True, capture_output=True, timeout=120,
        )
        return

    # Second pass: apply measured values with linear=true for transparent gain only
    # Use soxr high-precision resampler
    af = (
        f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
        f"measured_I={measured_i}:measured_TP={measured_tp}:measured_LRA={measured_lra}:"
        f"measured_thresh={measured_thresh}:offset={target_offset}:"
        f"linear=true:dual_mono={dual_mono}:print_format=summary"
    )

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_in,
            "-af", af,
            "-ar", "44100", "-ac", str(channels),
            "-c:a", "pcm_s16le",
            wav_out,
        ],
        check=True, capture_output=True, timeout=120,
    )


def clean_one(m4b_path: str, mode: str = "basic", keep_original: bool = False,
              skip_threshold_db: float = -35.0, chunk_s: float = 60.0, overlap_s: float = 2.0,
              device: str = None) -> None:
    """Run the cleanup pipeline on one m4b in place, preserving chapters + cover.

    chunk_s/overlap_s/device only used for ml mode.
    """
    print(f"\n=== Cleaning: {m4b_path} (mode={mode}) ===")

    noise = ffmpeg_utils.probe_noise_floor(m4b_path)
    if noise is None:
        print(f"  [warn] could not probe mean volume, proceeding with cleaning")
    elif noise < skip_threshold_db:
        print(f"  mean volume {noise:.1f}dB below threshold {skip_threshold_db}dB; skipping")
        return
    else:
        print(f"  mean volume {noise:.1f}dB (threshold {skip_threshold_db}dB) -> cleaning")

    # Use same filesystem as output for atomic replace safety
    output_parent = os.path.dirname(os.path.abspath(m4b_path))
    with tempfile.TemporaryDirectory(prefix="m4b_clean_", dir=output_parent) as tmp:
        wav_decoded = os.path.join(tmp, "decoded.wav")
        wav_cleaned = os.path.join(tmp, "cleaned.wav")
        wav_normalized = os.path.join(tmp, "normalized.wav")
        chapters_ini = os.path.join(tmp, "chapters.ini")
        cleaned_m4b = os.path.join(tmp, "cleaned.m4b")

        # Choose decode params based on mode to avoid unnecessary resampling
        # Both modes now decode mono for final mono 64k standard (pro audiobook mono)
        # Basic: 1ch 44.1k keeps native SR, ML/ML-Rust: 1ch 48k matches DF3 native
        if mode in ("ml", "ml-rust"):
            _decode_to_wav(m4b_path, wav_decoded, channels=1, sample_rate=48000)
        else:
            _decode_to_wav(m4b_path, wav_decoded, channels=1, sample_rate=44100)

        # Extract chapters and cover from source
        try:
            ffmpeg_utils.extract_chapters(m4b_path, chapters_ini)
        except subprocess.CalledProcessError as e:
            print(f"  [warn] chapter extraction failed: {e.stderr[:200] if e.stderr else e}")
            chapters_ini = None  # proceed without chapters

        cover_bytes = ffmpeg_utils.extract_cover(m4b_path)
        if cover_bytes:
            print(f"  cover preserved ({len(cover_bytes)} bytes)")
        else:
            print(f"  no cover found (will be stripped)")

        if mode == "basic":
            _sox_pipeline(wav_decoded, wav_cleaned, tmp)
        elif mode == "ml":
            # ml: Python torch DF3, with Rust fallback if binary available (old behavior)
            try:
                from m4b_lib.cleanup_ml_rust import _find_binary, rust_enhance
                if _find_binary() is not None:
                    print(f"  using Rust deep-filter backend ({_find_binary()}) for ml mode")
                    rust_enhance(wav_decoded, wav_cleaned)
                else:
                    raise FileNotFoundError("Rust binary not found")
            except Exception as e:
                print(f"  Rust backend not available ({e}), falling back to Python torch DF3")
                from m4b_lib.cleanup_ml import df3_enhance, _ensure_model
                if device:
                    _ensure_model(device_override=device)
                df3_enhance(wav_decoded, wav_cleaned, chunk_s=chunk_s, overlap_s=overlap_s)
        elif mode == "ml-rust":
            # ml-rust: Rust binary only, no torch fallback — explicit 3rd option for A/B
            from m4b_lib.cleanup_ml_rust import _find_binary, rust_enhance
            binary = _find_binary()
            if binary is None:
                raise FileNotFoundError(
                    "deep-filter Rust binary not found in PATH — install via "
                    "cargo install deep_filter --features cli or download from "
                    "https://github.com/Rikorose/DeepFilterNet/releases"
                )
            print(f"  using Rust deep-filter backend ({binary}) [ml-rust mode]")
            rust_enhance(wav_decoded, wav_cleaned)
        else:
            raise ValueError(f"unknown mode: {mode}")

        # Loudness normalize — unified mono 64k standard for audiobooks (pro: 64k mono = higher quality per channel than 64k stereo)
        _loudness_normalize(wav_cleaned, wav_normalized, channels=1)

        ffmpeg_utils.embed_chapters_and_meta(
            wav_normalized, chapters_ini=chapters_ini, cover_bytes=cover_bytes,
            output_m4b=cleaned_m4b, tmpdir=tmp,
        )

        # Validate output before overwriting original
        if not os.path.exists(cleaned_m4b) or os.path.getsize(cleaned_m4b) < 1024:
            raise RuntimeError(f"cleaning produced invalid output: {cleaned_m4b} size={os.path.getsize(cleaned_m4b) if os.path.exists(cleaned_m4b) else 'missing'}")

        # Duration validation: cleaned should be within 1% or 0.5s of decoded
        try:
            orig_dur = ffmpeg_utils.get_duration(m4b_path)
            cleaned_dur = ffmpeg_utils.get_duration(cleaned_m4b)
            if orig_dur is not None and cleaned_dur is not None:
                diff = abs(cleaned_dur - orig_dur)
                if diff > max(0.5, orig_dur * 0.01):
                    print(f"  [warn] duration drift: original {orig_dur:.1f}s vs cleaned {cleaned_dur:.1f}s diff {diff:.1f}s")
        except Exception as e:
            print(f"  [warn] duration validation failed: {e}")

        # Handle keep_original safely: copy original to backup BEFORE replacing, so if replace fails
        # original still exists and we don't end up with missing file
        if keep_original:
            backup = os.path.splitext(m4b_path)[0] + ".orig.m4b"
            # Avoid clobbering existing backup — version it
            if os.path.exists(backup):
                base, ext = os.path.splitext(backup)
                i = 1
                while os.path.exists(f"{base}.{i}{ext}"):
                    i += 1
                backup = f"{base}.{i}{ext}"
            # Copy, not move, to preserve original until replace succeeds
            shutil.copy2(m4b_path, backup)
            print(f"  original kept at: {backup}")

        # Atomic replace if same FS (tmp is in same dir as output, so replace is atomic)
        try:
            os.replace(cleaned_m4b, m4b_path)
        except OSError as e:
            # Fallback across FS (shouldn't happen since tmp is in output_parent)
            # If keep_original was True, original already backed up, but we copied not moved,
            # so original still exists. Try move, and if that also fails, backup remains.
            try:
                shutil.move(cleaned_m4b, m4b_path)
            except Exception as move_e:
                # If we had backed up via copy, we still have original intact
                # Clean up and raise with context
                raise RuntimeError(f"Failed to replace {m4b_path}: replace failed {e}, move failed {move_e}") from move_e

        print(f"  done: {m4b_path}")
