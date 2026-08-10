"""ffmpeg/ffprobe subprocess wrappers + chapter and noise-floor helpers."""
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional


def duration_timeout(audio_seconds: Optional[float], minimum: float = 300.0,
                      factor: float = 0.25) -> float:
    """Wall-clock bound for an ffmpeg call over `audio_seconds` of audio.

    This is hang detection, not a performance budget, so the bound must stay
    far looser than measured throughput while still catching a real hang in
    bounded time. factor=0.25 gives ~11x margin over the measured single-worker
    loudnorm baseline (0.25D vs. D/45) and ~3.75x even against a hypothetical
    3x-worse-than-baseline contention scenario — false positives stay
    implausible — while capping the undetected-hang window at a quarter of the
    book's length instead of the full length a factor of 1.0 would allow.
    """
    if not audio_seconds or audio_seconds <= 0:
        return minimum
    return max(minimum, audio_seconds * factor)


def get_duration(file_path: str) -> Optional[float]:
    """Return the audio duration in seconds, or None on failure."""
    r = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True, text=True, errors="replace", timeout=30,
    )
    if r.returncode != 0:
        return None
    out = r.stdout.strip()
    if not out:
        return None
    try:
        val = float(out)
        if val <= 0:
            return None
        return val
    except ValueError:
        return None


def transcode_mp3_to_m4a(mp3_path: str, out_path: str, bitrate: str = "64k") -> None:
    """Re-encode an mp3 to AAC m4a. Drops video/cover streams."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", mp3_path,
            "-vn", "-c:a", "aac", "-b:a", bitrate,
            out_path,
        ],
        check=True, capture_output=True, timeout=300,
    )


def parallel_transcode(mp3_files: Iterable[str], output_dir: str, bitrate: str = "64k",
                       max_workers: Optional[int] = None) -> list[str]:
    """Transcode many mp3s to m4a in parallel. Returns list in input order.

    Preserves natural order (no alphabetical re-sort). Handles duplicate basenames
    by adding suffix (_1, _2). On failure, only deletes files newly created by this
    run, never pre-existing files, and cancels pending futures.
    """
    os.makedirs(output_dir, exist_ok=True)
    mp3_list = list(mp3_files)  # materialize to preserve order
    # Track basename collision — make unique output names, avoiding existing files on disk
    basename_counts: dict[str, int] = {}
    used_names: set[str] = set()
    paths: list[str] = []
    orig_exists: set[str] = set()
    # Snapshot existing files to avoid overwriting + TOCTOU protection via O_EXCL reservation
    existing_on_disk = set(os.listdir(output_dir)) if os.path.isdir(output_dir) else set()
    for mp3 in mp3_list:
        base = os.path.splitext(os.path.basename(mp3))[0]
        count = basename_counts.get(base, 0)
        while True:
            out_name = base + ".m4a" if count == 0 else f"{base}_{count}.m4a"
            if out_name not in used_names and out_name not in existing_on_disk:
                # Try to reserve with O_EXCL|O_NOFOLLOW to prevent symlink race
                out_path = os.path.join(output_dir, out_name)
                try:
                    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
                    os.close(fd)
                    # Successfully reserved, now we will overwrite with ffmpeg -y (we own the file)
                    break
                except FileExistsError:
                    # File exists or symlink, try next suffix
                    count += 1
                    continue
                except OSError:
                    # Other error (e.g., permission), try next
                    count += 1
                    continue
            count += 1
        basename_counts[base] = count + 1
        used_names.add(out_name)
        out = os.path.join(output_dir, out_name)
        paths.append(out)
        # Reserved file exists now, not pre-existing (we created it)
        # If file existed before our reservation attempt, we already avoided it

    # Guard: if any output path already exists that we didn't just create (race), raise
    # This should not happen due to O_EXCL reservation, but keep as safety
    pre_existing_found = []
    for p in paths:
        # If file existed before and we didn't create it via reservation, it's in existing_on_disk
        # Our reservation ensures we created it, so existing_on_disk check already avoided
        if os.path.basename(p) in existing_on_disk and p not in [os.path.join(output_dir, n) for n in used_names]:
            pre_existing_found.append(p)
    # No raise needed since we version away from existing, but keep for external race
    if pre_existing_found:
        # Clean up reserved empty files we created
        for p in paths:
            try:
                if os.path.exists(p) and os.path.getsize(p) == 0:
                    os.unlink(p)
            except OSError:
                pass
        raise FileExistsError(f"Refusing to overwrite pre-existing files: {pre_existing_found}")

    # Use ThreadPoolExecutor to avoid fork-after-threads deadlock with torch
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for mp3, out in zip(mp3_list, paths):
            fut = ex.submit(transcode_mp3_to_m4a, mp3, out, bitrate)
            futures[fut] = out
        try:
            for f in as_completed(futures):
                f.result()
        except Exception:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                pass
            # Only delete files that didn't exist before this run
            for p in paths:
                if p in orig_exists:
                    continue
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except OSError:
                    pass
            raise
    # Return in input order, not sorted, to preserve natural chapter order
    return paths


def concat_audio_to_m4b(audio_files: list[str], output_m4b: str, mp3_passthrough: bool = False) -> None:
    """Concatenate ordered audio files into a single m4b via ffmpeg concat demuxer.

    mp3_passthrough=True is for bind's audio_mode="copy": ffmpeg's extension-
    inferred "ipod" muxer refuses to mux an mp3 stream ("Could not find tag
    for codec mp3 in stream"), so this pins the muxer to plain mp4 (which
    accepts mp3 and aac alike) and pins the brand back to M4A, since mp4's
    own default brand is the generic "isom" rather than the M4A/M4B identity
    players use to recognize audiobook-shaped files. Every other (existing)
    caller leaves the muxer selection untouched.
    """
    if not audio_files:
        raise ValueError("concat_audio_to_m4b: no input files")
    import tempfile
    # Use mkstemp in same dir as output to avoid symlink attack and cross-FS issues
    out_dir = os.path.dirname(os.path.abspath(output_m4b)) or None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fd, listfile = tempfile.mkstemp(suffix=".concat.txt", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for p in audio_files:
                if "\n" in p or "\r" in p:
                    raise ValueError(f"concat list does not support newline in path: {p!r}")
                safe_p = p.replace("\\", "\\\\").replace("'", r"'\''")
                f.write(f"file '{safe_p}'\n")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", listfile,
            "-c", "copy",
        ]
        if mp3_passthrough:
            cmd += ["-f", "mp4", "-brand", "M4A "]
        cmd += [output_m4b]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
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
        check=True, capture_output=True, timeout=120,
    )


def embed_chapters_and_meta(
    audio_in: str, chapters_ini: Optional[str], cover_bytes: Optional[bytes],
    output_m4b: str, title: str = "", author: str = "", bitrate: str = "64k",
    tmpdir: Optional[str] = None, copy_audio: bool = False,
    mp3_passthrough: bool = False,
) -> None:
    """Re-encode audio_in to m4b, attaching chapters from ffmetadata file + optional cover.

    If cover_bytes present, writes a temporary jpg in tmpdir (or same dir as output)
    and attaches it as mjpeg attached_pic with front-cover metadata.
    Adds faststart for streaming.

    If copy_audio is True, the audio stream is copied without re-encoding
    (used by bind to avoid double lossy transcode). Otherwise re-encodes to aac.

    mp3_passthrough=True is for bind's audio_mode="copy": same "ipod" muxer
    problem and same fix as concat_audio_to_m4b (see its docstring) -- pin
    the muxer to mp4 and the brand back to M4A only when an mp3 stream is
    actually being copied through. Every other (existing) caller is
    unaffected: the muxer is left to ffmpeg's own extension inference.
    """
    import tempfile
    # Ensure final_dir exists before any mkstemp that may use it
    final_dir = os.path.dirname(os.path.abspath(output_m4b))
    os.makedirs(final_dir, exist_ok=True)
    # Determine where to put temporary cover file (same filesystem as output for atomicity)
    cover_tmpdir = tmpdir or final_dir or None
    if cover_tmpdir:
        os.makedirs(cover_tmpdir, exist_ok=True)

    # Collect all inputs first, then add mapping options (ffmpeg requires -map_metadata after inputs)
    cmd = ["ffmpeg", "-y", "-i", audio_in]
    input_index = 1
    chapters_input_index = None
    if chapters_ini:
        cmd += ["-i", chapters_ini]
        chapters_input_index = input_index
        input_index += 1

    cover_path = None
    cover_input_index = None
    if cover_bytes:
        fd, cover_path = tempfile.mkstemp(suffix=".jpg", dir=cover_tmpdir)
        with os.fdopen(fd, "wb") as f:
            f.write(cover_bytes)
        cmd += ["-i", cover_path]
        cover_input_index = input_index
        input_index += 1

    # Now mapping / metadata options after all inputs
    if chapters_input_index is not None:
        cmd += ["-map_metadata", str(chapters_input_index)]

    # Audio mapping always from first input
    cmd += ["-map", "0:a"]

    # Video/cover mapping if present
    if cover_input_index is not None:
        cmd += ["-map", f"{cover_input_index}:v",
                "-c:v", "mjpeg",
                "-metadata:s:v", 'title="Cover (front)"',
                "-metadata:s:v", 'comment="Cover (front)"',
                "-disposition:v:0", "attached_pic"]

    if title:
        safe_title = title.replace("\n", " ").replace("\r", " ")
        cmd += ["-metadata", f"title={safe_title}"]
    if author:
        safe_author = author.replace("\n", " ").replace("\r", " ")
        cmd += ["-metadata", f"artist={safe_author}"]

    if copy_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", bitrate]

    if mp3_passthrough:
        cmd += ["-f", "mp4", "-brand", "M4A "]

    # H1 fix: write to temp file in same dir as final output, then atomic replace
    fd, tmp_output = tempfile.mkstemp(suffix=".tmp.m4b", dir=final_dir)
    os.close(fd)

    cmd += ["-movflags", "faststart", tmp_output]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, output=result.stdout, stderr=result.stderr
            )
        # Validate temp output
        if not os.path.exists(tmp_output) or os.path.getsize(tmp_output) < 1024:
            raise RuntimeError(f"ffmpeg produced invalid output: {tmp_output} size {os.path.getsize(tmp_output) if os.path.exists(tmp_output) else 'missing'}")
        # Atomic replace final destination
        os.replace(tmp_output, output_m4b)
    except Exception:
        # Clean up temp output on failure
        try:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
        except OSError:
            pass
        raise
    finally:
        if cover_path and os.path.exists(cover_path):
            try:
                os.unlink(cover_path)
            except OSError:
                pass
        # Ensure tmp_output removed if still exists (e.g., after successful replace it's gone)
        try:
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
        except OSError:
            pass


def probe_noise_floor(audio_path: str) -> Optional[float]:
    """Estimate mean volume (dBFS) using ffmpeg volumedetect.

    For files >1h, samples 60s from middle (1800s offset) to avoid 36min scan
    for 30h books (was timeout 120s killing Wizard). Falls back to whole file.

    Returns mean_volume in dB, or None if undetectable.
    For n/a / -inf (digital silence) returns -91.0 dB (very quiet).
    Higher (less negative) = louder. Used by cleanup pre-flight to skip
    already-quiet inputs.
    """
    # For long files >1h, sample 60s from middle for speed (avoid whole-file scan)
    dur = get_duration(audio_path)
    if dur is not None and dur > 3600:
        # Sample 60s from 30min or middle-30s for representation
        ss = min(1800, max(0, dur / 2 - 30))
        cmd = [
            "ffmpeg", "-ss", str(ss), "-t", "60", "-i", audio_path,
            "-af", "volumedetect", "-vn", "-f", "null", "-",
        ]
        timeout = 60
    else:
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", "volumedetect", "-vn", "-f", "null", "-",
        ]
        timeout = 600  # increased from 120 to handle 30h if we do whole file

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        # Fallback to 60s sample on timeout
        r = subprocess.run(
            ["ffmpeg", "-ss", "1800", "-t", "60", "-i", audio_path,
             "-af", "volumedetect", "-vn", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=60,
        )
    # ffmpeg can output "mean_volume: n/a" or "mean_volume: -inf dB" or numeric.
    # Handle cases with and without dB suffix, scientific notation like -1.2e-05
    m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|-\s*inf|n/a)\s*(?:dB)?", r.stderr, re.IGNORECASE)
    if not m:
        # Could not parse at all - ffprobe failed, no audio stream, etc.
        return None
    raw = m.group(1).strip().lower().replace(" ", "")
    if raw in ("n/a", "-inf"):
        return -91.0
    try:
        return float(raw)
    except ValueError:
        return None


def _find_silence_windows_for_atten(wav_path: str, thresholds: list[int] = [-40, -35, -30, -25], min_dur: float = 0.5) -> list[tuple[float, float]]:
    """Adaptive silence finder for auto atten-lim: try thresholds strict to loose."""
    all_windows = []
    for db in thresholds:
        r = subprocess.run(
            ["ffmpeg", "-i", wav_path, "-af", f"silencedetect=noise={db}dB:d={min_dur}", "-f", "null", "-"],
            capture_output=True, text=True, errors="replace", timeout=120,
        )
        starts = re.findall(r"silence_start:\s*([\d.]+)", r.stderr)
        ends = re.findall(r"silence_end:\s*([\d.]+)", r.stderr)
        for s, e in zip(starts, ends):
            try:
                s_f, e_f = float(s), float(e)
                if e_f - s_f >= min_dur:
                    all_windows.append((s_f, e_f))
            except:
                continue
        if len(all_windows) >= 3:
            break
    all_windows = sorted(set(all_windows), key=lambda x: x[1]-x[0], reverse=True)
    filtered = []
    for s, e in all_windows:
        if not any(abs(s-fs) < 0.1 and abs(e-fe) < 0.1 for fs, fe in filtered):
            overlap = any(not (e <= fs or s >= fe) for fs, fe in filtered)
            if not overlap or len(filtered) < 2:
                filtered.append((s, e))
        if len(filtered) >= 5:
            break
    return filtered


def estimate_optimal_atten_lim(m4b_path: str, margin_db: float = 3.0,
                               target_snr_db: float = 51.0) -> tuple[Optional[float], dict]:
    """Estimate optimal attenuation limit for deep-filter based on SNR of silence vs speech.

    Returns (optimal_atten_db, info_dict) where info contains noise_max, overall, snr, windows.
    Uses max (least quiet) silence window as representative hiss, not average of deepest silence.

    The measured SNR is programme level minus the worst silence-window noise
    floor. The attenuation limit is the shortfall against `target_snr_db` plus
    `margin_db` of headroom, so a NOISIER file (low SNR) gets a LARGER limit and
    a clean file gets a small one. `target_snr_db` is the calibration knob: it is
    the residual SNR the cleanup aims for, not a measured property of the file.
    51 dB is an anchor, not a measurement: a median library book measures ~42 dB
    of gap SNR, and 42 -> 12 dB reproduces the listening-validated default (see
    m4binder-research). It has not itself been validated by listening.

    Clamped to 6-30 dB for audiobooks (6=no NR, 30=keeps room tone, 100=full would be overkill per user test).
    """
    import tempfile
    info = {"file": os.path.basename(m4b_path), "windows": [], "noise_vals": [], "noise_max": None, "overall": None, "snr": None, "optimal": None}

    with tempfile.TemporaryDirectory(prefix="atten_est_") as tmp:
        wav = os.path.join(tmp, "tmp.wav")
        # Sample 60s from 30min offset for speed (was whole file 30h = 2.4h scan)
        # Use get_duration to compute offset, fallback to 1800s
        dur_full = get_duration(m4b_path)
        if dur_full and dur_full > 1800:
            ss_off = min(1800, dur_full / 2)
            r = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ss_off), "-t", "60", "-i", m4b_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                capture_output=True, text=True, errors="replace", timeout=60,
            )
        else:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", m4b_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                capture_output=True, text=True, errors="replace", timeout=120,
            )
        if not os.path.exists(wav):
            return None, info

        windows = _find_silence_windows_for_atten(wav)
        info["windows"] = windows
        if not windows:
            info["optimal"] = 20.0
            return 20.0, info

        noise_vals = []
        for s, e in windows[:3]:
            # atrim, not -ss/-t: an output -t stops the muxer, not the filter, so
            # volumedetect kept measuring a frame or more past the end of the
            # silence window and averaged in the speech that follows it. On a
            # quiet gap that swamps the reading (measured -33 dB where the gap
            # was -88 dB), which flattened every SNR to the same value.
            r = subprocess.run(
                ["ffmpeg", "-i", wav, "-af", f"atrim=start={s}:end={e},volumedetect",
                 "-vn", "-f", "null", "-"],
                capture_output=True, text=True, errors="replace", timeout=30,
            )
            m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)", r.stderr)
            if m:
                try:
                    noise_vals.append(float(m.group(1)))
                except:
                    pass

        if not noise_vals:
            info["optimal"] = 20.0
            return 20.0, info

        info["noise_vals"] = noise_vals
        noise_max = max(noise_vals)
        info["noise_max"] = noise_max

        overall = probe_noise_floor(wav)
        if overall is None:
            r = subprocess.run(
                ["ffmpeg", "-i", wav, "-af", "volumedetect", "-vn", "-f", "null", "-"],
                capture_output=True, text=True, errors="replace", timeout=60,
            )
            m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)", r.stderr)
            if m:
                overall = float(m.group(1))

        info["overall"] = overall
        if overall is None:
            info["optimal"] = 20.0
            return 20.0, info

        snr = overall - noise_max
        info["snr"] = snr
        # Attenuation needed = how far short of the target SNR this file falls.
        # (It used to be `snr + margin_db`, which ranked files backwards: a clean
        # file with a 40 dB SNR asked for the most aggressive setting and a noisy
        # one with a 5 dB SNR asked for the mildest.)
        optimal = max(6.0, min(30.0, round(target_snr_db - snr + margin_db)))
        info["optimal"] = optimal
        return optimal, info


def resolve_atten_lim(spec: str, probe_path: str) -> float:
    """Resolve a CLI atten-lim spec ("12", "auto", ...) to a float dB value.

    Shared by `clean` (probes the source m4b) and `migrate`/`bind --clean`
    (probes the concatenated wav the timeline builds, since there's no
    separate uncleaned m4b when the source is mp3 chapters) — one place for
    the estimate/fallback/print logic instead of two copies drifting apart.

    "auto" estimates from probe_path via estimate_optimal_atten_lim (SNR-based),
    printing what it chose and why, and falls back to 12.0 if estimation
    can't produce a value (e.g. an unreadable probe file). Anything else is
    parsed as a plain float — raises ValueError if it isn't one.
    """
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        optimal, info = estimate_optimal_atten_lim(probe_path)
        if optimal is None:
            print(f"  [auto] {os.path.basename(probe_path)}: atten-lim estimation failed, "
                  f"falling back to 12.0dB")
            return 12.0
        print(f"  [auto] {os.path.basename(probe_path)}: SNR {info.get('snr')}dB "
              f"(noise_max={info.get('noise_max')}, overall={info.get('overall')}) "
              f"-> atten-lim {optimal}dB")
        return optimal
    return float(spec)


def extract_cover(m4b_path: str, max_size: int = 20 * 1024 * 1024) -> Optional[bytes]:
    """Extract attached picture / cover art bytes from an m4b if present, capped at max_size."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="m4b_cover_") as tmp:
        out = os.path.join(tmp, "cover.jpg")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", m4b_path, "-an", "-vcodec", "copy", out],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
        if r.returncode != 0 or not os.path.exists(out):
            return None
        try:
            size = os.path.getsize(out)
            if size == 0 or size > max_size:
                if size > max_size:
                    print(f"  [warn] cover too large ({size} > {max_size}), skipping")
                return None
            # Check magic - JPEG FF D8 or PNG 89 50 4E 47
            with open(out, "rb") as f:
                head = f.read(8)
                if not (head.startswith(b"\xff\xd8") or head.startswith(b"\x89PNG")):
                    print(f"  [warn] cover is neither JPEG nor PNG, skipping")
                    return None
                f.seek(0)
                return f.read()
        except OSError:
            return None


def _escape_ffmetadata_value(s: str) -> str:
    """Escape ffmetadata value per ffmpeg spec: backslash, newline, =, ;, #"""
    # Order matters: escape backslash first
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("=", "\\=")
    s = s.replace(";", "\\;")
    s = s.replace("#", "\\#")
    return s


def create_chapters_ffmetadata(mp3_files: list[str], out_path: str) -> None:
    """Create an ffmetadata file with chapter markers derived from mp3_files.

    Each mp3 is one chapter. Duration from get_duration (ffprobe) on mp3,
    title from EasyID3 'title' tag or fallback 'Chapter N'. Times in ms.
    Skips zero-duration files with warning to avoid invalid overlapping chapters.
    """
    from m4b_lib.metadata import extract_id3_tags

    lines: list[str] = [";FFMETADATA1"]
    current_start_ms = 0
    skipped = 0
    for idx, file_path in enumerate(mp3_files, start=1):
        duration_sec = get_duration(file_path)
        if duration_sec is None or duration_sec <= 0.1:  # consider <100ms or failure as zero/invalid
            print(f"  [warn] skipping zero-duration/invalid file: {file_path} (duration={duration_sec})")
            skipped += 1
            continue
        duration_ms = int(round(duration_sec * 1000))
        chapter_start = current_start_ms
        chapter_end = chapter_start + duration_ms

        tags = extract_id3_tags(file_path)
        mp3_title = tags.get("title") or ""
        if not mp3_title:
            mp3_title = f"Chapter {idx - skipped}" if skipped else f"Chapter {idx}"
        mp3_title = _escape_ffmetadata_value(mp3_title)

        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={chapter_start}")
        lines.append(f"END={chapter_end}")
        lines.append(f"title={mp3_title}")

        current_start_ms = chapter_end

    # Ensure at least one chapter exists — if all skipped, create a dummy 1s chapter to avoid empty file
    if current_start_ms == 0 and skipped == len(mp3_files):
        print(f"  [warn] all {len(mp3_files)} files were zero-duration; creating placeholder chapter")
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append("START=0")
        lines.append("END=1000")
        lines.append("title=Chapter 1")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def probe_chapters(media_path: str) -> list[dict]:
    """Chapters as [{"start": float, "end": float, "title": str}, ...], in order."""
    import json
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_chapters", "-of", "json", media_path],
        capture_output=True, text=True, errors="replace",
    )
    if r.returncode != 0:
        return []
    try:
        raw = json.loads(r.stdout).get("chapters", [])
    except ValueError:
        return []
    out = []
    for c in raw:
        try:
            start, end = float(c.get("start_time", 0)), float(c.get("end_time", 0))
        except (TypeError, ValueError):
            continue
        if end > start:
            out.append({"start": start, "end": end,
                        "title": c.get("tags", {}).get("title", "")})
    return out


def probe_format_tags(path: str) -> dict:
    """Container-level metadata tags, lowercased keys. {} if unreadable.

    Used so `clean` can replay a book's album, album_artist, genre, narrator and
    description into the output instead of discarding them.
    """
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "json", path],
        capture_output=True, text=True, errors="replace",
    )
    try:
        tags = json.loads(r.stdout).get("format", {}).get("tags") or {}
    except (ValueError, AttributeError):
        return {}
    return {str(k).lower(): v for k, v in tags.items()}
