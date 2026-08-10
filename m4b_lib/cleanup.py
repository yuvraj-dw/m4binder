"""Audiobook cleanup orchestration.

clean_one and migrate both build a Timeline and hand it to clean_timeline; the
only difference is where the audio came from.
"""
import glob
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

# `<name>.orig.m4b`, plus the `.orig.1.m4b` form clean_one uses when a backup
# already exists. Anchored to the extension so a book actually called
# "Original Sin.m4b" is not mistaken for one.
_BACKUP_RE = re.compile(r"\.orig(\.\d+)?\.m4b$", re.IGNORECASE)


def is_backup(path: str) -> bool:
    """True for the `.orig.m4b` files `clean_one --keep-original` writes.

    A directory sweep must skip these: re-cleaning a backup puts a second lossy
    generation on the one file whose purpose is to be pristine, and the user
    finds out only when they go looking for the original.
    """
    return bool(_BACKUP_RE.search(os.path.basename(path)))


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
        # pathlib's ** does not follow symlinked directories, so recursion is
        # bounded without needing an explicit visited-inode set.
        yielded = 0
        try:
            if "**" in pattern:
                # rglob ignores pattern prefix, so we need to handle **/*.m4b -> *.m4b recursive
                # Extract suffix after **
                suffix = pattern.split("**")[-1].lstrip("/\\")
                if not suffix:
                    suffix = "*.m4b"
                for fp_path in sorted(p.rglob(suffix)):
                    # pathlib's ** does not follow symlinked directories, so recursion is
                    # bounded without needing an explicit visited-inode set.
                    try:
                        if fp_path.is_symlink() and fp_path.is_dir():
                            continue
                    except OSError:
                        continue
                    fp = str(fp_path)
                    if is_backup(fp):
                        continue          # never re-clean our own backup
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
                    if is_backup(fp):
                        continue          # never re-clean our own backup
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


def _workers(overrides: dict | None, stage: str) -> int:
    from m4b_lib import scheduler
    if overrides and stage in overrides:
        return max(1, int(overrides[stage]))
    return scheduler.default_workers(stage)


def _validate_output(staged_path: str, expected_duration: float) -> None:
    """Raise if staged_path is missing/too small, or its duration drifts too far.

    clean_timeline can raise outright on failure, but it can also succeed while
    producing a plausible-but-wrong file — truncated, or silently short by a
    chapter. Nothing downstream catches that, and the caller's os.replace would
    happily overwrite the user's only copy with it, so this is the last check
    before that happens.
    """
    from m4b_lib import ffmpeg_utils

    if not os.path.exists(staged_path) or os.path.getsize(staged_path) < 1024:
        size = os.path.getsize(staged_path) if os.path.exists(staged_path) else "missing"
        raise RuntimeError(f"cleaning produced invalid output: {staged_path} size={size}")

    staged_duration = ffmpeg_utils.get_duration(staged_path)
    if staged_duration is None:
        raise RuntimeError(f"cleaning produced an unreadable output: {staged_path}")

    tolerance = max(0.5, expected_duration * 0.01)
    diff = abs(staged_duration - expected_duration)
    if diff > tolerance:
        raise RuntimeError(
            f"cleaning produced a duration mismatch for {staged_path}: "
            f"expected {expected_duration:.1f}s, got {staged_duration:.1f}s "
            f"(diff {diff:.1f}s > tolerance {tolerance:.1f}s)"
        )


def clean_timeline(tl, out_path: str, *, atten_lim_db: float = 12.0,
                   pf: bool = False, backend: str = "df3", device: str = "cpu",
                   workers: dict | None = None, bitrate: str = "64k",
                   on_stage: Optional[Callable[[str, float], None]] = None) -> None:
    """Run enhance -> loudness -> encode+mux and write out_path.

    `on_stage(name, seconds)`, if given, is called after each of the three
    stages with its wall-clock duration. This exists so a measurement tool
    can report real per-stage timing without re-implementing this
    orchestration itself — a duplicate of this sequence in a second place
    is exactly what drifts (m4binder-research's measure_pipeline.py used to
    run loudness serially while this function always ran it in a
    ThreadPoolExecutor, silently measuring a slower pipeline than the one
    that ships).
    """
    # Imported here, not at module scope: these pull in soundfile/numpy, and
    # iter_targets above must stay importable without the ml extra installed.
    from m4b_lib import encode, loudness, scheduler
    from m4b_lib.enhance import EnhanceConfig
    from m4b_lib.enhance import df3 as _df3  # noqa: F401  (registers "df3")
    from m4b_lib.streams import plan_streams

    workdir = os.path.dirname(os.path.abspath(tl.wav_path))
    cfg = EnhanceConfig(atten_lim_db=atten_lim_db, pf=pf)

    t0 = time.time()
    enhanced = scheduler.enhance_timeline(
        tl, os.path.join(workdir, "enhanced.wav"), backend=backend, cfg=cfg,
        device=device, workers=_workers(workers, "df3"),
    )
    if on_stage:
        on_stage("enhance", time.time() - t0)

    t0 = time.time()
    specs = plan_streams(tl.frames, _workers(workers, "loudness"))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        measured = list(ex.map(
            lambda s: loudness.analyze(enhanced, s.start_frame, s.end_frame, tl.sample_rate),
            specs,
        ))
    gain = loudness.gain_db(loudness.combine(measured))
    if on_stage:
        on_stage("loudness", time.time() - t0)

    t0 = time.time()
    enc_specs = plan_streams(tl.frames, _workers(workers, "encode"))
    parts = encode.encode_streams(
        enhanced, enc_specs, gain_db=gain, sample_rate=tl.sample_rate,
        workdir=os.path.join(workdir, "parts_enc"), bitrate=bitrate,
        workers=_workers(workers, "encode"),
    )
    encode.concat_and_mux(parts, tl.chapters, out_path,
                          cover_bytes=tl.cover_bytes, title=tl.title,
                          author=tl.author, source_tags=tl.source_tags)
    if on_stage:
        on_stage("encode+mux", time.time() - t0)


def clean_one(m4b_path: str, *, keep_original: bool = True,
              atten_lim_db: float = 12.0, pf: bool = False, backend: str = "df3",
              device: str = "cpu", workers: dict | None = None,
              bitrate: str = "64k") -> None:
    """Clean `m4b_path` in place, keeping `<name>.orig.m4b` unless told not to.

    **`keep_original` defaults to True as of 2026-08-10, reversing the previous
    default.** This function overwrites the user's file with model output, and
    there is a known class of book where that output is worse: DeepFilterNet3
    treats a music bed as noise. In the round-9 blind test the untouched file
    took first place on the one book in the set with music, ahead of every
    processed variant, and the damage is silent — the book still plays.

    A backup is cheap and reversible; an over-suppressed music bed is neither.
    Callers that genuinely want the old behaviour pass `keep_original=False`,
    which is now an explicit choice rather than the path of least resistance.
    """
    from m4b_lib import ffmpeg_utils, timeline

    source_duration = ffmpeg_utils.get_duration(m4b_path)

    parent = os.path.dirname(os.path.abspath(m4b_path))
    with tempfile.TemporaryDirectory(prefix="m4b_clean_", dir=parent) as tmp:
        tl = timeline.from_m4b(m4b_path, tmp)
        staged = os.path.join(tmp, "cleaned.m4b")
        clean_timeline(tl, staged, atten_lim_db=atten_lim_db, pf=pf, backend=backend,
                       device=device, workers=workers, bitrate=bitrate)
        _validate_output(staged, source_duration if source_duration is not None else tl.duration)

        if keep_original:
            backup = os.path.splitext(m4b_path)[0] + ".orig.m4b"
            base, ext = os.path.splitext(backup)
            i = 1
            while os.path.exists(backup):
                backup = f"{base}.{i}{ext}"
                i += 1
            shutil.copy2(m4b_path, backup)
        os.replace(staged, m4b_path)
