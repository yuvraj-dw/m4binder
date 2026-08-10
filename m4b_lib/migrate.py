"""One-step migrate: mp3 chapters → cleaned m4b in single lossy encode.

Two architectures:
- A) Simple (bind then clean) — double lossy, reuses existing code, 20 LOC
- B) Optimal single lossy (implemented here) — mp3 decode → wav concat → clean wav → m4b encode once

B is better quality: cleaner sees raw tape hiss not AAC artifacts, single AAC encode.

For v2.0 single best set: ml-rust 12dB is default, no fallback.
"""

import os
from typing import List

from m4b_lib.bind import BindOptions, _natural_key


def _list_mp3s(folder: str) -> List[str]:
    return sorted(
        (os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp3")),
        key=lambda p: _natural_key(os.path.basename(p)),
    )


def _bind_and_clean_one(input_folder: str, output_m4b: str, bind_opts: BindOptions,
                        clean_atten_lim: str = "12", clean_pf: bool = False,
                        keep_original: bool = False) -> None:
    import os
    import shutil
    import tempfile

    from m4b_lib import ffmpeg_utils, metadata, timeline
    from m4b_lib.cleanup import _validate_output, clean_timeline

    mp3s = _list_mp3s(input_folder)
    if not mp3s:
        raise ValueError(f"No mp3 files in {input_folder}")
    meta = metadata.resolve_metadata(bind_opts.metadata_source, bind_opts.title,
                                     bind_opts.author, first_mp3=mp3s[0])

    parent = os.path.dirname(os.path.abspath(output_m4b))
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m4b_migrate_", dir=parent) as tmp:
        tl = timeline.from_mp3s(mp3s, tmp, title=meta.title, author=meta.author,
                                cover_bytes=meta.cover_bytes)
        staged = os.path.join(tmp, "cleaned.m4b")
        # No separate uncleaned m4b to probe here — mp3s are the source, so
        # "auto" estimates SNR from the concatenated wav the timeline just
        # built (same estimator `clean --atten-lim auto` uses on a source m4b).
        atten = ffmpeg_utils.resolve_atten_lim(clean_atten_lim, tl.wav_path)
        clean_timeline(tl, staged, atten_lim_db=atten, pf=clean_pf,
                       bitrate=bind_opts.bitrate)
        # No source m4b to probe here — mp3s are the source, so tl.duration
        # (from the decoded/concatenated wav) is the correct reference.
        _validate_output(staged, tl.duration)

        if keep_original and os.path.exists(output_m4b):
            backup = os.path.splitext(output_m4b)[0] + ".orig.m4b"
            base, ext = os.path.splitext(backup)
            i = 1
            while os.path.exists(backup):
                backup = f"{base}.{i}{ext}"
                i += 1
            shutil.copy2(output_m4b, backup)
        os.replace(staged, output_m4b)
    print(f"Created cleaned audiobook: {output_m4b}")


def bind_and_clean_single(bind_opts: BindOptions, clean_atten_lim: str = "12", clean_pf: bool = False, keep_original: bool = False) -> None:
    if not bind_opts.output_file:
        raise ValueError("bind_and_clean_single requires output_file")
    _bind_and_clean_one(bind_opts.input_folder, bind_opts.output_file, bind_opts, clean_atten_lim, clean_pf, keep_original)


def bind_and_clean_multiple(bind_opts: BindOptions, clean_atten_lim: str = "12", clean_pf: bool = False, keep_original: bool = False) -> None:
    out_dir = bind_opts.output_folder or bind_opts.input_folder
    os.makedirs(out_dir, exist_ok=True)
    for entry in sorted(os.listdir(bind_opts.input_folder)):
        sub = os.path.join(bind_opts.input_folder, entry)
        if not os.path.isdir(sub):
            continue
        if not any(f.lower().endswith(".mp3") for f in os.listdir(sub)):
            print(f"[WARN] No MP3 files in subfolder: {sub}. Skipping.")
            continue
        out = os.path.join(out_dir, entry + ".m4b")
        if os.path.exists(out) and not bind_opts.overwrite:
            print(f"[WARN] Output {out} exists, skipping (use --overwrite)")
            continue
        print(f"[INFO] Migrating (bind+clean) subfolder: {sub} -> {out} (atten-lim {clean_atten_lim}dB)")
        try:
            _bind_and_clean_one(sub, out, bind_opts, clean_atten_lim, clean_pf, keep_original)
        except Exception as e:
            print(f"[ERROR] Failed on {sub}: {e}")
        else:
            print(f"[INFO] Finished {out}")
