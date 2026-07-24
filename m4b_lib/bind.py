"""mp3 chapters -> m4b assembly. Public API: bind_single, bind_multiple."""
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

from m4b_lib import ffmpeg_utils, metadata


@dataclass
class BindOptions:
    input_folder: str
    output_file: Optional[str] = None       # single mode
    output_folder: Optional[str] = None     # multiple mode
    metadata_source: str = "none"
    title: str = ""
    author: str = ""
    bitrate: str = "64k"
    overwrite: bool = False


import re as _re

def _natural_key(s: str):
    """Natural sort key: splits digits and text for human ordering."""
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r'(\d+)', s)]

def _list_mp3s(folder: str) -> list[str]:
    return sorted(
        (os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp3")),
        key=lambda p: _natural_key(os.path.basename(p)),
    )


def _bind_one(input_folder: str, output_m4b: str, opts: BindOptions) -> None:
    """Convert one folder of mp3s -> m4b with chapter markers."""
    mp3s = _list_mp3s(input_folder)
    if not mp3s:
        raise ValueError(f"No mp3 files in {input_folder}")

    meta = metadata.resolve_metadata(
        opts.metadata_source, opts.title, opts.author, first_mp3=mp3s[0]
    )

    with tempfile.TemporaryDirectory(prefix="m4binder_") as tmp:
        m4a_dir = os.path.join(tmp, "m4a")
        m4a_files = ffmpeg_utils.parallel_transcode(mp3s, m4a_dir, bitrate=opts.bitrate)
        intermediate = os.path.join(tmp, "combined.m4a")
        ffmpeg_utils.concat_audio_to_m4b(m4a_files, intermediate)

        # Generate chapters ffmetadata from mp3 durations + ID3 titles
        chapters_ini = os.path.join(tmp, "chapters.ffmetadata")
        ffmpeg_utils.create_chapters_ffmetadata(mp3s, chapters_ini)

        # Wrap into final m4b with metadata and chapters - copy audio to avoid double lossy encode
        ffmpeg_utils.embed_chapters_and_meta(
            intermediate,
            chapters_ini=chapters_ini,
            cover_bytes=meta.cover_bytes,
            output_m4b=output_m4b,
            title=meta.title,
            author=meta.author,
            bitrate=opts.bitrate,
            tmpdir=tmp,
            copy_audio=True,
        )
    print(f"Created audiobook: {output_m4b}")


def bind_single(opts: BindOptions) -> None:
    if not opts.output_file:
        raise ValueError("bind_single requires output_file")
    _bind_one(opts.input_folder, opts.output_file, opts)


def bind_multiple(opts: BindOptions) -> None:
    out_dir = opts.output_folder or opts.input_folder
    os.makedirs(out_dir, exist_ok=True)
    for entry in sorted(os.listdir(opts.input_folder)):
        sub = os.path.join(opts.input_folder, entry)
        if not os.path.isdir(sub):
            continue
        # Only descend into dirs that actually contain mp3s
        if not any(f.lower().endswith(".mp3") for f in os.listdir(sub)):
            print(f"[WARN] No MP3 files in subfolder: {sub}. Skipping.")
            continue
        out = os.path.join(out_dir, entry + ".m4b")
        # H3 fix: guard against silently overwriting existing final m4b, unless --overwrite
        if os.path.exists(out) and not opts.overwrite:
            print(f"[WARN] Output {out} already exists, skipping (use --overwrite to allow)")
            continue
        print(f"[INFO] Converting subfolder: {sub}")
        try:
            _bind_one(sub, out, opts)
        except Exception as e:
            print(f"[ERROR] Failed on {sub}: {e}")
        else:
            print(f"[INFO] Finished subfolder -> {out}")
