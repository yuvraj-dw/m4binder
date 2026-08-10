"""One decoded representation for both entry paths.

`clean` starts from an m4b, `migrate` and `bind --clean` start from a folder of
mp3s. Both produce a Timeline: a single mono PCM wav plus a chapter map. Nothing
downstream needs to know which it was.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import soundfile as sf

from m4b_lib import ffmpeg_utils


@dataclass(frozen=True)
class Chapter:
    start: float
    end: float
    title: str


@dataclass
class Timeline:
    wav_path: str
    sample_rate: int
    frames: int
    chapters: list[Chapter] = field(default_factory=list)
    cover_bytes: bytes | None = None
    title: str = ""
    author: str = ""
    # The source container's format tags, carried so the mux can replay them.
    # Without this a clean discards album, album_artist, genre, narrator and the
    # rest -- everything a library is organised by.
    source_tags: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def _decode(src: str, dst: str, sample_rate: int) -> int:
    """Decode any container to mono PCM at sample_rate. Returns decoded frames.

    Written as Wave64, not WAV. RIFF stores its sizes in unsigned 32-bit fields,
    so a WAV cannot exceed 4 GiB -- which at mono 16-bit 48 kHz is exactly
    44739.24 s, or 12.43 hours. Books past that were silently truncated at that
    boundary: two different sources, 12.7 h and 24.2 h, both produced output of
    exactly 44739.3 s. Wave64 uses 64-bit sizes and libsndfile reads it with the
    same read/seek/blocks calls, so nothing downstream changes.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src,
             "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
             "-f", "w64", dst],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"decode failed for {src}: {e.stderr or e}") from e
    return sf.info(dst).frames


def from_m4b(m4b_path: str, workdir: str, sample_rate: int = 48000) -> Timeline:
    os.makedirs(workdir, exist_ok=True)
    wav = os.path.join(workdir, "timeline.wav")
    frames = _decode(m4b_path, wav, sample_rate)

    chapters = [
        Chapter(start=c["start"], end=c["end"], title=c.get("title", ""))
        for c in ffmpeg_utils.probe_chapters(m4b_path)
    ]
    if not chapters:
        chapters = [Chapter(0.0, frames / sample_rate, "Chapter 1")]

    return Timeline(
        wav_path=wav, sample_rate=sample_rate, frames=frames, chapters=chapters,
        cover_bytes=ffmpeg_utils.extract_cover(m4b_path),
        source_tags=ffmpeg_utils.probe_format_tags(m4b_path),
    )


def from_mp3s(mp3_paths: list[str], workdir: str, sample_rate: int = 48000,
              jobs: int = 4, title: str = "", author: str = "",
              cover_bytes: bytes | None = None) -> Timeline:
    """Decode mp3s in parallel and concatenate; marks come from decoded frames.

    Chapter marks are derived from actual decoded sample counts rather than
    ffprobe durations, because mp3 encoder delay/padding makes the two differ
    per file and the error accumulates across chapters.
    """
    from m4b_lib.metadata import extract_id3_tags

    if not mp3_paths:
        raise ValueError("from_mp3s: mp3_paths is empty, nothing to build a Timeline from")

    os.makedirs(workdir, exist_ok=True)
    parts_dir = os.path.join(workdir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    jobs = max(1, min(jobs, len(mp3_paths)))
    parts = [os.path.join(parts_dir, f"{i:05d}.wav") for i in range(len(mp3_paths))]

    def _one(pair):
        src, dst = pair
        return _decode(src, dst, sample_rate)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        counts = list(ex.map(_one, zip(mp3_paths, parts)))

    chapters, cursor = [], 0
    for idx, (src, n) in enumerate(zip(mp3_paths, counts), start=1):
        if n <= 0:
            continue
        tags = extract_id3_tags(src)
        chapters.append(Chapter(
            start=cursor / sample_rate,
            end=(cursor + n) / sample_rate,
            title=tags.get("title") or f"Chapter {idx}",
        ))
        cursor += n

    wav = os.path.join(workdir, "timeline.wav")
    listfile = os.path.join(workdir, "parts.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % p.replace("\\", "\\\\").replace("'", r"'\''"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", listfile, "-c", "copy", "-f", "w64", wav],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"concat failed: {e.stderr or e}") from e

    frames = sf.info(wav).frames
    if frames != cursor:
        raise RuntimeError(f"concat produced {frames} frames, expected {cursor}")

    if not chapters:
        print(f"  [warn] all {len(mp3_paths)} files were zero-duration; creating placeholder chapter")
        chapters = [Chapter(0.0, frames / sample_rate, "Chapter 1")]

    return Timeline(wav_path=wav, sample_rate=sample_rate, frames=frames,
                    chapters=chapters, cover_bytes=cover_bytes,
                    title=title, author=author)
