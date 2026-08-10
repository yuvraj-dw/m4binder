"""Parallel AAC encode, concat, and mux.

AAC encoding is the one tail stage that scales well (measured 33x realtime
single, 390x at 24 workers), so streams are encoded concurrently and stitched
with the concat demuxer. The loudness gain rides along here rather than costing
a separate full-length pass.
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from m4b_lib.ffmpeg_utils import _escape_ffmetadata_value, duration_timeout
from m4b_lib.streams import StreamSpec
from m4b_lib.timeline import Chapter


def encode_streams(wav: str, specs: list[StreamSpec], gain_db: float, sample_rate: int,
                   workdir: str, bitrate: str = "64k", workers: int = 24) -> list[str]:
    os.makedirs(workdir, exist_ok=True)
    jobs = []
    for s in specs:
        out = os.path.join(workdir, f"part_{s.index:05d}.m4a")
        ss = s.start_frame / sample_rate
        dur = (s.end_frame - s.start_frame) / sample_rate
        af = f"volume={gain_db:.4f}dB" if abs(gain_db) > 1e-6 else "anull"
        jobs.append((
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{ss:.6f}", "-t", f"{dur:.6f}",
             "-i", wav, "-af", af, "-ac", "1", "-c:a", "aac", "-b:a", bitrate, out],
            out, dur,
        ))

    def _run(job):
        cmd, out, dur = job
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           timeout=duration_timeout(dur))
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"encode failed for {out}: {e.stderr or e}") from e
        return out

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        return list(ex.map(_run, jobs))


def write_chapters_ffmetadata(chapters: list[Chapter], path: str) -> None:
    lines = [";FFMETADATA1"]
    for c in chapters:
        lines += ["[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={int(round(c.start * 1000))}",
                  f"END={int(round(c.end * 1000))}",
                  f"title={_escape_ffmetadata_value(c.title)}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# Tags that describe the container rather than the book. Copying these forward
# would misdescribe the file we just wrote.
_CONTAINER_TAGS = frozenset({
    "major_brand", "minor_version", "compatible_brands", "encoder",
    "creation_time", "handler_name", "vendor_id", "duration",
})


def concat_and_mux(parts: list[str], chapters: list[Chapter], out_path: str,
                   cover_bytes: bytes | None = None, title: str = "",
                   author: str = "", source_tags: dict | None = None) -> None:
    """Mux encoded parts into an m4b, preserving the source's library tags.

    `source_tags` is the source container's format tags. Without it, cleaning a
    book silently discarded everything a library is organised by: measured on a
    real file, 8 tags in and 4 out, all four of them container boilerplate. A
    40-file census found `album` lost on 40/40 books and `album_artist` on 39/40,
    which is how Audiobookshelf groups a library -- and `clean` overwrites in
    place with --keep-original off by default.
    """
    if not parts:
        raise ValueError("concat_and_mux: parts is empty, nothing to mux")
    workdir = os.path.dirname(os.path.abspath(parts[0]))
    final_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(final_dir, exist_ok=True)

    listfile = os.path.join(workdir, "parts.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in parts:
            f.write("file '%s'\n" % p.replace("\\", "\\\\").replace("'", r"'\''"))

    meta = os.path.join(workdir, "chapters.ini")
    write_chapters_ffmetadata(chapters, meta)

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "concat", "-safe", "0", "-i", listfile, "-i", meta]
    cover_path = None
    if cover_bytes:
        cover_path = os.path.join(workdir, "cover.jpg")
        with open(cover_path, "wb") as f:
            f.write(cover_bytes)
        cmd += ["-i", cover_path]

    cmd += ["-map_metadata", "1", "-map", "0:a", "-c:a", "copy"]
    if cover_path:
        cmd += ["-map", "2:v", "-c:v", "mjpeg", "-disposition:v:0", "attached_pic"]
    # Source tags first, so explicit title/author below still win: with repeated
    # -metadata keys ffmpeg takes the last one.
    for key, value in (source_tags or {}).items():
        if key.lower() in _CONTAINER_TAGS or value in (None, ""):
            continue
        cmd += ["-metadata", f"{key}={value}"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if author:
        cmd += ["-metadata", f"artist={author}"]

    tmp_out = out_path + ".tmp.m4b"
    cmd += ["-movflags", "faststart", tmp_out]

    total_dur = sum(c.end - c.start for c in chapters)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=duration_timeout(total_dur))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"mux failed for {out_path}: {e.stderr or e}") from e

    if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) < 1024:
        try:
            os.unlink(tmp_out)
        except OSError:
            pass
        raise RuntimeError(f"mux produced an invalid file for {out_path}")
    os.replace(tmp_out, out_path)
