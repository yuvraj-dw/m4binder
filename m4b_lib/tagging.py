"""Write tags and cover art onto an existing .m4b, without touching the audio.

Deliberately not ffmpeg. Re-muxing a finished audiobook to change its tags
rewrites hundreds of megabytes to alter a few hundred bytes, and ffmpeg's mp4
muxer forces a choice: `-movflags use_metadata_tags` is required for freeform
atoms (series, isbn) but silently drops cover art, while omitting it keeps the
cover and discards every non-standard key. mutagen edits the atoms in place,
so both survive, the audio is byte-identical by construction rather than by
verification, and a 477MB book is tagged in milliseconds.

Callers pass generic tag names ("title", "series", "series-part"); the mapping
to MP4 atoms lives here so they do not have to know about `\xa9nam` or
`----:com.apple.iTunes:`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from mutagen.mp4 import MP4, MP4Cover

# Generic name -> standard MP4 atom. Anything not listed is written as an
# iTunes freeform atom, which is how arbitrary keys survive at all.
_ATOMS = {
    "title": "\xa9nam",
    "artist": "\xa9ART",
    "album_artist": "aART",
    "album": "\xa9alb",
    "composer": "\xa9wrt",          # audiobook players read the narrator here
    "date": "\xa9day",
    "genre": "\xa9gen",
    "comment": "\xa9cmt",
    "description": "desc",
    "lyrics": "\xa9lyr",
    "encoder": "\xa9too",
    "copyright": "cprt",
    "grouping": "\xa9grp",
}

# Integer-valued atoms. `movement` is how iTunes-style players carry a
# position within a series, so a series index goes here as well as into a
# freeform atom -- different readers look in different places.
_INT_ATOMS = {"series-part": "\xa9mvi"}

# Written to a standard atom *and* a freeform one, because support varies.
_ALSO_FREEFORM = {"series": "\xa9mvn"}

_FREEFORM_PREFIX = "----:com.apple.iTunes:"


class TaggingFailed(Exception):
    """The file could not be tagged.

    It may be PARTIALLY REWRITTEN: MP4.save() edits atoms in place and,
    when a cover has to be inserted ahead of a faststart mdat, shifts the
    entire payload. There is no temp file and no rollback. Callers must not
    assume the file is unchanged.
    """


def _freeform(name: str) -> str:
    return f"{_FREEFORM_PREFIX}{name.upper().replace('-', '_')}"


def write_m4b_tags(
    path: Path | str,
    tags: dict,
    cover: Optional[bytes] = None,
    cover_format: str = "jpeg",
) -> Path:
    """Apply `tags` (and optionally `cover`) to the .m4b at `path`.

    Empty values are skipped rather than written blank: an empty tag looks
    authoritative and is worse than an absent one.
    """
    path = Path(path)
    try:
        audio = MP4(path)
    except Exception as exc:                       # noqa: BLE001
        raise TaggingFailed(f"{path.name}: not a readable MP4 ({exc})") from exc

    for key, value in (tags or {}).items():
        if value in (None, "", []):
            continue
        if key in _INT_ATOMS:
            try:
                audio[_INT_ATOMS[key]] = [int(str(value).strip())]
            except (TypeError, ValueError):
                pass                               # non-numeric index: freeform only
            audio[_freeform(key)] = str(value).encode()
        elif key in _ALSO_FREEFORM:
            audio[_ALSO_FREEFORM[key]] = [str(value)]
            audio[_freeform(key)] = str(value).encode()
        elif key in _ATOMS:
            audio[_ATOMS[key]] = [str(value)]
        else:
            audio[_freeform(key)] = str(value).encode()

    if cover:
        fmt = (
            MP4Cover.FORMAT_PNG
            if cover_format.lower() == "png"
            else MP4Cover.FORMAT_JPEG
        )
        audio["covr"] = [MP4Cover(cover, imageformat=fmt)]

    try:
        audio.save()
    except Exception as exc:                       # noqa: BLE001
        raise TaggingFailed(f"{path.name}: could not save tags ({exc})") from exc
    return path


def read_m4b_tags(path: Path | str) -> dict:
    """Current tags, keyed by the same generic names `write_m4b_tags` takes."""
    audio = MP4(Path(path))
    reverse = {v: k for k, v in _ATOMS.items()}
    reverse.update({v: k for k, v in _INT_ATOMS.items()})
    reverse.update({v: k for k, v in _ALSO_FREEFORM.items()})
    out: dict = {}
    for atom, value in (audio.tags or {}).items():
        if atom == "covr":
            out["cover_bytes"] = len(bytes(value[0])) if value else 0
            continue
        if atom.startswith(_FREEFORM_PREFIX):
            name = atom[len(_FREEFORM_PREFIX):].lower().replace("_", "-")
            out.setdefault(name, value[0].decode("utf-8", "replace")
                           if isinstance(value[0], bytes) else str(value[0]))
            continue
        name = reverse.get(atom, atom)
        # Not every MP4 atom is a list. `cpil` (compilation) and `pgap` are
        # plain bools, and `trkn`/`disk` are tuples, so indexing [0]
        # unconditionally raises "'bool' object is not subscriptable" on any
        # file that carries one. Every Libation rip does: this made 90% of a
        # 1,800-book library look unreadable, and the failure LOOKED like
        # corrupt files rather than a reader bug.
        if isinstance(value, list):
            out[name] = value[0] if value else None
        else:
            out[name] = value
    return out
