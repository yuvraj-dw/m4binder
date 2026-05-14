"""Metadata helpers: ID3 tag reads, OpenLibrary lookups, cover art fetching."""
import io
from dataclasses import dataclass, field
from typing import Optional

import requests
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC
from mutagen.mp3 import MP3

try:
    from olclient.openlibrary import OpenLibrary
    import olclient.common as ol_common
    _OPENLIB_AVAILABLE = True
except ImportError:
    OpenLibrary = None
    _OPENLIB_AVAILABLE = False


@dataclass
class BookMetadata:
    title: str = ""
    author: str = ""
    cover_bytes: Optional[bytes] = None
    extra: dict = field(default_factory=dict)


def extract_id3_tags(mp3_path: str) -> dict:
    """Return a dict with 'title', 'artist', 'album' from an MP3 file's ID3 tags."""
    try:
        tags = EasyID3(mp3_path)
        return {
            "title": (tags.get("title") or [""])[0],
            "artist": (tags.get("artist") or [""])[0],
            "album": (tags.get("album") or [""])[0],
        }
    except Exception:
        return {"title": "", "artist": "", "album": ""}


def extract_embedded_cover(mp3_path: str) -> Optional[bytes]:
    """Return APIC cover image bytes from an MP3, or None if absent."""
    try:
        audio = ID3(mp3_path)
        for tag in audio.values():
            if isinstance(tag, APIC):
                return tag.data
    except Exception:
        pass
    return None


def lookup_openlibrary(title: str, author: str) -> BookMetadata:
    """Query Open Library for a matching book. Returns BookMetadata; fields may be empty."""
    if not _OPENLIB_AVAILABLE or not (title or author):
        return BookMetadata(title=title, author=author)
    try:
        ol = OpenLibrary()
        results = ol.Work.search(title=title, author=author, limit=1)
        if not results:
            return BookMetadata(title=title, author=author)
        work = results[0]
        cover = None
        cover_id = getattr(work, "cover_id", None)
        if cover_id:
            r = requests.get(
                f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg", timeout=15
            )
            if r.ok:
                cover = r.content
        return BookMetadata(
            title=getattr(work, "title", title) or title,
            author=author,
            cover_bytes=cover,
        )
    except Exception:
        return BookMetadata(title=title, author=author)


def resolve_metadata(source: str, title: str, author: str, first_mp3: Optional[str]) -> BookMetadata:
    """Top-level dispatcher matching original m4binder behavior.

    source: 'openlibrary' | 'google' | 'none'.
    If title/author missing, fall back to ID3 tags of first_mp3.
    """
    if first_mp3 and (not title or not author):
        tags = extract_id3_tags(first_mp3)
        title = title or tags["title"]
        author = author or tags["artist"]
    if source == "openlibrary":
        meta = lookup_openlibrary(title, author)
    else:
        meta = BookMetadata(title=title, author=author)
    if not meta.cover_bytes and first_mp3:
        meta.cover_bytes = extract_embedded_cover(first_mp3)
    return meta
