"""`clean` must not strip the tags a library is organised by.

Measured on a real book before this was fixed: the source carried 8 tags and the
cleaned output carried 4, all of them container boilerplate. A 40-file census
found album lost on 40/40 books, album_artist on 39/40, genre 38/40, date 38/40,
composer (the narrator) 33/40, description 32/40.

album and album_artist are how Audiobookshelf groups a library, and `clean`
overwrites in place with --keep-original off by default, so the loss is
irreversible on a library it has been pointed at.
"""
import json
import shutil
import subprocess

import pytest

from m4b_lib import encode
from m4b_lib.encode import Chapter

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not on PATH")

SOURCE_TAGS = {
    "album": "The Expanse",
    "album_artist": "James S. A. Corey",
    "genre": "Science Fiction",
    "date": "2011",
    "composer": "Jefferson Mays",
    "comment": "Book 1 of 9",
    "copyright": "(c) 2011 Orbit",
    "description": "A blurb about the book.",
    "track": "1",
}


def _tags(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags",
                        "-of", "json", path], capture_output=True, text=True)
    return {k.lower(): v for k, v in
            (json.loads(r.stdout).get("format", {}).get("tags") or {}).items()}


@pytest.fixture
def part(tmp_path):
    """One short encoded part, as the encode stage would produce."""
    p = str(tmp_path / "part0.m4a")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=300:duration=2", "-ac", "1",
                    "-c:a", "aac", "-b:a", "64k", p], check=True)
    return p


def test_source_tags_are_carried_into_the_output(part, tmp_path):
    out = str(tmp_path / "out.m4b")
    encode.concat_and_mux([part], [Chapter(0.0, 2.0, "Ch 1")], out,
                          title="The Expanse", author="James S. A. Corey",
                          source_tags=SOURCE_TAGS)
    got = _tags(out)
    missing = [k for k in SOURCE_TAGS if k not in got]
    assert not missing, f"lost tags: {missing}"
    assert got["album"] == "The Expanse"
    assert got["composer"] == "Jefferson Mays"


def test_title_and_author_still_win_over_source_tags(part, tmp_path):
    """Explicit values are the caller's intent and must not be overwritten by a
    stale tag from the container."""
    out = str(tmp_path / "out.m4b")
    encode.concat_and_mux([part], [Chapter(0.0, 2.0, "Ch 1")], out,
                          title="Corrected Title", author="Corrected Author",
                          source_tags={**SOURCE_TAGS, "title": "Old Title",
                                       "artist": "Old Author"})
    got = _tags(out)
    assert got["title"] == "Corrected Title"
    assert got["artist"] == "Corrected Author"


def test_container_boilerplate_is_not_copied(part, tmp_path):
    """major_brand and friends describe the old container, not the book."""
    out = str(tmp_path / "out.m4b")
    encode.concat_and_mux([part], [Chapter(0.0, 2.0, "Ch 1")], out,
                          source_tags={"album": "A", "major_brand": "ZZZZ",
                                       "encoder": "some old encoder v1"})
    got = _tags(out)
    assert got["album"] == "A"
    # ffmpeg writes its own major_brand and encoder for the container it just
    # produced, so the check is that the SOURCE values did not survive, using
    # values ffmpeg would never generate itself.
    assert got.get("major_brand") != "ZZZZ"
    assert got.get("encoder") != "some old encoder v1"


def test_no_source_tags_still_works(part, tmp_path):
    out = str(tmp_path / "out.m4b")
    encode.concat_and_mux([part], [Chapter(0.0, 2.0, "Ch 1")], out, title="T")
    assert _tags(out)["title"] == "T"
