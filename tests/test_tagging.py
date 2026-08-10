"""Tests for in-place m4b tagging."""
import subprocess

import pytest

from m4b_lib.tagging import TaggingFailed, read_m4b_tags, write_m4b_tags


@pytest.fixture
def m4b(tmp_path):
    out = tmp_path / "t.m4b"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=1", "-c:a", "libmp3lame", "-b:a", "64k",
         "-f", "mp4", "-brand", "M4A ", str(out)],
        check=True, capture_output=True)
    return out


def _jpeg(tmp_path):
    p = tmp_path / "c.jpg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         "color=c=red:s=64x64", "-frames:v", "1", str(p)],
        check=True, capture_output=True)
    return p.read_bytes()


def test_standard_tags_round_trip(m4b):
    write_m4b_tags(m4b, {"title": "T", "artist": "A", "album": "Alb",
                         "composer": "Narrator", "genre": "G", "date": "2026-01-01"})
    got = read_m4b_tags(m4b)
    assert got["title"] == "T"
    assert got["composer"] == "Narrator"
    assert got["album"] == "Alb"


def test_unknown_keys_become_freeform_atoms(m4b):
    """Without freeform atoms, series/isbn/publisher are simply lost."""
    write_m4b_tags(m4b, {"isbn": "9781464284410", "publisher": "Books on Tape"})
    got = read_m4b_tags(m4b)
    assert got["isbn"] == "9781464284410"
    assert got["publisher"] == "Books on Tape"


def test_series_is_written_to_BOTH_the_movement_atom_and_a_freeform_atom(m4b):
    """read_m4b_tags maps both back to the same key, so asserting through it
    cannot tell which one was written. Readers disagree about where a series
    lives; check the raw atoms."""
    from mutagen.mp4 import MP4
    write_m4b_tags(m4b, {"series": "Dragon Girl", "series-part": "2"})
    raw = MP4(m4b)
    assert raw["\xa9mvn"] == ["Dragon Girl"]
    assert raw["\xa9mvi"] == [2]
    assert raw["----:com.apple.iTunes:SERIES"][0] == b"Dragon Girl"


def test_a_non_numeric_series_part_still_survives(m4b):
    """Half-books exist ("1.5"); the integer atom must not eat them."""
    write_m4b_tags(m4b, {"series-part": "1.5"})
    assert read_m4b_tags(m4b)["series-part"] == "1.5"


def test_cover_art_and_freeform_atoms_coexist(m4b, tmp_path):
    """The whole reason this is not ffmpeg: -movflags use_metadata_tags keeps
    freeform atoms but silently drops the cover, and omitting it does the
    reverse."""
    write_m4b_tags(m4b, {"series": "S", "isbn": "123"}, cover=_jpeg(tmp_path))
    got = read_m4b_tags(m4b)
    assert got["series"] == "S" and got["isbn"] == "123"
    assert got["cover_bytes"] > 0
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(m4b)], capture_output=True, text=True).stdout
    assert "video" in streams


def test_empty_values_are_skipped_not_written_blank(m4b):
    write_m4b_tags(m4b, {"title": "T", "artist": "", "genre": None, "album": []})
    got = read_m4b_tags(m4b)
    assert got["title"] == "T"
    assert not got.get("artist")
    assert "genre" not in got


def test_the_audio_is_byte_identical_not_merely_equivalent(m4b):
    """mutagen edits atoms in place -- the audio payload is untouched, which
    is stronger than a decoded-PCM match after a remux."""
    def audio_bytes(path):
        # -map_metadata -1 is essential: without it ffmpeg copies the
        # container's tags into the extracted mp3's ID3 header, so the tags
        # under test appear in the "audio" and the comparison measures
        # itself. The audio frames are what must not change.
        return subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a",
             "-map_metadata", "-1", "-c", "copy", "-f", "mp3", "-"],
            capture_output=True).stdout
    before = audio_bytes(m4b)
    write_m4b_tags(m4b, {"title": "T"})
    assert audio_bytes(m4b) == before


def test_tagging_a_non_mp4_fails_cleanly(tmp_path):
    junk = tmp_path / "x.m4b"
    junk.write_bytes(b"not an mp4")
    with pytest.raises(TaggingFailed):
        write_m4b_tags(junk, {"title": "T"})
    assert junk.read_bytes() == b"not an mp4"


def test_a_failed_save_is_reported_not_swallowed(m4b, monkeypatch):
    """P8 survived: a swallowed save() failure is exactly the Critical in
    libby -- the caller then treats a half-rewritten file as fine."""
    from mutagen.mp4 import MP4
    def boom(self, *a, **k):
        raise OSError("no space left on device")
    monkeypatch.setattr(MP4, "save", boom)
    with pytest.raises(TaggingFailed):
        write_m4b_tags(m4b, {"title": "T"})


def test_the_freeform_namespace_is_the_one_players_read(m4b):
    """P5 survived: changing the namespace to one no player reads left every
    test green while the tags became invisible."""
    from mutagen.mp4 import MP4
    write_m4b_tags(m4b, {"isbn": "123"})
    assert any(k.startswith("----:com.apple.iTunes:") for k in MP4(m4b).keys())


def test_composer_is_written_to_the_narrator_atom(m4b):
    """P9 survived: composer routed to a different atom, but read_m4b_tags
    shares the same _ATOMS mapping with write_m4b_tags, so reading back
    through it cannot distinguish the wrong atom from the right one. Check
    the raw atom mutagen itself resolves."""
    from mutagen.mp4 import MP4
    write_m4b_tags(m4b, {"composer": "Narrator"})
    assert MP4(m4b)["\xa9wrt"] == ["Narrator"]


def test_cover_format_png_is_honoured(m4b, tmp_path):
    """P10 survived: cover_format="png" was ignored and every cover written
    as JPEG regardless -- cover_bytes > 0 can't tell the difference, only the
    raw MP4Cover's declared image format can."""
    from mutagen.mp4 import MP4, MP4Cover
    write_m4b_tags(m4b, {"title": "T"}, cover=_jpeg(tmp_path), cover_format="png")
    assert MP4(m4b)["covr"][0].imageformat == MP4Cover.FORMAT_PNG
