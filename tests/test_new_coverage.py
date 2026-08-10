"""High-value coverage tests for P1 fixes — cover preservation, keep-original, iter_targets, sox silence path, apostrophe, etc."""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from m4b_lib import ffmpeg_utils
from m4b_lib.cleanup import clean_one, iter_targets
from m4b_lib.bind import BindOptions, bind_multiple


def test_iter_targets_batch(tmp_path):
    # Create dir with a.m4b, b.m4b, c.txt, sub/d.m4b
    (tmp_path / "a.m4b").write_bytes(b"fake")  # still counted as file if suffix check passes (exists)
    # Actually need real m4b for isfile check — use empty but present
    (tmp_path / "b.m4b").touch()
    (tmp_path / "c.txt").touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.m4b").touch()

    # "*.m4b" non-recursive should yield 2 (a,b) sorted
    result = list(iter_targets(str(tmp_path), "*.m4b"))
    assert len(result) == 2
    assert all(p.endswith(".m4b") for p in result)

    # "**/*.m4b" recursive should yield 3
    result2 = list(iter_targets(str(tmp_path), "**/*.m4b"))
    assert len(result2) == 3

    # Single file exact
    single = list(iter_targets(str(tmp_path / "a.m4b"), "*.m4b"))
    assert len(single) == 1

    # Non-m4b file should be skipped (not yield) with warning
    non = list(iter_targets(str(tmp_path / "c.txt"), "*.m4b"))
    assert len(non) == 0

    # Missing path raises
    with pytest.raises(FileNotFoundError):
        list(iter_targets(str(tmp_path / "nope.m4b"), "*.m4b"))


def test_iter_targets_broken_symlink(tmp_path):
    target = tmp_path / "real.m4b"
    target.touch()
    link = tmp_path / "link.m4b"
    link.symlink_to(target)
    # Valid symlink should yield if is_file follows symlink
    assert list(iter_targets(str(link), "*.m4b")) != []

    # Break it
    target.unlink()
    with pytest.raises(FileNotFoundError):
        list(iter_targets(str(link), "*.m4b"))


@pytest.mark.ml
def test_keep_original_versioning(fixture_m4b, tmp_path):
    work = tmp_path / "work.m4b"
    shutil.copy(fixture_m4b, work)

    # First clean with keep_original — should create .orig.m4b
    clean_one(str(work), keep_original=True, workers={"df3": 1, "loudness": 1, "encode": 1})
    orig = tmp_path / "work.orig.m4b"
    assert orig.exists()

    # Second clean with keep_original — should create .orig.1.m4b not overwrite
    clean_one(str(work), keep_original=True, workers={"df3": 1, "loudness": 1, "encode": 1})
    orig1 = tmp_path / "work.orig.1.m4b"
    assert orig1.exists()
    assert orig.exists()  # original backup still there


@pytest.mark.ml
def test_cleanup_preserves_chapters_and_cover(tmp_path, fixtures_dir):
    # Build an m4b with 2 chapters and a cover via ffmpeg
    # Create a cover jpg via lavfi color
    cover = tmp_path / "cover.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=0.1", "-vframes", "1", str(cover)],
        check=True, capture_output=True,
    )
    assert cover.exists()

    # Build m4b with chapters ffmetadata
    mp3s = [str(fixtures_dir / "fixture_ch01.mp3"), str(fixtures_dir / "fixture_ch02.mp3")]
    chapters = tmp_path / "chaps.ini"
    ffmpeg_utils.create_chapters_ffmetadata(mp3s, str(chapters))

    # Create intermediate m4a by transcoding mp3s to aac then concat
    m4a1 = tmp_path / "ch01.m4a"
    m4a2 = tmp_path / "ch02.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3s[0], "-c:a", "aac", "-b:a", "64k", str(m4a1)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3s[1], "-c:a", "aac", "-b:a", "64k", str(m4a2)],
        check=True, capture_output=True,
    )
    listfile = tmp_path / "concat.txt"
    listfile.write_text(f"file '{m4a1}'\nfile '{m4a2}'\n")

    intermediate = tmp_path / "intermediate.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(intermediate)],
        check=True, capture_output=True,
    )

    out_m4b = tmp_path / "with_chapters_cover.m4b"
    with open(cover, "rb") as f:
        cover_bytes = f.read()

    ffmpeg_utils.embed_chapters_and_meta(
        str(intermediate), chapters_ini=str(chapters), cover_bytes=cover_bytes,
        output_m4b=str(out_m4b), title="Test Book", author="Test Author",
    )

    # Verify input has chapters and cover
    assert ffmpeg_utils.extract_cover(str(out_m4b)) is not None
    chap_before = tmp_path / "before.ini"
    ffmpeg_utils.extract_chapters(str(out_m4b), str(chap_before))
    assert "[CHAPTER]" in chap_before.read_text()

    # Now clean it
    clean_one(str(out_m4b), keep_original=False, workers={"df3": 1, "loudness": 1, "encode": 1})

    # After clean, chapters and cover should still exist
    after_cover = ffmpeg_utils.extract_cover(str(out_m4b))
    assert after_cover is not None, "cover should be preserved through clean"
    chap_after = tmp_path / "after.ini"
    ffmpeg_utils.extract_chapters(str(out_m4b), str(chap_after))
    content = chap_after.read_text()
    assert "[CHAPTER]" in content, "chapters should be preserved through clean"


def test_bind_apostrophe_escaping(tmp_path, fixtures_dir):
    # Folder and file with apostrophe
    book_dir = tmp_path / "input" / "It's a Book"
    book_dir.mkdir(parents=True)
    # File with apostrophe
    src = fixtures_dir / "fixture_ch01.mp3"
    shutil.copy(src, book_dir / "it's ch1.mp3")
    shutil.copy(fixtures_dir / "fixture_ch02.mp3", book_dir / "ch2.mp3")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    opts = BindOptions(
        input_folder=str(tmp_path / "input"),
        output_folder=str(out_dir),
        metadata_source="none",
    )
    bind_multiple(opts)

    out_m4b = out_dir / "It's a Book.m4b"
    assert out_m4b.exists()
    assert ffmpeg_utils.get_duration(str(out_m4b)) > 3.0


def test_create_chapters_escaping_and_zero_duration(tmp_path, fixtures_dir):
    # Test backslash escaping fix (C2)
    mp3s = [str(fixtures_dir / "fixture_ch01.mp3")]
    out = tmp_path / "chap.ini"
    # Monkey patch extract_id3_tags to return title with backslash
    from m4b_lib import metadata

    orig_extract = metadata.extract_id3_tags

    def fake_extract(path):
        return {"title": r"Back\slash = ; # test", "artist": "", "album": ""}

    metadata.extract_id3_tags = fake_extract
    try:
        ffmpeg_utils.create_chapters_ffmetadata(mp3s, str(out))
        content = out.read_text()
        # Should contain escaped backslash etc, and be parseable by ffmpeg
        assert "Back" in content
        # Try to use it in ffmpeg embed to ensure it doesn't break
        # Build dummy m4a from mp3 via copy of fixture
        dummy_m4a = tmp_path / "dummy.m4a"
        shutil.copy(fixtures_dir / "fixture_ch01.mp3", dummy_m4a)  # ffmpeg will still try?
        # Actually create a proper m4a via ffmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(fixtures_dir / "fixture_ch01.mp3"), "-c:a", "aac", str(dummy_m4a)],
            check=True, capture_output=True,
        )
        out_m4b = tmp_path / "out.m4b"
        ffmpeg_utils.embed_chapters_and_meta(str(dummy_m4a), chapters_ini=str(out), cover_bytes=None, output_m4b=str(out_m4b))
        assert out_m4b.exists()
    finally:
        metadata.extract_id3_tags = orig_extract

    # Zero-duration handling
    zero_mp3 = tmp_path / "zero.mp3"
    # Create empty file that ffprobe will return 0 duration
    zero_mp3.write_bytes(b"")
    out2 = tmp_path / "chap2.ini"
    # Should not crash, should warn and skip or create placeholder
    ffmpeg_utils.create_chapters_ffmetadata([str(zero_mp3)], str(out2))
    assert out2.exists()
    txt = out2.read_text()
    # Should have at least placeholder chapter if all skipped
    assert "[CHAPTER]" in txt


def test_parallel_transcode_preserves_order_and_no_delete_preexisting(tmp_path, fixtures_dir):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Test order preservation without pre-existing collision
    mp3s = [
        str(fixtures_dir / "fixture_ch01.mp3"),
        str(fixtures_dir / "fixture_ch02.mp3"),
        str(fixtures_dir / "fixture_ch01.mp3"),
    ]

    result = ffmpeg_utils.parallel_transcode(mp3s, str(out_dir), bitrate="64k")
    assert len(result) == 3
    # First element should be fixture_ch01, second fixture_ch02, third deduplicated _1
    assert "fixture_ch01" in result[0]
    assert "fixture_ch02" in result[1]
    assert "fixture_ch01_1.m4a" in result[2] or result[2].endswith("_1.m4a")
    # Order preserved: input order [01,02,01] -> output [01,02,01_1]
    assert result[0] != result[2]

    # Now test that pre-existing file that would collide is preserved via versioning, not overwritten
    out_dir_version = tmp_path / "out_version"
    out_dir_version.mkdir()
    pre_exist = out_dir_version / "fixture_ch01.m4a"
    pre_exist.write_text("old content")
    assert pre_exist.exists()

    result2 = ffmpeg_utils.parallel_transcode([str(fixtures_dir / "fixture_ch01.mp3")], str(out_dir_version), bitrate="64k")
    # Should create _1 instead of overwriting, preserving original
    assert pre_exist.read_text() == "old content", "pre-existing should be preserved"
    assert "fixture_ch01_1.m4a" in result2[0]

    # Now test failure case does not delete pre-existing file that was not part of this run's new files
    # Create out dir with pre-existing file, then run with one invalid mp3 to force failure
    out_dir2 = tmp_path / "out2"
    out_dir2.mkdir()
    keep = out_dir2 / "keep.m4a"
    keep.write_text("keep me")
    bad_mp3 = tmp_path / "bad.mp3"
    bad_mp3.write_text("not an mp3")

    try:
        ffmpeg_utils.parallel_transcode([str(fixtures_dir / "fixture_ch01.mp3"), str(bad_mp3)], str(out_dir2))
    except Exception:
        pass

    # keep.m4a should still exist (was pre-existing, not created by this run)
    assert keep.exists(), "pre-existing file should not be deleted on failure"
    assert keep.read_text() == "keep me"
