"""TDD tests for H1, H3, H4 — should FAIL before fixes, PASS after."""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path

import pytest

from m4b_lib import ffmpeg_utils
from m4b_lib.bind import BindOptions, _bind_one, bind_multiple


# H4: get_duration should NOT return 0.0 for missing/corrupt files
def test_h4_get_duration_missing_returns_none_or_raises():
    """H4: missing file should not return 0.0 — should be None or raise."""
    result = ffmpeg_utils.get_duration("/nonexistent/path/file.mp3")
    # After fix, should be None or raise, NOT 0.0
    # This test expects None (or exception) — will FAIL if returns 0.0
    assert result is None or isinstance(result, float) and result != 0.0, \
        f"get_duration returned 0.0 for missing file, masking error — got {result}"
    # Also test that it actually returns None, not 0.0
    assert result is None, f"Expected None for missing file, got {result}"


def test_h4_get_duration_corrupt_returns_none():
    """Corrupt empty file should not return 0.0."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.mp3"
        bad.write_text("not an mp3")
        result = ffmpeg_utils.get_duration(str(bad))
        assert result is None, f"Expected None for corrupt file, got {result}"


# H3: parallel_transcode should NOT overwrite pre-existing files on success
def test_h3_parallel_transcode_does_not_overwrite_preexisting(fixtures_dir, tmp_path):
    """H3: pre-existing file in output dir should not be silently overwritten."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pre_exist = out_dir / "fixture_ch01.m4a"
    pre_exist.write_text("original content")

    mp3s = [str(fixtures_dir / "fixture_ch01.mp3")]

    # After fix, should either raise FileExistsError or preserve original via versioning
    # Before fix, it overwrites with -y and test fails
    try:
        result = ffmpeg_utils.parallel_transcode(mp3s, str(out_dir), bitrate="64k")
    except FileExistsError:
        # Acceptable fix: raises instead of overwriting
        assert pre_exist.read_text() == "original content"
        return

    # If it didn't raise, check content was NOT overwritten (i.e., still original)
    # Current buggy behavior overwrites, so this assertion will FAIL before fix
    content = pre_exist.read_bytes()
    # If file is now valid m4a, it was overwritten — fail
    assert b"original content" in content or content == b"original content", \
        "pre-existing file was overwritten — should be preserved or raise"


# H3: bind final output should not silently overwrite
def test_h3_bind_does_not_silently_overwrite_final(fixtures_dir, tmp_path):
    """H3: bind_single / _bind_one final m4b should not silently overwrite existing without atomic safety."""
    # Create input folder with mp3s
    input_dir = tmp_path / "book"
    input_dir.mkdir()
    shutil.copy(fixtures_dir / "fixture_ch01.mp3", input_dir / "01.mp3")
    shutil.copy(fixtures_dir / "fixture_ch02.mp3", input_dir / "02.mp3")

    output_m4b = tmp_path / "output.m4b"
    output_m4b.write_text("original m4b content")

    opts = BindOptions(
        input_folder=str(input_dir),
        output_file=str(output_m4b),
        metadata_source="none",
        bitrate="64k",
    )

    # Before fix: will overwrite original with -y directly
    # After fix: should either write to temp then replace atomically with validation,
    # and ideally should preserve backup or raise if exists (or at least not leave partial on failure)
    # For TDD, we test that if bind fails, original is not left truncated

    # Make bind fail during embed by providing invalid bitrate that ffmpeg will reject
    opts_fail = BindOptions(
        input_folder=str(input_dir),
        output_file=str(output_m4b),
        metadata_source="none",
        bitrate="invalid_bitrate$$$",
    )

    try:
        _bind_one(str(input_dir), str(output_m4b), opts_fail)
    except Exception:
        # After failure, original file should still exist and not be truncated to 0/partial
        # Before fix: ffmpeg -y truncates final file at open, so original is destroyed
        assert output_m4b.exists(), "output file should still exist after failed bind"
        # Content should be either original or not truncated to < original size?
        # We check that file is not empty and not partial m4a header only — should still be original or at least >0 and contains original marker
        # For strict TDD: if atomic fix applied, original should be intact
        data = output_m4b.read_bytes()
        # If atomic, original content preserved
        if b"original m4b content" in data:
            # Good — atomic preserved original
            pass
        else:
            # If file was overwritten/truncated, this fails — showing H1 bug too
            pytest.fail(f"output file was truncated/corrupted on failed bind — original lost: size {len(data)}")


# H1: bind should be atomic — no partial file on failure
def test_h1_bind_atomic_no_partial_on_failure(fixtures_dir, tmp_path):
    """H1: _bind_one should not leave partial/corrupt file at final location on failure."""
    input_dir = tmp_path / "book"
    input_dir.mkdir()
    shutil.copy(fixtures_dir / "fixture_ch01.mp3", input_dir / "01.mp3")

    output_m4b = tmp_path / "output.m4b"
    # Ensure output does NOT exist before
    if output_m4b.exists():
        output_m4b.unlink()

    opts = BindOptions(
        input_folder=str(input_dir),
        output_file=str(output_m4b),
        metadata_source="none",
        bitrate="invalid!!!",  # will cause ffmpeg to fail
    )

    try:
        _bind_one(str(input_dir), str(output_m4b), opts)
    except Exception:
        pass

    # After failed bind, there should be NO partial file left, or if file exists it should be valid (>1k and probeable)
    # Before fix: ffmpeg -y creates output_m4b then fails, leaving 0-byte or partial file
    if output_m4b.exists():
        size = output_m4b.stat().st_size
        # If file exists but is <1024, it's partial — bug
        assert size < 1024 and size == 0 or size >= 1024, \
            f"Partial file left at {output_m4b} size {size} — should be cleaned up or not exist"
        if size > 0 and size < 1024:
            pytest.fail(f"H1 bug: partial corrupt file left at final location size {size}")
        # If we decide atomic fix, file should not exist at all after failure when it didn't before
        # So we assert it does NOT exist after failure (cleaned up)
        assert not output_m4b.exists(), \
            f"H1 bug: partial file left at final location after failure — should be removed (atomic)"


def test_h1_embed_atomic_direct(tmp_path, fixtures_dir):
    """H1: embed_chapters_and_meta should write via temp + replace, not direct -y truncation."""
    # Create existing output with original content
    output_m4b = tmp_path / "final.m4b"
    output_m4b.write_text("original")

    # Create valid intermediate audio via transcoding one fixture
    intermediate = tmp_path / "inter.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(fixtures_dir / "fixture_ch01.mp3"), "-c:a", "aac", "-b:a", "64k", str(intermediate)],
        check=True, capture_output=True,
    )

    # Now call embed with invalid cover or invalid bitrate to force failure after file open
    # Use invalid bitrate to make ffmpeg fail
    try:
        ffmpeg_utils.embed_chapters_and_meta(
            str(intermediate), chapters_ini=None, cover_bytes=None,
            output_m4b=str(output_m4b), bitrate="not_a_bitrate", title="test"
        )
    except Exception:
        pass

    # After failure, original should still be intact if atomic
    assert output_m4b.exists(), "output should still exist after failed embed"
    data = output_m4b.read_bytes()
    # Before fix: file truncated to 0 or partial m4a
    if b"original" not in data:
        pytest.fail(f"H1: original file was destroyed on failed embed — atomic fix needed, got size {len(data)}")


# H3: bind_multiple should not overwrite existing final m4b silently (without --overwrite)
def test_h3_bind_multiple_no_overwrite_without_flag(fixtures_dir, tmp_path):
    """H3: bind_multiple should not silently overwrite existing book.m4b"""
    # Build input tree with one book "MyBook"
    input_root = tmp_path / "input"
    book_dir = input_root / "MyBook"
    book_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "fixture_ch01.mp3", book_dir / "01.mp3")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "MyBook.m4b"
    existing.write_text("existing book")

    opts = BindOptions(
        input_folder=str(input_root),
        output_folder=str(out_dir),
        metadata_source="none",
    )

    # Call bind_multiple — before fix, it will overwrite existing file
    bind_multiple(opts)

    # After fix, should either preserve existing, or raise, or version
    # For TDD red: we expect existing content preserved or FileExistsError
    # If overwritten, this fails
    if existing.exists():
        content = existing.read_bytes()
        if b"existing book" not in content:
            pytest.fail("H3: bind_multiple overwrote existing final m4b without warning — should guard")
    else:
        pytest.fail("H3: existing file was deleted")
