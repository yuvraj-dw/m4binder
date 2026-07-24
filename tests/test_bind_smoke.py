"""Smoke test: bind_multiple produces valid m4b files from a fixture tree."""
import os
import shutil
import subprocess
from m4b_lib.bind import BindOptions, bind_multiple
from m4b_lib import ffmpeg_utils


def test_bind_multiple_smoke(fixtures_dir, tmp_path):
    """
    Test bind_multiple end-to-end:
    - Build a "multiple mode" input tree with one synthetic book
    - Run bind_multiple
    - Verify the output m4b exists with expected duration and chapters
    """
    # Build a "multiple mode" input tree with one synthetic book
    book_dir = tmp_path / "input" / "Test Book"
    book_dir.mkdir(parents=True)
    shutil.copy(fixtures_dir / "fixture_ch01.mp3", book_dir / "01.mp3")
    shutil.copy(fixtures_dir / "fixture_ch02.mp3", book_dir / "02.mp3")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    opts = BindOptions(
        input_folder=str(tmp_path / "input"),
        output_folder=str(out_dir),
        metadata_source="none",
    )
    bind_multiple(opts)

    out_m4b = out_dir / "Test Book.m4b"
    assert out_m4b.exists(), f"expected output at {out_m4b}"
    duration = ffmpeg_utils.get_duration(str(out_m4b))
    assert 3.8 < duration < 4.5, f"unexpected duration {duration}"

    # Verify chapters were generated (regression test for lost chapters bug)
    chapters_ini = tmp_path / "chapters.ini"
    ffmpeg_utils.extract_chapters(str(out_m4b), str(chapters_ini))
    content = chapters_ini.read_text()
    assert "[CHAPTER]" in content, "bind should produce chapter markers"


def test_bind_natural_sort(tmp_path, fixtures_dir):
    """Files should be naturally sorted: 1,2,10 not 1,10,2."""
    book_dir = tmp_path / "input" / "SortTest"
    book_dir.mkdir(parents=True)
    # Create files with natural sort challenge
    for name in ["1.mp3", "10.mp3", "2.mp3"]:
        shutil.copy(fixtures_dir / "fixture_ch01.mp3", book_dir / name)

    out_dir = tmp_path / "output2"
    out_dir.mkdir()

    opts = BindOptions(
        input_folder=str(tmp_path / "input"),
        output_folder=str(out_dir),
        metadata_source="none",
    )
    bind_multiple(opts)
    out_m4b = out_dir / "SortTest.m4b"
    assert out_m4b.exists()
    # Duration should be ~6s (3*2s) if all 3 files counted in order
    duration = ffmpeg_utils.get_duration(str(out_m4b))
    assert 5.5 < duration < 6.5, f"expected ~6s for 3 files, got {duration}"
