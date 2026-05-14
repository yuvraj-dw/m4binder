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
    - Verify the output m4b exists with expected duration
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
