"""Tests for m4b_lib.migrate — one-step bind+clean.

Task 8 rewired clean_timeline's signature but left migrate.py doing a bare
float(clean_atten_lim), which crashes on "auto" (ValueError: could not
convert string to float: 'auto'). Task 9 restored "auto" here via the shared
ffmpeg_utils.resolve_atten_lim — this file is the regression test for that
fix, since nothing previously exercised migrate.py's atten-lim handling at
all.
"""
import pytest

from m4b_lib import ffmpeg_utils
from m4b_lib.bind import BindOptions


@pytest.mark.ml
def test_bind_and_clean_one_atten_lim_auto(fixtures_dir, tmp_path):
    """--atten-lim auto must resolve to a real float and run the full
    bind+clean pipeline end to end, not raise on the string 'auto'."""
    from m4b_lib.migrate import bind_and_clean_single

    out = tmp_path / "out.m4b"
    opts = BindOptions(
        input_folder=str(fixtures_dir),
        output_file=str(out),
        metadata_source="none",
    )
    bind_and_clean_single(opts, clean_atten_lim="auto")

    assert out.exists()
    dur = ffmpeg_utils.get_duration(str(out))
    assert 3.8 < dur < 4.5  # two 2s fixture chapters, with concat slop
