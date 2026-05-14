"""Smoke tests for ffmpeg utility functions."""
import subprocess
from m4b_lib import ffmpeg_utils


def test_get_duration(fixture_m4b):
    dur = ffmpeg_utils.get_duration(str(fixture_m4b))
    assert 3.8 < dur < 4.5  # two 2s sines, with concat slop


def test_chapter_extract_embed_roundtrip(fixture_m4b, tmp_path):
    chapters_ini = tmp_path / "chapters.ini"
    ffmpeg_utils.extract_chapters(str(fixture_m4b), str(chapters_ini))
    assert chapters_ini.exists()
    # Synthetic fixture has no chapters yet (just concatenated audio).
    # We assert the function ran successfully and produced a valid ffmetadata file.
    content = chapters_ini.read_text()
    assert content.startswith(";FFMETADATA1")


def test_probe_noise_floor(fixture_m4b):
    nf = ffmpeg_utils.probe_noise_floor(str(fixture_m4b))
    # Sine wave at 64kbps will have a noise floor floor far above silence;
    # just assert it returns a float in a reasonable dB range.
    assert -100.0 < nf < 0.0
