"""End-to-end cleanup smoke test using the fixture m4b."""
import shutil
import subprocess
from m4b_lib.cleanup import clean_one
from m4b_lib import ffmpeg_utils


def test_cleanup_basic_preserves_duration(fixture_m4b, tmp_path):
    """
    Test that cleanup in basic mode preserves the audio duration (within 0.5s).
    Forces processing by using a very high skip-threshold so even the clean
    sine-wave fixture gets processed.
    """
    work = tmp_path / "work.m4b"
    shutil.copy(fixture_m4b, work)
    original_dur = ffmpeg_utils.get_duration(str(work))

    # Use a very high skip-threshold to force processing on the clean fixture
    clean_one(str(work), mode="basic", keep_original=False, skip_threshold_db=1.0)

    cleaned_dur = ffmpeg_utils.get_duration(str(work))
    assert abs(cleaned_dur - original_dur) < 0.5, \
        f"duration mismatch: original={original_dur}, cleaned={cleaned_dur}"


def test_cleanup_skips_already_quiet(fixture_m4b, tmp_path):
    """
    Test that cleanup skips processing when the noise floor is already below
    the skip_threshold_db. Verifies the file is not modified (mtime unchanged).
    """
    work = tmp_path / "work2.m4b"
    shutil.copy(fixture_m4b, work)
    mtime_before = work.stat().st_mtime

    # Default threshold should skip on a quiet sine wave
    clean_one(str(work), mode="basic", skip_threshold_db=-5.0)

    # File should be untouched (no replace)
    assert work.stat().st_mtime == mtime_before
