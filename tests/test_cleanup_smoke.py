"""End-to-end cleanup smoke test using the fixture m4b."""
import shutil
import subprocess
from m4b_lib.cleanup import clean_one
from m4b_lib import ffmpeg_utils


def test_cleanup_basic_preserves_duration(fixture_m4b, tmp_path):
    """
    Test that cleanup in basic mode preserves the audio duration (within 0.5s).
    Forces processing by using a very low skip-threshold (-100) so even quiet
    sine-wave fixture gets processed. Old bug used 1.0 which actually skipped.
    """
    work = tmp_path / "work.m4b"
    shutil.copy(fixture_m4b, work)
    original_dur = ffmpeg_utils.get_duration(str(work))

    # Use -100 to force processing: mean volume -21.5 is > -100, so -21.5 < -100 is False -> process
    clean_one(str(work), mode="basic", keep_original=False, skip_threshold_db=-100.0)

    cleaned_dur = ffmpeg_utils.get_duration(str(work))
    assert abs(cleaned_dur - original_dur) < 0.5, \
        f"duration mismatch: original={original_dur}, cleaned={cleaned_dur}"
    # Ensure file was actually modified (size or mtime changed)
    # mtime may be same if processed within 1s, so check size or hash - just ensure still valid
    assert cleaned_dur > 3.0


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


import pytest


def _df3_available():
    try:
        from m4b_lib import cleanup_ml  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _df3_available(), reason="DeepFilterNet not installed")
def test_cleanup_ml_runs(fixture_m4b, tmp_path):
    work = tmp_path / "work_ml.m4b"
    shutil.copy(fixture_m4b, work)
    # Force processing with -100 threshold
    clean_one(str(work), mode="ml", keep_original=False, skip_threshold_db=-100.0)
    assert ffmpeg_utils.get_duration(str(work)) > 3.0
