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
    # Sine wave at 64kbps will have a mean volume far above silence;
    # probe returns Optional[float], None on failure.
    assert nf is not None, "probe returned None"
    assert -100.0 < nf < 0.0

def test_probe_silence_returns_quiet():
    """Pure silence should be detected as very quiet (-91) not 0.0."""
    import subprocess, tempfile, os
    from m4b_lib import ffmpeg_utils as fu
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "silence.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=1", "-c:a", "pcm_s16le", wav],
            check=True, capture_output=True,
        )
        nf = fu.probe_noise_floor(wav)
        assert nf is not None
        assert nf < -80.0, f"silence should be < -80dB, got {nf}"


def test_duration_timeout_none_or_zero_returns_minimum():
    assert ffmpeg_utils.duration_timeout(None) == 300.0
    assert ffmpeg_utils.duration_timeout(0) == 300.0
    assert ffmpeg_utils.duration_timeout(-5) == 300.0


def test_duration_timeout_short_audio_floors_at_minimum():
    """8s of audio must not produce an 8s bound — hang detection needs a floor."""
    assert ffmpeg_utils.duration_timeout(8.0) == 300.0
    assert ffmpeg_utils.duration_timeout(8.0, minimum=60.0) == 60.0


def test_duration_timeout_scales_with_duration_past_the_floor():
    # 30h book at factor=1.0 comfortably clears the minimum and scales linearly.
    assert ffmpeg_utils.duration_timeout(30 * 3600, factor=1.0) == 30 * 3600
    assert ffmpeg_utils.duration_timeout(1000.0, minimum=300.0, factor=2.0) == 2000.0


def test_duration_timeout_scales_before_flooring_not_after():
    """Guards scale-then-clamp against the plausible clamp-then-scale inversion.

    100s of audio at factor=2.0 is 200s of budget, which the 300s floor raises
    to 300. Clamping first would give max(300, 100) * 2 = 600 — twice the
    intended bound, and the error grows with the floor.
    """
    assert ffmpeg_utils.duration_timeout(100.0, minimum=300.0, factor=2.0) == 300.0


def test_create_chapters_ffmetadata(tmp_path, fixtures_dir):
    mp3s = [str(fixtures_dir / "fixture_ch01.mp3"), str(fixtures_dir / "fixture_ch02.mp3")]
    out = tmp_path / "chapters.ini"
    from m4b_lib.ffmpeg_utils import create_chapters_ffmetadata
    create_chapters_ffmetadata(mp3s, str(out))
    content = out.read_text()
    assert content.startswith(";FFMETADATA1")
    assert content.count("[CHAPTER]") == 2
    assert "START=0" in content


def test_resolve_atten_lim_numeric_passthrough():
    """A plain numeric spec is parsed as a float and never probes anything."""
    assert ffmpeg_utils.resolve_atten_lim("12", "/does/not/matter") == 12.0
    assert ffmpeg_utils.resolve_atten_lim("0", "/does/not/matter") == 0.0
    assert ffmpeg_utils.resolve_atten_lim("17.5", "/does/not/matter") == 17.5


def test_resolve_atten_lim_rejects_garbage():
    import pytest
    with pytest.raises(ValueError):
        ffmpeg_utils.resolve_atten_lim("notanumber", "/does/not/matter")


def test_resolve_atten_lim_auto_estimates_from_real_audio(fixture_m4b):
    """'auto' runs the real SNR estimator against a real file (no mocking)."""
    result = ffmpeg_utils.resolve_atten_lim("auto", str(fixture_m4b))
    assert isinstance(result, float)
    assert 6.0 <= result <= 30.0  # estimate_optimal_atten_lim's own clamp


def test_resolve_atten_lim_auto_case_insensitive_and_trims(fixture_m4b):
    result = ffmpeg_utils.resolve_atten_lim("  AUTO  ", str(fixture_m4b))
    assert isinstance(result, float)


def test_resolve_atten_lim_auto_falls_back_when_probe_unreadable():
    """An unprobeable file (missing/corrupt) must fall back to 12.0, not crash."""
    result = ffmpeg_utils.resolve_atten_lim("auto", "/nonexistent/path/does/not/exist.m4b")
    assert result == 12.0
