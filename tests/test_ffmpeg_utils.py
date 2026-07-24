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


def test_create_chapters_ffmetadata(tmp_path, fixtures_dir):
    mp3s = [str(fixtures_dir / "fixture_ch01.mp3"), str(fixtures_dir / "fixture_ch02.mp3")]
    out = tmp_path / "chapters.ini"
    from m4b_lib.ffmpeg_utils import create_chapters_ffmetadata
    create_chapters_ffmetadata(mp3s, str(out))
    content = out.read_text()
    assert content.startswith(";FFMETADATA1")
    assert content.count("[CHAPTER]") == 2
    assert "START=0" in content
