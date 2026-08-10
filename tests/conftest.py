"""Pytest fixtures: synthesize tiny audio files via ffmpeg lavfi at session start."""
import subprocess
import importlib.util
import pytest


def _ml_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "df", "soundfile"))


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.ml tests when the ml extra isn't installed."""
    if _ml_available():
        return
    skip = pytest.mark.skip(reason="ML extra not installed (pip install -e '.[ml]')")
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory):
    """Returns a dir containing fixture_ch01.mp3 and fixture_ch02.mp3 (2s sine waves)."""
    d = tmp_path_factory.mktemp("fixtures")
    for i, freq in enumerate([440, 880], start=1):
        out = d / f"fixture_ch{i:02d}.mp3"
        subprocess.run(
            [
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"sine=frequency={freq}:duration=2",
                "-metadata", f"title=Chapter {i}",
                "-metadata", "artist=Test Author",
                "-metadata", "album=Test Book",
                "-c:a", "libmp3lame", "-b:a", "64k",
                str(out),
            ],
            check=True, capture_output=True,
        )
    return d


@pytest.fixture(scope="session")
def fixture_m4b(fixtures_dir, tmp_path_factory):
    """A small m4b assembled from fixture_ch01/02.mp3 — used by cleanup tests."""
    out = tmp_path_factory.mktemp("m4b") / "fixture.m4b"
    # Build a tiny m4b by concatenating the two mp3s into an aac stream
    listfile = out.parent / "concat.txt"
    listfile.write_text(
        f"file '{fixtures_dir / 'fixture_ch01.mp3'}'\n"
        f"file '{fixtures_dir / 'fixture_ch02.mp3'}'\n"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listfile),
            "-c:a", "aac", "-b:a", "64k", str(out),
        ],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def noisy_wav_48k(tmp_path_factory):
    """8s of mono 48k speech-ish tone plus white noise, as float wav.

    Long enough to exercise two 3s chunks with 1s overlap in stream tests.
    """
    out = tmp_path_factory.mktemp("wav") / "noisy.wav"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=8:sample_rate=48000",
            "-f", "lavfi", "-i", "anoisesrc=duration=8:sample_rate=48000:amplitude=0.05",
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest",
            "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(out),
        ],
        check=True, capture_output=True,
    )
    return out
