"""Pytest fixtures: synthesize tiny audio files via ffmpeg lavfi at session start."""
import os
import subprocess
import shutil
import pytest


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
