"""ffmpeg echoes paths and tags to stderr; neither is guaranteed to be UTF-8.

This library holds a large number of Japanese audiobooks, and a real one --
`細雪.m4b` -- crashed `estimate_optimal_atten_lim` with

    UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 2700-2701

because ffmpeg echoed its Shift-JIS tags to stderr and every `subprocess.run` in
`ffmpeg_utils` decoded with `text=True` and the strict default error handler. A
crash, not a bad answer, on a substantial fraction of the library.

Linux filenames are bytes, so an undecodable *path* reproduces the same failure
without needing a specially tagged m4b, and does it for every helper in the
module at once.
"""
import os
import subprocess
import tempfile

import pytest

from m4b_lib import ffmpeg_utils

# Lone 0xFF: never valid anywhere in UTF-8.
BAD_NAME = b"probe-\xff-tone.wav"


@pytest.fixture
def undecodable_wav():
    d = tempfile.mkdtemp()
    path = os.path.join(os.fsdecode(d), os.fsdecode(BAD_NAME))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2", "-ac", "1", "-ar", "16000",
         os.fsencode(path)],
        check=True, capture_output=True)
    yield path
    os.remove(os.fsencode(path))
    os.rmdir(d)


def test_noise_floor_survives_an_undecodable_path(undecodable_wav):
    """`probe_noise_floor` parses ffmpeg's stderr, which echoes the input path.

    The bug was a raised exception, so the assertion is that we get a number at
    all. Strict decoding raises UnicodeDecodeError; `errors="replace"` returns
    the measurement. Deliberately not asserting on the substituted characters --
    the contract is that a mangled tag never costs us the audio measurement.

    The sine measures about -21 dBFS mean. The bound is loose because the point
    is "measured something real", but it is not vacuous: it excludes the -91.0
    sentinel this function returns for digital silence, which is what a mangled
    parse would most plausibly degrade into.
    """
    level = ffmpeg_utils.probe_noise_floor(undecodable_wav)
    assert level is not None
    assert -40.0 < level < 0.0


def test_the_fixture_really_is_undecodable():
    """Guards the guard: if this filename ever became valid UTF-8, the test
    above would pass for the wrong reason and stop protecting anything."""
    with pytest.raises(UnicodeDecodeError):
        BAD_NAME.decode("utf-8")
