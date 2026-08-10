"""Intermediate audio must not be capped at RIFF's 4 GiB limit.

A real 12.7-hour book and a real 24.2-hour book both produced cleaned output of
exactly 44739.3 s. That is not a coincidence and it is not bad metadata: RIFF
stores its sizes in 32-bit fields, so a WAV cannot exceed 4 GiB, which at mono
16-bit 48 kHz is exactly 44739.24 s = 12.43 hours. Every book longer than that
was being truncated at the boundary.

The whole suite passed while this was broken, because nothing generated a file
anywhere near 4 GiB. These tests assert on the CONTAINER instead, which is the
property that actually decides the ceiling, and they run in milliseconds.
"""
import os
import subprocess

import numpy as np
import pytest
import soundfile as sf

RIFF_LIMIT_BYTES = 2 ** 32
SR = 48000
BYTES_PER_FRAME_PCM16_MONO = 2
RIFF_CAP_SECONDS = RIFF_LIMIT_BYTES / BYTES_PER_FRAME_PCM16_MONO / SR   # 44739.24


def test_the_arithmetic_that_defines_the_bug():
    """Documents where the magic number comes from, so a future reader does not
    have to rediscover why 44739.24 matters."""
    assert RIFF_CAP_SECONDS == pytest.approx(44739.24, abs=0.01)
    assert RIFF_CAP_SECONDS / 3600 == pytest.approx(12.43, abs=0.01)


def test_enhanced_output_container_has_no_4gib_ceiling(tmp_path):
    """The enhanced timeline holds the whole book, so its container decides the
    maximum book length the tool can process.

    Calls the real writer from scheduler. An earlier version of this test wrote
    the file with its own helper that hardcoded format="W64", so it passed with
    the fix reverted -- it was testing the test.
    """
    from m4b_lib.scheduler import open_timeline_writer

    out = tmp_path / "enhanced.wav"
    with open_timeline_writer(str(out), SR) as f:
        f.write(np.zeros(1000, dtype="float32"))
    assert sf.info(str(out)).format == "W64"


def test_decoded_timeline_container_has_no_4gib_ceiling(tmp_path):
    """timeline._decode must emit Wave64 regardless of the .wav filename."""
    from m4b_lib import timeline

    src = tmp_path / "src.wav"
    sf.write(str(src), np.zeros(SR, dtype="float32"), SR)
    dst = tmp_path / "timeline.wav"
    frames = timeline._decode(str(src), str(dst), SR)
    assert frames == SR
    assert sf.info(str(dst)).format == "W64"


@pytest.mark.skipif(not os.path.exists("/usr/bin/ffmpeg") and
                    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0,
                    reason="ffmpeg not on PATH")
def test_w64_survives_the_read_patterns_the_pipeline_uses(tmp_path):
    """Wave64 is only a safe swap if every access pattern still works."""
    p = tmp_path / "x.wav"
    data = np.sin(np.arange(SR) * 0.01).astype("float32")
    sf.write(str(p), data, SR, subtype="PCM_16", format="W64")

    assert sf.read(str(p))[0].shape == (SR,)
    assert next(sf.blocks(str(p), blocksize=1000, dtype="float32")).shape == (1000,)
    with sf.SoundFile(str(p)) as f:
        f.seek(500)
        assert f.read(100).shape == (100,)
    assert sf.info(str(p)).frames == SR
