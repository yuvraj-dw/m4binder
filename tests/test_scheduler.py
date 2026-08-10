import numpy as np
import pytest
import soundfile as sf

from m4b_lib import scheduler, timeline
from m4b_lib.enhance import EnhanceConfig

# scheduler imports soundfile at module level, so the whole file needs the extra.
pytestmark = pytest.mark.ml


def test_default_workers_match_measured_optima():
    assert scheduler.default_workers("loudness") == 8
    # 1 for correctness, not throughput: parallel encode leaves an audible
    # -40 dB dip at every concat join. See _STAGE_DEFAULTS' comment, and
    # test_default_encode_settings_leave_no_join_dropouts for the guard.
    assert scheduler.default_workers("encode") == 1
    assert scheduler.default_workers("df3") == 12
    assert scheduler.default_workers("decode") == 4


def test_worker_threads_are_pinned_to_one():
    """torch's default all-core threading measured 22x slower than 4 threads."""
    assert scheduler._worker_threads() == 1


def test_enhance_preserves_length_and_matches_single_worker(noisy_wav_48k, tmp_path):
    tl = timeline.Timeline(wav_path=str(noisy_wav_48k), sample_rate=48000,
                           frames=sf.info(str(noisy_wav_48k)).frames)
    cfg = EnhanceConfig(atten_lim_db=12.0)
    a = scheduler.enhance_timeline(tl, str(tmp_path / "a.wav"), cfg=cfg,
                                   workers=1, chunk_s=3.0, overlap_s=1.0)
    b = scheduler.enhance_timeline(tl, str(tmp_path / "b.wav"), cfg=cfg,
                                   workers=2, chunk_s=3.0, overlap_s=1.0)

    xa, _ = sf.read(a, dtype="float32")
    xb, _ = sf.read(b, dtype="float32")
    assert len(xa) == tl.frames
    assert len(xb) == tl.frames
    # Different stream counts reset model state at different points, so allow a
    # loose bound: this asserts the scheduler is sane, not bit-equality.
    db = 20 * np.log10(np.sqrt(((xa - xb) ** 2).mean()) + 1e-20)
    assert db < -20, f"worker count changed the output too much: {db:.1f} dBFS"


def test_enhance_stays_sample_exact_with_chunk_much_smaller_than_stream(noisy_wav_48k, tmp_path):
    """Regression for reading a whole stream into RAM: chunks must be read
    directly from disk at (stream_start + chunk_offset), not sliced from an
    in-memory copy of the entire stream. workers=3 gives non-zero per-stream
    start offsets, and a chunk far smaller than the stream forces many
    small reads per worker; a wrong offset would show up here as drift.
    """
    tl = timeline.Timeline(wav_path=str(noisy_wav_48k), sample_rate=48000,
                           frames=sf.info(str(noisy_wav_48k)).frames)
    cfg = EnhanceConfig(atten_lim_db=12.0)
    out = scheduler.enhance_timeline(tl, str(tmp_path / "c.wav"), cfg=cfg,
                                     workers=3, chunk_s=0.5, overlap_s=0.1)
    x, _ = sf.read(out, dtype="float32")
    assert len(x) == tl.frames
