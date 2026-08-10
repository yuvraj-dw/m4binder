import numpy as np
from m4b_lib.streams import StreamSpec, plan_streams, plan_chunks, overlap_add


def test_streams_tile_the_timeline_exactly():
    specs = plan_streams(frames=1000, n=3)
    assert specs[0].start_frame == 0
    assert specs[-1].end_frame == 1000
    for a, b in zip(specs, specs[1:]):
        assert a.end_frame == b.start_frame, "gap or overlap between streams"
    assert sum(s.end_frame - s.start_frame for s in specs) == 1000


def test_streams_never_exceed_requested_count_or_emit_empties():
    specs = plan_streams(frames=5, n=8)
    assert len(specs) <= 8
    assert all(s.end_frame > s.start_frame for s in specs)


def test_chunks_cover_length_with_requested_overlap():
    chunks = plan_chunks(length=250, chunk=100, overlap=20)
    assert chunks[0] == (0, 100)
    assert chunks[-1][1] == 250
    for a, b in zip(chunks, chunks[1:]):
        assert b[0] == a[1] - 20


def test_overlap_add_reconstructs_a_known_signal():
    """A correlated signal must survive the crossfade unchanged.

    Equal-power (sin/cos) crossfade produces up to +3 dB in the overlap region
    for correlated inputs; linear crossfade reconstructs exactly.
    """
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(250).astype("float32")
    pieces = [signal[a:b].copy() for a, b in plan_chunks(250, 100, 20)]
    out = overlap_add(pieces, overlap=20)
    assert out.shape == signal.shape
    err = 20 * np.log10(np.sqrt(((out - signal) ** 2).mean()) + 1e-20)
    assert err < -60, f"reconstruction error {err:.1f} dBFS"
