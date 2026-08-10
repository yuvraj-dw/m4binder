import pytest
from m4b_lib.enhance import get_backend, register, available_backends
from m4b_lib.enhance.base import EnhanceConfig


def test_registry_returns_registered_backend():
    @register("dummy")
    class Dummy:
        name = "dummy"
        sample_rate = 48000
        def __init__(self, cfg): self.cfg = cfg
        def load(self, device): self.device = device
        def enhance_batch(self, x, state=None): return x, None
        def close(self): pass

    assert "dummy" in available_backends()
    b = get_backend("dummy", EnhanceConfig(atten_lim_db=12.0))
    assert b.name == "dummy"
    assert b.cfg.atten_lim_db == 12.0


def test_unknown_backend_raises():
    with pytest.raises(KeyError, match="nosuch"):
        get_backend("nosuch", EnhanceConfig())


import numpy as np

pytestmark = pytest.mark.ml  # every test below needs torch + deepfilternet

import torch
import soundfile as sf
from m4b_lib.enhance import df3 as _df3  # noqa: F401  (registers the backend)


@pytest.fixture(scope="module")
def df3_backend():
    b = get_backend("df3", EnhanceConfig(atten_lim_db=12.0))
    b.load("cpu")
    yield b
    b.close()


def _chunks(path, n, seconds):
    data, sr = sf.read(str(path), dtype="float32")
    assert sr == 48000
    step = int(seconds * sr)
    return torch.stack([torch.from_numpy(data[i * step:(i + 1) * step].copy()) for i in range(n)])


def test_length_is_preserved(df3_backend, noisy_wav_48k):
    x = _chunks(noisy_wav_48k, 2, 3.0)
    out, _ = df3_backend.enhance_batch(x)
    assert out.shape == x.shape


def test_batched_matches_per_stream(df3_backend, noisy_wav_48k):
    x = _chunks(noisy_wav_48k, 2, 3.0)
    batched, _ = df3_backend.enhance_batch(x)
    solo = torch.cat([df3_backend.enhance_batch(x[i:i + 1])[0] for i in range(x.shape[0])])
    resid = (batched - solo).pow(2).mean().sqrt().item()
    db = 20 * np.log10(resid + 1e-20)
    assert db < -120, f"batched != per-stream: {db:.1f} dBFS"


def test_atten_lim_zero_is_passthrough(noisy_wav_48k):
    """Rust CLI semantics: 0 dB means no processing. torch's enhance() would
    read 0 as 'no limit' and apply FULL noise reduction — the adapter must not."""
    b = get_backend("df3", EnhanceConfig(atten_lim_db=0.0))
    b.load("cpu")
    try:
        x = _chunks(noisy_wav_48k, 1, 3.0)
        out, _ = b.enhance_batch(x)
        assert torch.equal(out, x)
    finally:
        b.close()


def test_enhance_batch_before_load_raises():
    b = get_backend("df3", EnhanceConfig(atten_lim_db=12.0))
    with pytest.raises(RuntimeError, match="load"):
        b.enhance_batch(torch.zeros(1, 100))


def test_enhance_batch_wrong_ndim_raises(df3_backend, noisy_wav_48k):
    x = _chunks(noisy_wav_48k, 1, 3.0)[0]  # 1-D: [T], not [B, T]
    with pytest.raises(ValueError, match=r"expected \[B, T\]"):
        df3_backend.enhance_batch(x)


def test_length_contract_violation_raises(df3_backend, noisy_wav_48k, monkeypatch):
    """If the underlying backend ever returns a different length, we must raise
    rather than silently hand back mismatched audio."""
    import sys

    import df.enhance  # noqa: F401  (ensures df.enhance is in sys.modules)

    # df/__init__.py does `from .enhance import enhance`, which shadows the
    # `df.enhance` attribute with the function of the same name — so we have
    # to reach the real submodule via sys.modules rather than `df.enhance`.
    df_enhance_module = sys.modules["df.enhance"]
    monkeypatch.setattr(df_enhance_module, "enhance", lambda *a, **k: torch.zeros(1, 10))
    x = _chunks(noisy_wav_48k, 1, 3.0)
    with pytest.raises(RuntimeError, match="length contract"):
        df3_backend.enhance_batch(x)
