"""The shim that lets DeepFilterNet import under modern torchaudio.

`df/io.py` does `from torchaudio.backend.common import AudioMetaData`. torchaudio
2.6 still ships that module; 2.11 removed it entirely. The training box runs
2.11 because the RTX 5090 needs torch >= 2.7 for sm_120, so without this shim the
DF3 baseline cannot be loaded on the only machine that can train against it.

These tests simulate the 2.11 situation on whatever version is installed, by
blocking the import rather than by requiring a particular torchaudio.
"""
import importlib.abc
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from m4b_lib import torchaudio_compat  # noqa: E402

pytestmark = pytest.mark.ml

BLOCKED = "torchaudio.backend.common"


class _Blocker(importlib.abc.MetaPathFinder):
    """Makes one module name un-importable, as if the release had removed it."""

    def __init__(self, name):
        self.name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            raise ImportError(f"simulated removal of {fullname}")
        return None


@pytest.fixture
def without_torchaudio_backend():
    import torchaudio

    saved = {k: sys.modules[k] for k in list(sys.modules)
             if k.startswith("torchaudio.backend")}
    saved_attr = getattr(torchaudio, "backend", None)
    for k in saved:
        del sys.modules[k]
    if hasattr(torchaudio, "backend"):
        del torchaudio.backend
    blocker = _Blocker(BLOCKED)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)
        if saved_attr is not None:
            torchaudio.backend = saved_attr


def test_the_fixture_really_breaks_the_import(without_torchaudio_backend):
    """Guards the test itself: if the simulation does not bite, everything
    below would pass vacuously on a torchaudio that still ships the module."""
    with pytest.raises(ImportError):
        __import__(BLOCKED)


def test_install_reconstructs_the_removed_module(without_torchaudio_backend):
    assert torchaudio_compat.install() is True
    from torchaudio.backend.common import AudioMetaData
    assert AudioMetaData is not None


def test_the_symbol_has_the_fields_deepfilternet_expects(without_torchaudio_backend):
    """Constructed positionally, because install() prefers torchaudio's own
    AudioMetaData when it still exists under a new name, and that class has no
    defaults. The fallback dataclass has to accept the same call."""
    torchaudio_compat.install()
    from torchaudio.backend.common import AudioMetaData
    meta = AudioMetaData(48000, 100, 1, 16, "PCM_S")
    assert meta.sample_rate == 48000
    for field in ("sample_rate", "num_frames", "num_channels",
                  "bits_per_sample", "encoding"):
        assert hasattr(meta, field), field


def test_the_fallback_dataclass_accepts_the_same_call():
    """The path taken on torchaudio 2.11, where nothing of the old type
    survives and our stand-in is what df gets."""
    meta = torchaudio_compat.AudioMetaData(48000, 100, 1, 16, "PCM_S")
    assert (meta.sample_rate, meta.num_channels, meta.encoding) == (48000, 1, "PCM_S")


def test_install_is_idempotent(without_torchaudio_backend):
    assert torchaudio_compat.install() is True
    assert torchaudio_compat.install() is False


def test_install_is_a_no_op_when_torchaudio_still_has_it():
    """On torchaudio 2.6 nothing should be touched."""
    try:
        import torchaudio.backend.common  # noqa: F401
    except Exception:
        pytest.skip("this torchaudio has already removed the module")
    assert torchaudio_compat.install() is False


def test_after_install_the_symbol_is_importable_either_way():
    """The postcondition callers actually depend on, version-independent."""
    torchaudio_compat.install()
    from torchaudio.backend.common import AudioMetaData  # noqa: F401


def test_df3_backend_installs_the_shim_before_importing_df(monkeypatch):
    """The regression this move exists for: the shipped backend must install the
    shim itself. Without it, `clean --backend df3` cannot load on the training
    box at all, which would only surface when comparing against the baseline."""
    from m4b_lib import torchaudio_compat as tc
    from m4b_lib.enhance import EnhanceConfig, get_backend
    from m4b_lib.enhance import df3 as _df3  # noqa: F401  registers "df3"

    calls = []
    monkeypatch.setattr(tc, "install", lambda: calls.append(1) or False)
    be = get_backend("df3", EnhanceConfig())
    try:
        be.load("cpu")
    except Exception:
        pass          # model download or df internals may fail; the call is the point
    assert calls, "df3.load() did not install the torchaudio shim"
