"""Let DeepFilterNet import under modern torchaudio.

`df/io.py` does `from torchaudio.backend.common import AudioMetaData`. That
module was deprecated and then removed: torchaudio 2.6 still ships it as a shim,
2.11 has neither `torchaudio.backend` nor `torchaudio.AudioMetaData`.

The alternative would be pinning torchaudio to an old release, but torch and
torchaudio are version-locked and the RTX 5090 needs torch >= 2.7 for sm_120
kernels — so pinning torchaudio back would mean giving up the GPU. This
reconstructs the one symbol DeepFilterNet actually wants.

Call `install()` BEFORE importing anything from `df`.
"""
import sys
import types
from dataclasses import dataclass


@dataclass
class AudioMetaData:
    """Stand-in for the removed torchaudio type. DeepFilterNet uses it only as
    a return annotation on its metadata helper, so the fields just need to
    exist with the same names."""
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = ""


def install() -> bool:
    """Return True if a shim was installed, False if none was needed."""
    try:
        import torchaudio.backend.common  # noqa: F401
        return False
    except Exception:
        pass

    import torchaudio

    meta = getattr(torchaudio, "AudioMetaData", None) or AudioMetaData

    backend = sys.modules.get("torchaudio.backend")
    if backend is None:
        backend = types.ModuleType("torchaudio.backend")
        sys.modules["torchaudio.backend"] = backend
        torchaudio.backend = backend

    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = meta
    sys.modules["torchaudio.backend.common"] = common
    backend.common = common
    return True
