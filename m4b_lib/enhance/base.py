"""Backend-agnostic speech enhancement interface.

A backend is any object that can take [B, T] float32 mono audio at its own
native sample rate and return enhanced audio of exactly the same shape.
"""
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class EnhanceConfig:
    """Knobs common to all backends.

    atten_lim_db: noise attenuation in dB. 0 means passthrough (no processing),
        matching the deep-filter CLI. 12 is the tuned default; see spec 6a.
    pf: enable a backend's post-filter, if it has one. Off by default.
    """
    atten_lim_db: float = 12.0
    pf: bool = False


@runtime_checkable
class Enhancer(Protocol):
    name: str
    sample_rate: int

    def load(self, device: str) -> None: ...
    def enhance_batch(self, x: Any, state: Any | None = None) -> tuple[Any, Any]: ...
    def close(self) -> None: ...
