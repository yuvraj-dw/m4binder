"""Enhancer backend registry."""
from m4b_lib.enhance.base import EnhanceConfig, Enhancer

_REGISTRY: dict = {}


def register(name: str):
    """Class decorator registering an Enhancer implementation under `name`."""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str, cfg: EnhanceConfig) -> Enhancer:
    if name not in _REGISTRY:
        raise KeyError(f"nosuch backend {name!r}; available: {available_backends()}")
    return _REGISTRY[name](cfg)


__all__ = ["EnhanceConfig", "Enhancer", "register", "get_backend", "available_backends"]
