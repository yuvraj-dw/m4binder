"""DeepFilterNet3 adapter.

DF3 batches natively: df.enhance.enhance() treats dim 0 of a [B, T] tensor as
the batch dimension and calls reset_h0(batch_size=B) itself. Verified during
design: batched output matches per-stream output at -157 dBFS.

atten_lim semantics differ between the deep-filter CLI and this Python API.
The CLI treats 0 as passthrough; enhance() guards with `abs(atten_lim_db) > 0`
and so treats 0 as *no limit*, i.e. maximum noise reduction. We implement the
CLI's meaning because that is what m4binder's --atten-lim has always meant.

`cfg.pf` is forwarded to init_df's `post_filter` kwarg.

**Device handling, and it is not the obvious one.** `load()` steers DF3 by setting
its `DEVICE` config key, because `enhance()` consults `get_device()` itself for
the feature tensors and for `reset_h0` — moving only the model leaves those on the
other device. The input tensor deliberately stays on the CPU whatever the device:
`enhance()` runs the Rust analysis on the host, so a CUDA tensor there is a
TypeError, not a slow path. Both directions were broken until 2026-07-29 and are
now measured (F-29): GPU and CPU outputs agree at -156.1 dBFS.
"""
from typing import Any

from m4b_lib.enhance import register
from m4b_lib.enhance.base import EnhanceConfig


@register("df3")
class DF3Enhancer:
    name = "df3"
    sample_rate = 48000

    def __init__(self, cfg: EnhanceConfig):
        self.cfg = cfg
        self._model = None
        self._state = None
        self._device = "cpu"

    def load(self, device: str = "cpu") -> None:
        import torch
        # Mandatory: torch's default all-core threading is 22x slower than 4
        # threads on this model. Workers are the unit of parallelism, not threads.
        torch.set_num_threads(1)
        # Must precede the first `df` import. df/io.py imports a torchaudio
        # symbol that 2.11 removed, and torch >= 2.7 is required for sm_120, so
        # on any CUDA-capable modern box this import fails without the shim.
        from m4b_lib.torchaudio_compat import install as _install_torchaudio_shim

        _install_torchaudio_shim()
        from df.enhance import init_df

        self._device = device
        self._model, self._state, _ = init_df(
            log_file=None, log_level="ERROR", post_filter=self.cfg.pf,
        )
        # init_df has already done `model.to(get_device())`, and get_device()
        # auto-selects cuda:0 whenever one is visible. Asking for "cpu" therefore
        # got us a GPU model, silently: 12 workers each built a CUDA context and
        # a ~0.66 GB working set, and an 8 GB card died partway through a book
        # with an out-of-memory error naming a device the caller never asked for.
        #
        # Overriding the config key rather than the model is deliberate.
        # enhance() calls get_device() itself for the feature tensors and for
        # reset_h0 (df/enhance.py:199-201, 226, 234), so moving only the model
        # leaves the features on the other device and trades an OOM for a
        # mismatch. The key is read per call, so setting it steers every one of
        # those sites at once.
        from df.config import config as _df_config

        _df_config.set("DEVICE", device, str, section="train")
        self._model = self._model.to(device)

    def enhance_batch(self, x: Any, state: Any | None = None) -> tuple[Any, Any]:
        """x: [B, T] float32 at 48 kHz. Returns ([B, T], None)."""
        import torch
        from df.enhance import enhance

        if self._model is None:
            raise RuntimeError("DF3Enhancer.load() must be called before enhance_batch()")
        if x.ndim != 2:
            raise ValueError(f"expected [B, T], got shape {tuple(x.shape)}")

        if self.cfg.atten_lim_db == 0:
            return x.clone(), None

        # The input stays on the CPU whatever the device. enhance() runs the
        # Rust analysis on the host -- `df.analysis(audio.numpy())`,
        # df/enhance.py:190 -- and a CUDA tensor there is a TypeError, not a
        # slow path. DF3 moves the *features* to the device itself. Passing
        # --device cuda used to crash here for exactly this reason.
        with torch.no_grad():
            out = enhance(
                self._model, self._state, x,
                pad=True, atten_lim_db=float(self.cfg.atten_lim_db),
            )
        out = out.cpu()
        if out.shape != x.shape:
            raise RuntimeError(f"backend broke the length contract: {tuple(out.shape)} != {tuple(x.shape)}")
        return out, None

    def close(self) -> None:
        self._model = None
        self._state = None
