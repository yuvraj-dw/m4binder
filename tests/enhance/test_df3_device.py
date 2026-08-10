"""`--device cpu` must actually mean the CPU.

`init_df()` ends with `model.to(get_device())`, and `df.utils.get_device()`
auto-selects `cuda:0` whenever one is visible. So on any CUDA-capable box the
project's own default invocation -- `m4binder.py:59` defaults `--device` to
`cpu` -- loaded a GPU model. It was masked only because the project venv holds
`torch==2.6.0+cpu`, while `requirements-ml.txt:6` pins bare `torch==2.6.0`,
which on Linux PyPI *is* the CUDA build. A fresh install following the project's
own ML instructions therefore produced a default run that OOMed partway through
a book: 12 workers, ~0.66 GB each, on an 8 GB card.

These tests run on any box, with or without a GPU, because the defect is in
which device is *requested*, not in whether one exists.
"""
import pytest

torch = pytest.importorskip("torch")

from m4b_lib.enhance.base import EnhanceConfig  # noqa: E402
from m4b_lib.enhance.df3 import DF3Enhancer     # noqa: E402


@pytest.fixture(scope="module")
def loaded_cpu():
    enh = DF3Enhancer(EnhanceConfig(atten_lim_db=12))
    enh.load("cpu")
    return enh


def test_model_parameters_are_on_the_requested_device(loaded_cpu):
    """The regression proper: every parameter must be where the caller asked.

    Before the fix this returned {'cuda:0'} on a CUDA-visible box while the
    caller had asked for 'cpu'.
    """
    devices = {p.device.type for p in loaded_cpu._model.parameters()}
    assert devices == {"cpu"}, f"asked for cpu, model is on {devices}"


def test_df_reports_the_requested_device(loaded_cpu):
    """enhance() consults get_device() for features and reset_h0, not the model.

    Moving only the model would leave those tensors on the other device, so this
    asserts the authority DF3 actually reads.
    """
    from df.utils import get_device

    assert get_device().type == "cpu"


def test_the_device_was_set_explicitly_not_inherited(loaded_cpu):
    """Distinguishes the fix from the absence of a GPU.

    On a CPU-only box -- which this repo's venv is, holding torch+cpu -- the two
    checks above pass whether or not `load()` sets anything, because
    `get_device()` falls through to "cpu" on its own. That would make them
    green here and useless on the box the bug actually bites.

    `get_device()` reads `config("DEVICE", ..., section="train")` and only
    auto-selects when that key is empty. Asserting the key is populated proves
    the requested device was recorded rather than coincidentally matched.
    """
    from df.config import config as df_config

    assert df_config("DEVICE", "", str, section="train") == "cpu"


def test_enhance_runs_and_preserves_length(loaded_cpu):
    """End-to-end guard: a device mismatch surfaces here as a RuntimeError.

    Not a quality check -- it asserts the contract enhance_batch documents, so a
    future device change cannot pass the two checks above while breaking the
    call they exist to protect.
    """
    x = torch.zeros(1, 48000, dtype=torch.float32)
    x[0, ::100] = 0.1
    out, _ = loaded_cpu.enhance_batch(x)
    assert out.shape == x.shape
    assert out.device.type == "cpu"
    assert torch.isfinite(out).all()
