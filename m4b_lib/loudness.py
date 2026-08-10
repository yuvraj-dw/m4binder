"""EBU R128 loudness, measured per stream and combined.

The old pipeline ran ffmpeg's two-pass loudnorm over the whole book: pass 1 is
the worst-scaling stage in the pipeline (measured 45x realtime single, peaking
at 107x with 8 workers and getting *worse* at 24). Measuring per stream and
combining lets pass 1 run concurrently.

With linear=true the second pass is a constant gain, so it is not a pass at all
here — gain_db() is folded into the encode step.
"""
import json
import math
import re
import subprocess

from m4b_lib.ffmpeg_utils import duration_timeout

TARGET_I = -19.0
TARGET_TP = -2.0
TARGET_LRA = 7.0


def analyze(wav: str, start_frame: int, end_frame: int, sample_rate: int) -> dict:
    """Run loudnorm's measurement pass over one stream of `wav`."""
    ss = start_frame / sample_rate
    dur = (end_frame - start_frame) / sample_rate
    r = subprocess.run(
        ["ffmpeg", "-v", "info", "-ss", f"{ss:.6f}", "-t", f"{dur:.6f}", "-i", wav,
         "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
                ":dual_mono=true:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=duration_timeout(dur),
    )
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', r.stderr, re.DOTALL)
    if not m:
        raise RuntimeError(f"loudnorm produced no JSON for {wav} [{ss:.1f}s +{dur:.1f}s]")
    stats = json.loads(m.group(0))
    return {
        "input_i": float(stats["input_i"]),
        "input_tp": float(stats["input_tp"]),
        "input_lra": float(stats["input_lra"]),
        "input_thresh": float(stats["input_thresh"]),
        "frames": end_frame - start_frame,
    }


def combine(measurements: list[dict]) -> dict:
    """Duration-weighted energy mean of per-stream loudness.

    Integrated loudness is a mean of gated block powers, so combining in the
    power domain weighted by duration is exact up to gating differences at
    stream boundaries — well inside the 1 LU that matters here. This is an
    approximation, not an identity: it is not a re-derivation of R128's own
    gating over the joined signal.

    `input_lra` and `input_thresh` are combined with max()/min(), which is
    NOT similarly justified: loudness range is a statistic over the whole
    signal's distribution of block powers, and concatenating segments with
    different mean loudness can produce a combined LRA *larger* than either
    segment's own, so max() can underestimate it. Nothing here consumes
    these two fields (gain_db reads only input_i/input_tp) — treat them as
    rough, not authoritative, if a future caller needs them.
    """
    if not measurements:
        raise ValueError("combine() needs at least one measurement")
    total = sum(m["frames"] for m in measurements) or 1
    power = sum(10 ** (m["input_i"] / 10) * m["frames"] for m in measurements) / total
    return {
        "input_i": 10 * math.log10(power) if power > 0 else -70.0,
        "input_tp": max(m["input_tp"] for m in measurements),
        "input_lra": max(m["input_lra"] for m in measurements),
        "input_thresh": min(m["input_thresh"] for m in measurements),
        "frames": total,
    }


def gain_db(combined: dict, target_i: float = TARGET_I) -> float:
    """Constant gain to reach target_i, clamped so true peak stays under TARGET_TP.

    A quiet-but-peaky recording must not be pushed into clipping just because
    its integrated loudness is low: the gain suggested by loudness alone is
    capped by whatever true-peak headroom is actually available.
    """
    wanted = target_i - float(combined["input_i"])
    headroom = TARGET_TP - float(combined["input_tp"])
    return min(wanted, headroom)
