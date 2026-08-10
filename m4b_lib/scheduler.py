"""Per-stage concurrency for the enhancement pass.

Measured on a 24-core box: DF3 saturates at 8-12 workers (5.7x at 8, 7.2x at 24
with each worker 3x slower), loudnorm peaks at 8 and is worse at 24, AAC scales
to 11.7x. So worker counts are per-stage, not one global --jobs.

CPU parallelism is by process, because a worker must own torch's thread setting
and the model's hidden state. CUDA work is serial, batch 1 -- `_run_stream` builds B=1 and GPU jobs
run one at a time. An earlier version of this comment claimed batching that
the code has never done (F-31); the GPU never sees a batch > 1.
"""
import multiprocessing as mp
import os

import soundfile as sf

from m4b_lib.enhance import EnhanceConfig, get_backend
from m4b_lib.enhance import df3 as _df3  # noqa: F401
from m4b_lib.streams import overlap_add, plan_chunks, plan_streams

# The df3 import above is load-bearing, not tidiness: worker processes use the
# "spawn" start method, so a child re-imports this module and must find the
# backend already registered. It imports torch only inside its methods, so
# importing it here stays cheap and safe when the ml extra is absent.
#
# The m4bnet backend was registered here too until 2026-08-10. It lived on the
# trained-model side and is now in m4binder-research; it never had a shipped
# checkpoint, and stock DeepFilterNet3 beat every model trained for it.

# "encode" is deliberately 1, not the 24 its throughput would justify. Every AAC
# encoder emits priming samples and pads its final frame to a 1024-sample
# boundary, so encoding N streams separately and stitching them with the concat
# demuxer leaves a -40 dB dip roughly 35ms wide at each of the N-1 joins. That
# is audible in narration — confirmed by listening, not just measured — and
# invisible to every duration-based guard, because the container duration is
# unchanged. Trimming it needs sample-accurate cuts, which an encoded AAC stream
# cannot express (frame granularity is ~21ms); doing it properly means building
# the MP4 sample tables by hand.
#
# A single continuous encode is correct by construction and is what the tool did
# before this pipeline existed. It costs wall clock — roughly 55 min rather than
# 5 on a 30h book — which is the right trade for output with no audible defects.
# The parallel path below still works and is still tested; raise --jobs-encode to
# opt back in once a gapless join exists.
_STAGE_DEFAULTS = {"df3": 12, "loudness": 8, "encode": 1, "decode": 4}


def default_workers(stage: str) -> int:
    """Measured optimum per stage, capped by the machine's core count.

    Except "encode", which is capped at 1 for correctness rather than speed —
    see the comment on _STAGE_DEFAULTS.
    """
    want = _STAGE_DEFAULTS.get(stage, 8)
    return max(1, min(want, os.cpu_count() or 1))


def _worker_threads() -> int:
    """Torch threads per worker. Always 1 — see module docstring."""
    return 1


def _run_stream(args) -> str:
    index, src, out_path, start, end, backend_name, cfg, device, chunk, overlap = args
    import torch

    torch.set_num_threads(_worker_threads())
    stream_len = end - start
    be = get_backend(backend_name, cfg)
    try:
        be.load(device)
        sr = sf.info(src).samplerate
        pieces = []
        # Read one chunk (plus overlap) at a time, not the whole stream: a
        # worker's live buffer must scale with chunk_s, not stream length, or
        # a 30-hour book pushes ~10GB total into RAM across the pool.
        for a, b in plan_chunks(stream_len, chunk, overlap):
            data, _ = sf.read(src, dtype="float32", start=start + a, stop=start + b,
                              always_2d=False)
            x = torch.from_numpy(data).unsqueeze(0)
            out, _ = be.enhance_batch(x)
            pieces.append(out.squeeze(0).cpu().numpy())
        joined = overlap_add(pieces, overlap)
        if len(joined) != stream_len:
            raise RuntimeError(
                f"stream {index} [{start}:{end}] length drift: {len(joined)} != {stream_len}"
            )
        sf.write(out_path, joined, sr, subtype="PCM_16", format="W64")
    except Exception as e:
        raise RuntimeError(f"stream {index} [{start}:{end}] ({out_path}) failed: {e}") from e
    finally:
        be.close()
    return out_path


def open_timeline_writer(path: str, sr: int):
    """Open a book-length mono PCM_16 output file.

    format="W64" is load-bearing, not cosmetic, and this is a named function so
    a test can assert on it. soundfile infers the container from the filename,
    and a .wav is RIFF, whose 32-bit size fields cap the file at 4 GiB --
    exactly 44739.24 s of mono 16-bit 48 kHz audio, or 12.43 hours. The enhanced
    timeline holds the whole book, so every longer book was truncated here: a
    12.7 h source and a 24.2 h source both produced exactly 44739.3 s of output.
    """
    return sf.SoundFile(path, mode="w", samplerate=sr, channels=1,
                        subtype="PCM_16", format="W64")


def enhance_timeline(tl, out_wav: str, backend: str = "df3",
                     cfg: EnhanceConfig | None = None, device: str = "cpu",
                     workers: int = 12, chunk_s: float = 60.0,
                     overlap_s: float = 2.0) -> str:
    cfg = cfg or EnhanceConfig()
    sr = tl.sample_rate
    chunk, overlap = int(chunk_s * sr), int(overlap_s * sr)
    workdir = os.path.dirname(os.path.abspath(out_wav))
    os.makedirs(workdir, exist_ok=True)

    specs = plan_streams(tl.frames, workers)
    jobs = [
        (s.index, tl.wav_path, os.path.join(workdir, f"enh_{s.index:05d}.wav"),
         s.start_frame, s.end_frame, backend, cfg, device, chunk, overlap)
        for s in specs
    ]
    part_paths = [j[2] for j in jobs]

    try:
        if device != "cpu" or len(jobs) == 1:
            parts = [_run_stream(j) for j in jobs]
        else:
            ctx = mp.get_context("spawn")  # never fork with threads already running
            with ctx.Pool(len(jobs)) as pool:
                parts = pool.map(_run_stream, jobs)

        with open_timeline_writer(out_wav, sr) as dst:
            for p in parts:
                dst.write(sf.read(p, dtype="float32")[0])

        written = sf.info(out_wav).frames
        if written != tl.frames:
            raise RuntimeError(f"enhanced timeline is {written} frames, expected {tl.frames}")
    finally:
        # A failed job leaves its siblings' completed parts on disk; clean up
        # on every exit path (success or failure) so a long unattended run
        # can't orphan gigabytes of partial per-stream wavs.
        for p in part_paths:
            if os.path.exists(p):
                os.unlink(p)

    return out_wav
