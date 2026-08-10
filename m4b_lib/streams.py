"""Splitting a decoded timeline into uniform work units.

Streams are contiguous, equal-sized segments of the whole timeline. Chunks are
fixed-size windows within a stream, overlapping so the model's state reset at
each window boundary is crossfaded away.

Uniform chunks (rather than chapter-sized ones) remove the longest-chapter
floor: a single 83-minute chapter would otherwise bound wall clock no matter
how many workers are available.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StreamSpec:
    index: int
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


def plan_streams(frames: int, n: int) -> list[StreamSpec]:
    """Tile [0, frames) into at most n contiguous streams, no gaps or overlap."""
    if frames <= 0:
        return []
    n = max(1, min(n, frames))
    size = frames // n
    specs = []
    for i in range(n):
        start = i * size
        end = frames if i == n - 1 else (i + 1) * size
        if end > start:
            specs.append(StreamSpec(index=i, start_frame=start, end_frame=end))
    return specs


def plan_chunks(length: int, chunk: int, overlap: int) -> list[tuple[int, int]]:
    """Windows of `chunk` frames advancing by (chunk - overlap), clipped to length."""
    if length <= 0:
        return []
    if overlap >= chunk:
        raise ValueError(f"overlap {overlap} must be < chunk {chunk}")
    if length <= chunk:
        return [(0, length)]
    hop = chunk - overlap
    out = []
    pos = 0
    while pos < length:
        end = min(pos + chunk, length)
        out.append((pos, end))
        if end >= length:
            break
        pos += hop
    return out


def overlap_add(pieces: list[np.ndarray], overlap: int) -> np.ndarray:
    """Reassemble chunk outputs with a linear crossfade over `overlap` frames.

    Linear (equal-gain), not equal-power: the two sides of a join are the same
    underlying audio processed with slightly different context, i.e. highly
    correlated. Equal-power crossfade would sum correlated signal to +3 dB.
    """
    if not pieces:
        return np.zeros(0, dtype="float32")
    if len(pieces) == 1:
        return pieces[0].astype("float32", copy=True)

    out = pieces[0].astype("float32", copy=True)
    for nxt in pieces[1:]:
        nxt = nxt.astype("float32", copy=False)
        n = min(overlap, len(out), len(nxt))
        if n <= 0:
            out = np.concatenate([out, nxt])
            continue
        fade = np.linspace(0.0, 1.0, n, dtype="float32")
        out[-n:] = out[-n:] * (1.0 - fade) + nxt[:n] * fade
        out = np.concatenate([out, nxt[n:]])
    return out
