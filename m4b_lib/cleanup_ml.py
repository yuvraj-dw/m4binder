"""DeepFilterNet 3 inference wrapper - chunked and streaming for audiobook-length files.

Loads DF3 model once (module-level) and exposes a function that processes a wav file.
GPU is auto-detected by the library; falls back to CPU.

Chunked inference avoids OOM on long files: 60s chunks with 2s overlap crossfade.
For files already at target SR mono (as produced by cleanup pipeline), uses
streaming soundfile I/O to avoid loading entire file into RAM.
"""

import threading
from typing import Optional
import gc

try:
    import torch
    import torchaudio
    from df.enhance import enhance, init_df
except ImportError as _e:
    raise ImportError(
        "ML dependencies missing — run: pip install -r requirements-ml.txt ; "
        f"original error: {_e}"
    ) from _e


_model = None
_df_state = None
_target_sr = None
_lock = threading.Lock()
_resampler_cache: dict[tuple[int, int], torch.nn.Module] = {}


def _ensure_model(device_override: Optional[str] = None):
    global _model, _df_state, _target_sr
    with _lock:
        if _model is not None:
            return
        if device_override:
            try:
                from df.config import config
                config.set("device", device_override, config_path=None)
            except Exception:
                pass
        try:
            _model, _df_state, _ = init_df(log_file=None, log_level="WARNING")
        except TypeError:
            _model, _df_state, _ = init_df()
        _target_sr = _df_state.sr()


def _get_resampler(orig_sr: int, target_sr: int):
    key = (orig_sr, target_sr)
    with _lock:
        if key not in _resampler_cache:
            _resampler_cache[key] = torchaudio.transforms.Resample(orig_sr, target_sr)
        return _resampler_cache[key]


def _enhance_chunk(audio_chunk: torch.Tensor) -> torch.Tensor:
    # Lock inference to avoid race on model hidden state reset and GPU context
    with _lock:
        return enhance(_model, _df_state, audio_chunk)


def _validate_chunk_params(chunk_s: float, overlap_s: float):
    if overlap_s >= chunk_s:
        raise ValueError(f"overlap_s ({overlap_s}) must be < chunk_s ({chunk_s})")
    if chunk_s <= 0 or overlap_s < 0:
        raise ValueError("chunk_s must be >0 and overlap_s >=0")


def _df3_enhance_streaming(wav_in: str, wav_out: str, chunk_s: float, overlap_s: float):
    """Streaming implementation using soundfile — avoids loading full file.

    Assumes input is mono at target SR (as from cleanup pipeline). If not, falls back
    to full-load path.
    """
    import soundfile as sf
    import numpy as np

    _validate_chunk_params(chunk_s, overlap_s)
    chunk_len = int(chunk_s * _target_sr)
    overlap = int(overlap_s * _target_sr)
    hop = chunk_len - overlap

    info = sf.info(wav_in)
    sr = info.samplerate
    channels = info.channels
    total_frames = info.frames
    total_frames_target = int(total_frames * _target_sr / sr) if sr != _target_sr else total_frames

    # Streaming resampling per chunk if SR mismatch — filter discontinuity acceptable vs OOM
    needs_resample = sr != _target_sr
    resampler = _get_resampler(sr, _target_sr) if needs_resample else None
    needs_downmix = channels != 1

    # Streaming: read hop-sized raw windows, enhance chunk_len windows, overlap-add write
    with sf.SoundFile(wav_out, mode='w', samplerate=_target_sr, channels=1, subtype='PCM_16') as out_sf:
        with sf.SoundFile(wav_in, mode='r') as in_sf:
            pos = 0  # in original SR frames
            overlap_buf: Optional[torch.Tensor] = None

            # Precompute original SR chunk sizes for reading
            chunk_len_orig = int(chunk_len * sr / _target_sr) if needs_resample else chunk_len
            hop_orig = int(hop * sr / _target_sr) if needs_resample else hop

            while pos < total_frames:
                # Seek and read chunk_len_orig samples from pos
                in_sf.seek(pos)
                frames_to_read = min(chunk_len_orig, total_frames - pos)
                data = in_sf.read(frames_to_read, dtype='float32', always_2d=False)
                if data.ndim == 0 or data.size == 0:
                    break
                if data.ndim == 1:
                    audio_chunk = torch.from_numpy(data).unsqueeze(0)
                else:
                    audio_chunk = torch.from_numpy(data.mean(axis=1)).unsqueeze(0) if needs_downmix or channels != 1 else torch.from_numpy(data.T).mean(dim=0, keepdim=True) if data.shape[1] > 1 else torch.from_numpy(data).T

                # Resample if needed per chunk (filter discontinuity acceptable vs OOM)
                if needs_resample:
                    audio_chunk = resampler(audio_chunk)
                if needs_downmix and audio_chunk.shape[0] > 1:
                    audio_chunk = audio_chunk.mean(dim=0, keepdim=True)

                # Pad last chunk if very short (<0.5s) to avoid model issues
                needs_pad = False
                orig_chunk_len = audio_chunk.shape[-1]
                if orig_chunk_len < chunk_len and pos + chunk_len_orig >= total_frames:
                    if orig_chunk_len < _target_sr * 1.0 and total_frames_target > chunk_len:
                        pad_len = chunk_len - orig_chunk_len
                        audio_chunk = torch.nn.functional.pad(audio_chunk, (0, pad_len))
                        needs_pad = True

                with torch.no_grad():
                    enh = _enhance_chunk(audio_chunk)

                if needs_pad:
                    enh = enh[:, :orig_chunk_len]

                # First chunk: write hop, keep tail
                if overlap_buf is None:
                    # First chunk — single chunk file?
                    if total_frames <= chunk_len_orig:
                        out_np = enh.squeeze(0).cpu().numpy()
                        out_sf.write(out_np)
                        break
                    write_len = min(hop, enh.shape[-1])
                    out_np = enh[:, :write_len].squeeze(0).cpu().numpy()
                    out_sf.write(out_np)
                    if enh.shape[-1] > hop:
                        overlap_buf = enh[:, hop:hop+overlap].clone()
                    else:
                        overlap_buf = None
                else:
                    actual_overlap = min(overlap, overlap_buf.shape[-1], enh.shape[-1]) if overlap_buf is not None else 0
                    if actual_overlap > 0:
                        device = overlap_buf.device
                        t = torch.linspace(0, 1, actual_overlap, device=device)
                        import math
                        fade_out = torch.cos(t * math.pi / 2)
                        fade_in = torch.sin(t * math.pi / 2)
                        blended = overlap_buf[:, -actual_overlap:] * fade_out + enh[:, :actual_overlap] * fade_in
                        out_sf.write(blended.squeeze(0).cpu().numpy())
                        middle_end = min(hop, enh.shape[-1])
                        if middle_end > actual_overlap:
                            middle = enh[:, actual_overlap:middle_end]
                            out_sf.write(middle.squeeze(0).cpu().numpy())
                        if enh.shape[-1] > hop:
                            overlap_buf = enh[:, hop:hop+overlap].clone()
                        else:
                            remaining = enh.shape[-1] - middle_end
                            if remaining > 0 and pos + hop_orig >= total_frames:
                                if enh.shape[-1] > middle_end:
                                    tail = enh[:, middle_end:]
                                    out_sf.write(tail.squeeze(0).cpu().numpy())
                            overlap_buf = None
                    else:
                        write_len = min(hop, enh.shape[-1])
                        out_sf.write(enh[:, :write_len].squeeze(0).cpu().numpy())
                        if enh.shape[-1] > hop:
                            overlap_buf = enh[:, hop:hop+overlap].clone()

                pos += hop_orig

    written = out_sf.tell()
    if abs(written - total_frames_target) > max(10, total_frames_target * 0.01):
        print(f"  [warn] streaming output frames {written} != input target {total_frames_target} (diff {written-total_frames_target})")

    return True


def _df3_enhance_full_load(wav_in: str, wav_out: str, chunk_s: float, overlap_s: float):
    """Full-load but chunked inference (fallback when SR mismatch or soundfile not available)."""
    _validate_chunk_params(chunk_s, overlap_s)
    audio, sr = torchaudio.load(wav_in)
    dur_sec = audio.shape[-1] / sr if sr else 0
    if dur_sec > 1800:
        print(f"  [info] ML enhancing long file ({dur_sec/3600:.1f}h) in {chunk_s}s chunks (full-load fallback)")

    if sr != _target_sr:
        resampler = _get_resampler(sr, _target_sr)
        audio = resampler(audio)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    total_samples = audio.shape[-1]
    chunk_len = int(chunk_s * _target_sr)
    overlap = int(overlap_s * _target_sr)
    hop = chunk_len - overlap

    if total_samples <= chunk_len:
        with torch.no_grad():
            enhanced = _enhance_chunk(audio)
        torchaudio.save(wav_out, enhanced, _target_sr, encoding="PCM_S", bits_per_sample=16)
        return

    enhanced_chunks: list[torch.Tensor] = []
    pos = 0
    while pos < total_samples:
        end = min(pos + chunk_len, total_samples)
        chunk = audio[:, pos:end]
        needs_pad = False
        if chunk.shape[-1] < chunk_len and pos + chunk_len >= total_samples:
            if chunk.shape[-1] < _target_sr * 1.0 and total_samples > chunk_len:
                pad_len = chunk_len - chunk.shape[-1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                needs_pad = True

        with torch.no_grad():
            enh = _enhance_chunk(chunk)

        if needs_pad:
            enh = enh[:, : (end - pos)]

        if enhanced_chunks and overlap > 0 and pos > 0:
            prev = enhanced_chunks[-1]
            actual_overlap = min(overlap, prev.shape[-1], enh.shape[-1])
            if actual_overlap > 0:
                device = prev.device
                t = torch.linspace(0, 1, actual_overlap, device=device)
                import math
                fade_out = torch.cos(t * math.pi / 2)
                fade_in = torch.sin(t * math.pi / 2)
                # Ensure enh on same device for blending
                enh_same = enh.to(device) if enh.device != device else enh
                prev_tail = prev[:, -actual_overlap:] * fade_out + enh_same[:, :actual_overlap] * fade_in
                enhanced_chunks[-1] = torch.cat([prev[:, :-actual_overlap], prev_tail], dim=1)
                enhanced_chunks.append(enh_same[:, actual_overlap:])
            else:
                enhanced_chunks.append(enh)
        else:
            enhanced_chunks.append(enh)

        pos += hop

    enhanced_full = torch.cat(enhanced_chunks, dim=1)
    if enhanced_full.shape[-1] > total_samples:
        enhanced_full = enhanced_full[:, :total_samples]

    torchaudio.save(wav_out, enhanced_full, _target_sr, encoding="PCM_S", bits_per_sample=16)
    # Free large tensors
    del enhanced_chunks, enhanced_full
    gc.collect()


def df3_enhance(wav_in: str, wav_out: str, chunk_s: float = 60.0, overlap_s: float = 2.0) -> None:
    """Enhance wav file, chunked, with streaming when possible."""
    _ensure_model()
    _validate_chunk_params(chunk_s, overlap_s)

    # Try streaming path first (RAM efficient) — returns None if not applicable
    try:
        result = _df3_enhance_streaming(wav_in, wav_out, chunk_s, overlap_s)
        if result is True:
            return
    except Exception as e:
        print(f"  [warn] streaming enhance failed ({e}), falling back to full-load")

    # Fallback: full load but chunked
    _df3_enhance_full_load(wav_in, wav_out, chunk_s, overlap_s)


def unload_model():
    global _model, _df_state, _target_sr
    with _lock:
        if _model is not None:
            del _model
            _model = None
        if _df_state is not None:
            del _df_state
            _df_state = None
        _target_sr = None
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _resampler_cache.clear()
        gc.collect()
