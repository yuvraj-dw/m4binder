"""DeepFilterNet 3 inference wrapper.

Loads DF3 model once (module-level) and exposes a function that processes a wav file.
GPU is auto-detected by the library; falls back to CPU.
"""
import torch
import torchaudio
from df.enhance import enhance, init_df


_model = None
_df_state = None
_target_sr = None


def _ensure_model():
    global _model, _df_state, _target_sr
    if _model is None:
        _model, _df_state, _ = init_df()
        _target_sr = _df_state.sr()


def df3_enhance(wav_in: str, wav_out: str) -> None:
    """Enhance the wav at wav_in and write to wav_out (mono, DF3-native sample rate)."""
    _ensure_model()
    audio, sr = torchaudio.load(wav_in)
    if sr != _target_sr:
        audio = torchaudio.transforms.Resample(sr, _target_sr)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)  # downmix to mono for DF3
    enhanced = enhance(_model, _df_state, audio)
    torchaudio.save(wav_out, enhanced, _target_sr)
