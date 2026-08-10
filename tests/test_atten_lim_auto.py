"""Regression tests for `--atten-lim auto` (ffmpeg_utils.estimate_optimal_atten_lim).

The estimator has to be monotonic in the right direction: a NOISIER file needs a
LARGER attenuation limit (more noise reduction allowed), a cleaner file a smaller
one. It used to return `snr + margin`, which is the exact opposite -- clean files
got the aggressive setting and noisy files the timid one.

Probes are synthesised, not fixtures: speech-shaped bursts separated by gaps that
carry only the noise floor, so the estimator's silence-window path is exercised
end to end (ffmpeg silencedetect + volumedetect) rather than mocked.
"""
import math
import wave

import numpy as np
import pytest

from m4b_lib import ffmpeg_utils

SR = 16000
SPEECH_DBFS = -20.0


def _write_probe(path, snr_db, speech_s=1.5, gap_s=1.0, reps=6, seed=1234):
    """Speech bursts over a constant noise floor `snr_db` below the speech level."""
    rng = np.random.default_rng(seed)
    n_sp, n_gap = int(speech_s * SR), int(gap_s * SR)
    t = np.arange(n_sp) / SR
    amp = 10 ** (SPEECH_DBFS / 20.0) * math.sqrt(2)
    burst = amp * (
        0.7 * np.sin(2 * math.pi * 130 * t)
        + 0.2 * np.sin(2 * math.pi * 390 * t)
        + 0.1 * np.sin(2 * math.pi * 910 * t)
    )
    period = np.concatenate([burst, np.zeros(n_gap)])
    x = np.tile(period, reps)
    x = x + rng.normal(0.0, 10 ** ((SPEECH_DBFS - snr_db) / 20.0), size=x.size)
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return str(path)


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    d = tmp_path_factory.mktemp("atten_probes")
    return {
        snr: _write_probe(d / f"snr{snr}.wav", snr)
        for snr in (45, 30, 18)
    }


def test_estimator_measures_the_synthetic_snr(probes):
    """Guard: the probes must exercise the measurement path, not the fallback."""
    for snr, path in probes.items():
        _, info = ffmpeg_utils.estimate_optimal_atten_lim(path)
        assert info["snr"] is not None, f"no silence windows found for SNR {snr} probe"
        assert abs(info["snr"] - snr) < 10, (
            f"measured SNR {info['snr']:.1f} is nowhere near the synthesised {snr} dB "
            f"(noise_max={info['noise_max']}, overall={info['overall']})"
        )


def test_noisier_file_gets_a_larger_atten_lim(probes):
    """The core inversion bug: noisy must be treated harder than clean."""
    clean, _ = ffmpeg_utils.estimate_optimal_atten_lim(probes[45])
    noisy, _ = ffmpeg_utils.estimate_optimal_atten_lim(probes[18])
    assert noisy > clean, (
        f"noisy probe (18 dB SNR) got atten-lim {noisy} but clean probe "
        f"(45 dB SNR) got {clean} -- the estimator is inverted"
    )


def test_atten_lim_is_monotonic_across_three_noise_levels(probes):
    vals = {snr: ffmpeg_utils.estimate_optimal_atten_lim(p)[0] for snr, p in probes.items()}
    assert vals[18] >= vals[30] >= vals[45], f"not monotonic in noise: {vals}"
    assert vals[18] > vals[45], f"no separation between noisiest and cleanest: {vals}"


def test_atten_lim_stays_inside_the_documented_clamp(probes):
    for snr, path in probes.items():
        val, _ = ffmpeg_utils.estimate_optimal_atten_lim(path)
        assert 6.0 <= val <= 30.0, f"SNR {snr} probe produced out-of-range {val}"
