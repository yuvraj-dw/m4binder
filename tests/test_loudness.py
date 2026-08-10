import soundfile as sf
from m4b_lib import loudness


def test_combine_matches_whole_file_analysis(noisy_wav_48k):
    """Per-stream analysis, energy-combined, must track a single whole-file pass."""
    frames = sf.info(str(noisy_wav_48k)).frames
    whole = loudness.analyze(str(noisy_wav_48k), 0, frames, 48000)

    half = frames // 2
    parts = [
        loudness.analyze(str(noisy_wav_48k), 0, half, 48000),
        loudness.analyze(str(noisy_wav_48k), half, frames, 48000),
    ]
    combined = loudness.combine(parts)
    assert abs(combined["input_i"] - whole["input_i"]) < 1.0


def test_gain_db_moves_measured_to_target():
    g = loudness.gain_db({"input_i": -25.0, "input_tp": -8.0}, target_i=-19.0)
    assert abs(g - 6.0) < 1e-6


def test_gain_is_capped_by_true_peak_headroom():
    """A peaky input must be attenuated, not amplified, when it already exceeds TARGET_TP.

    A stream measuring -25 LUFS wants +6 dB to reach -19, but its true peak is
    -1 dBTP against a -2 dBTP ceiling — already 1 dB over. The cap must pull it
    down by 1 dB, landing output true peak exactly on the ceiling.
    """
    g = loudness.gain_db({"input_i": -25.0, "input_tp": -1.0}, target_i=-19.0)
    assert abs(g - (-1.0)) < 1e-6


def test_gain_cap_allows_headroom_not_just_forbids_excess():
    """A peak 1 dB *under* the ceiling should still permit that 1 dB of gain.

    This discriminates the correct signed formula (TARGET_TP - input_tp) from
    a plausible-but-wrong abs-difference variant: the abs variant matches the
    -1 dBTP case above by coincidence but forbids gain here, even though the
    stream demonstrably has a dB of headroom before the ceiling.
    """
    g = loudness.gain_db({"input_i": -25.0, "input_tp": -3.0}, target_i=-19.0)
    assert abs(g - 1.0) < 1e-6
