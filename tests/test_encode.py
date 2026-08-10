from m4b_lib import encode, ffmpeg_utils, timeline
from m4b_lib.streams import plan_streams


def test_chapter_table_survives_encode_and_mux(noisy_wav_48k, tmp_path):
    frames = 8 * 48000
    chapters = [
        timeline.Chapter(0.0, 4.0, "One"),
        timeline.Chapter(4.0, 8.0, "Two = odd; title#"),
    ]
    specs = plan_streams(frames, 2)
    parts = encode.encode_streams(str(noisy_wav_48k), specs, gain_db=0.0,
                                  sample_rate=48000, workdir=str(tmp_path), workers=2)
    assert len(parts) == 2

    out = tmp_path / "out.m4b"
    encode.concat_and_mux(parts, chapters, str(out), title="T", author="A")

    got = ffmpeg_utils.probe_chapters(str(out))
    assert [c["title"] for c in got] == ["One", "Two = odd; title#"]
    assert abs(got[0]["start"] - 0.0) < 0.05
    assert abs(got[-1]["end"] - 8.0) < 0.5
    assert abs(ffmpeg_utils.get_duration(str(out)) - 8.0) < 0.5


def test_chapter_title_with_backslash_survives_escaping(noisy_wav_48k, tmp_path):
    """A raw (unescaped) backslash is silently consumed by ffmpeg's ffmetadata
    parser as an escape prefix — e.g. "back\\slash" round-trips as "backslash"
    if the backslash itself isn't escaped first. `=`/`;`/`#` mid-value don't
    actually corrupt the round trip even unescaped (only comment-leading ;/#
    and backslash do), so this is the case that would catch escaping being
    silently dropped.
    """
    frames = 8 * 48000
    chapters = [timeline.Chapter(0.0, 8.0, r"back\slash and \; combo")]
    specs = plan_streams(frames, 1)
    parts = encode.encode_streams(str(noisy_wav_48k), specs, gain_db=0.0,
                                  sample_rate=48000, workdir=str(tmp_path), workers=1)

    out = tmp_path / "out.m4b"
    encode.concat_and_mux(parts, chapters, str(out))

    got = ffmpeg_utils.probe_chapters(str(out))
    assert got[0]["title"] == r"back\slash and \; combo"


def test_gain_is_applied(noisy_wav_48k, tmp_path):
    specs = plan_streams(8 * 48000, 1)
    quiet = encode.encode_streams(str(noisy_wav_48k), specs, gain_db=-20.0,
                                  sample_rate=48000, workdir=str(tmp_path / "q"), workers=1)
    loud = encode.encode_streams(str(noisy_wav_48k), specs, gain_db=0.0,
                                 sample_rate=48000, workdir=str(tmp_path / "l"), workers=1)
    assert ffmpeg_utils.probe_noise_floor(quiet[0]) < ffmpeg_utils.probe_noise_floor(loud[0]) - 10


def test_default_encode_settings_leave_no_join_dropouts(tmp_path):
    """A continuous tone must come out continuous at the default worker count.

    Every AAC encoder emits priming samples and pads its final frame, so
    encoding N streams separately and stitching them leaves a ~35ms, -40 dB
    dip at each of the N-1 joins — audible in narration, and invisible to
    duration checks because the container duration is unchanged.

    This asserts on content rather than length, which is the whole point: the
    defect this guards against cannot change the reported duration, so every
    shape/length assertion in the suite passed while it was present. Raising
    the encode default back above 1 fails this test.
    """
    import subprocess

    import numpy as np
    import soundfile as sf

    from m4b_lib import scheduler, timeline

    src = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=30:sample_rate=48000",
         "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(src)],
        check=True,
    )
    frames, sr = sf.info(str(src)).frames, 48000

    specs = plan_streams(frames, scheduler.default_workers("encode"))
    parts = encode.encode_streams(str(src), specs, gain_db=0.0, sample_rate=sr,
                                  workdir=str(tmp_path / "parts"),
                                  workers=scheduler.default_workers("encode"))
    out = tmp_path / "joined.m4b"
    encode.concat_and_mux(parts, [timeline.Chapter(0.0, frames / sr, "One")], str(out))

    back = tmp_path / "back.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-i", str(out),
         "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(back)],
        check=True,
    )
    y, _ = sf.read(str(back), dtype="float32")

    w = 256
    rms = np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same"))
    guard = sr // 2                      # ignore the encoder's own head/tail ramp
    interior = rms[guard:len(rms) - guard]
    nominal = float(np.median(rms))
    worst = float(interior.min())

    assert worst > nominal * 0.5, (
        f"level drops to {worst / nominal:.1%} of nominal inside a continuous "
        f"tone ({20 * np.log10(max(worst / nominal, 1e-9)):.1f} dB) — encode "
        f"joins are being introduced at {len(specs)} streams"
    )
