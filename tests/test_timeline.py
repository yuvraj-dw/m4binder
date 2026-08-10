import numpy as np
import pytest
import soundfile as sf
from m4b_lib import timeline


def test_from_mp3s_marks_match_decoded_sample_counts(fixtures_dir, tmp_path):
    mp3s = sorted(str(p) for p in fixtures_dir.glob("*.mp3"))
    tl = timeline.from_mp3s(mp3s, str(tmp_path), sample_rate=48000, jobs=2)

    info = sf.info(tl.wav_path)
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.frames == tl.frames

    # Chapter marks must tile the decoded timeline exactly, with no drift.
    assert len(tl.chapters) == len(mp3s)
    assert tl.chapters[0].start == 0.0
    for a, b in zip(tl.chapters, tl.chapters[1:]):
        assert abs(a.end - b.start) < 1e-9
    assert abs(tl.chapters[-1].end - tl.frames / 48000) < 1e-9


def test_from_mp3s_uses_id3_titles(fixtures_dir, tmp_path):
    mp3s = sorted(str(p) for p in fixtures_dir.glob("*.mp3"))
    tl = timeline.from_mp3s(mp3s, str(tmp_path), jobs=2)
    assert [c.title for c in tl.chapters] == ["Chapter 1", "Chapter 2"]


def test_from_m4b_copies_source_chapters(fixture_m4b, tmp_path):
    tl = timeline.from_m4b(str(fixture_m4b), str(tmp_path))
    info = sf.info(tl.wav_path)
    assert info.channels == 1
    assert info.samplerate == 48000
    assert tl.frames == info.frames
    assert abs(tl.frames / 48000 - 4.0) < 0.5  # two 2s fixtures


def test_from_mp3s_empty_list_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="mp3_paths"):
        timeline.from_mp3s([], str(tmp_path))


def test_from_mp3s_all_zero_frames_falls_back_to_placeholder_chapter(fixtures_dir, tmp_path, monkeypatch):
    mp3s = sorted(str(p) for p in fixtures_dir.glob("*.mp3"))

    def _zero_decode(src, dst, sample_rate):
        sf.write(dst, np.zeros((0,), dtype="int16"), sample_rate, subtype="PCM_16")
        return 0

    monkeypatch.setattr(timeline, "_decode", _zero_decode)
    tl = timeline.from_mp3s(mp3s, str(tmp_path), sample_rate=48000, jobs=2)

    assert tl.frames == 0
    assert len(tl.chapters) == 1
    assert tl.chapters[0].title == "Chapter 1"
    assert tl.chapters[0].start == 0.0
    assert tl.chapters[0].end == 0.0
