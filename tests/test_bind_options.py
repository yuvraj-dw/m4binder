# tests/test_bind_options.py
import json
import os
import subprocess
import shutil
from unittest.mock import patch

import pytest

from m4b_lib import bind


def _mp3s(tmp_path, n=2):
    folder = tmp_path / "in"
    folder.mkdir()
    for i in range(n):
        p = folder / f"part{i:02d}.mp3"
        p.write_bytes(b"\xff\xfb\x00" * 10)
    return str(folder)


def test_defaults_are_unchanged():
    opts = bind.BindOptions(input_folder="/x")
    assert opts.chapters_file is None
    assert opts.audio_mode == "transcode"


def test_supplied_chapters_file_is_used_instead_of_deriving(tmp_path):
    chapters = tmp_path / "c.ffmetadata"
    chapters.write_text(";FFMETADATA1\n")
    opts = bind.BindOptions(input_folder=_mp3s(tmp_path),
                            chapters_file=str(chapters))

    with patch.object(bind.ffmpeg_utils, "parallel_transcode",
                      side_effect=lambda mp3s, d, **kw: list(mp3s)), \
         patch.object(bind.ffmpeg_utils, "concat_audio_to_m4b"), \
         patch.object(bind.ffmpeg_utils, "create_chapters_ffmetadata") as derive, \
         patch.object(bind.ffmpeg_utils, "embed_chapters_and_meta") as embed:
        bind._bind_one(opts.input_folder, str(tmp_path / "out.m4b"), opts)

    derive.assert_not_called()
    assert embed.call_args.kwargs["chapters_ini"] == str(chapters)


def test_chapters_are_still_derived_when_none_supplied(tmp_path):
    opts = bind.BindOptions(input_folder=_mp3s(tmp_path))

    with patch.object(bind.ffmpeg_utils, "parallel_transcode",
                      side_effect=lambda mp3s, d, **kw: list(mp3s)), \
         patch.object(bind.ffmpeg_utils, "concat_audio_to_m4b"), \
         patch.object(bind.ffmpeg_utils, "create_chapters_ffmetadata") as derive, \
         patch.object(bind.ffmpeg_utils, "embed_chapters_and_meta"):
        bind._bind_one(opts.input_folder, str(tmp_path / "out.m4b"), opts)

    derive.assert_called_once()


def test_copy_mode_skips_the_transcode(tmp_path):
    opts = bind.BindOptions(input_folder=_mp3s(tmp_path), audio_mode="copy")

    with patch.object(bind.ffmpeg_utils, "parallel_transcode") as transcode, \
         patch.object(bind.ffmpeg_utils, "concat_audio_to_m4b") as concat, \
         patch.object(bind.ffmpeg_utils, "create_chapters_ffmetadata"), \
         patch.object(bind.ffmpeg_utils, "embed_chapters_and_meta"):
        bind._bind_one(opts.input_folder, str(tmp_path / "out.m4b"), opts)

    transcode.assert_not_called()
    # the mp3s go straight into the concat
    concatted = concat.call_args.args[0]
    assert all(p.endswith(".mp3") for p in concatted)


def test_transcode_mode_still_transcodes(tmp_path):
    opts = bind.BindOptions(input_folder=_mp3s(tmp_path))

    with patch.object(bind.ffmpeg_utils, "parallel_transcode",
                      side_effect=lambda mp3s, d, **kw: ["a.m4a"]) as transcode, \
         patch.object(bind.ffmpeg_utils, "concat_audio_to_m4b"), \
         patch.object(bind.ffmpeg_utils, "create_chapters_ffmetadata"), \
         patch.object(bind.ffmpeg_utils, "embed_chapters_and_meta"):
        bind._bind_one(opts.input_folder, str(tmp_path / "out.m4b"), opts)

    transcode.assert_called_once()


def test_unknown_audio_mode_is_rejected(tmp_path):
    opts = bind.BindOptions(input_folder=_mp3s(tmp_path), audio_mode="nonsense")
    with pytest.raises(ValueError, match="audio_mode"):
        bind._bind_one(opts.input_folder, str(tmp_path / "out.m4b"), opts)


def test_copy_mode_with_external_chapters_real_ffmpeg(tmp_path, fixtures_dir, monkeypatch):
    """No mocks: real ffmpeg end to end for the exact combination libby needs
    (mp3 passthrough + externally-supplied chapters). The other tests above
    mock out every ffmpeg_utils call, so they only prove _bind_one *calls*
    the right functions -- they can't catch a muxer/path bug inside those
    functions, which is exactly what slipped through here twice before this
    test existed (an "ipod" muxer rejecting mp3, and a relative-path 404 in
    the concat demuxer). This is the test that would have caught both.

    input_folder is deliberately relative (and the cwd changed to match) --
    that's what the CLI passes in practice (see Step 8 of the task brief:
    `--input-folder in`), and it's the exact shape that tripped the
    relative-path bug: ffmpeg's concat demuxer resolves relative list
    entries against its own tempdir, not the process cwd.
    """
    book_dir = tmp_path / "in"
    book_dir.mkdir()
    shutil.copy(fixtures_dir / "fixture_ch01.mp3", book_dir / "part01.mp3")
    shutil.copy(fixtures_dir / "fixture_ch02.mp3", book_dir / "part02.mp3")

    chapters = tmp_path / "c.ffmetadata"
    chapters.write_text(
        ";FFMETADATA1\n\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=One\n\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=4000\ntitle=Two\n"
    )

    monkeypatch.chdir(tmp_path)
    opts = bind.BindOptions(
        input_folder="in",
        chapters_file="c.ffmetadata",
        audio_mode="copy",
    )
    bind._bind_one("in", "out.m4b", opts)

    out = tmp_path / "out.m4b"
    assert out.exists()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_entries", "stream=codec_name", "-show_chapters", str(out)],
        check=True, capture_output=True, text=True,
    )
    info = json.loads(probe.stdout)
    audio_codecs = [s["codec_name"] for s in info["streams"] if s["codec_name"] != "bin_data"]
    assert audio_codecs == ["mp3"], f"expected passthrough mp3, got {audio_codecs}"

    titles = [c["tags"]["title"] for c in info["chapters"]]
    assert titles == ["One", "Two"]


def test_default_transcode_bind_keeps_the_m4a_brand_real_ffmpeg(tmp_path, fixtures_dir):
    """No mocks: a plain default-options bind (no --chapters-file, no
    --audio-mode) must keep the exact container brand ffmpeg's "ipod" muxer
    stamps (major_brand=M4A, compatible_brands=M4A isomiso2), not silently
    drift towards mp4's generic defaults. major_brand is the identity
    iTunes/Apple Books/other players use to recognize a file as an
    audiobook; it's silent when lost or altered (the file still builds and
    plays), which is exactly why a previous round of this change broke it
    without any test noticing (mp3_passthrough was being forced
    unconditionally in concat_audio_to_m4b/embed_chapters_and_meta).

    Note: asserting major_brand alone is not enough to catch that specific
    regression, because the fix that followed pairs `-brand "M4A "`
    with `-f mp4` -- so even if mp3_passthrough leaks True on every call,
    major_brand still reads "M4A". What actually drifts under that fault
    is compatible_brands (ipod's "M4A isomiso2" vs mp4's own
    "M4A iso2mp41") and a few bytes of file size from the different
    default compatible-brands list. Both are asserted here so this test
    reproduces the fail-then-pass cycle the fault injection is meant to
    prove, not just the part that happens to still read right.

    This test does not touch audio_mode or chapters_file at all, matching
    every pre-existing caller of bind_single/bind_multiple.
    """
    book_dir = tmp_path / "in"
    book_dir.mkdir()
    shutil.copy(fixtures_dir / "fixture_ch01.mp3", book_dir / "part01.mp3")
    shutil.copy(fixtures_dir / "fixture_ch02.mp3", book_dir / "part02.mp3")

    out = tmp_path / "out.m4b"
    opts = bind.BindOptions(input_folder=str(book_dir))
    bind._bind_one(str(book_dir), str(out), opts)

    assert out.exists()

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_entries", "format_tags=major_brand,compatible_brands", str(out)],
        check=True, capture_output=True, text=True,
    )
    tags = json.loads(probe.stdout)["format"]["tags"]
    assert tags["major_brand"].strip() == "M4A", \
        f"expected M4A brand, got {tags['major_brand']!r}"
    assert tags["compatible_brands"].strip() == "M4A isomiso2", \
        f"expected ipod-muxer compatible_brands, got {tags['compatible_brands']!r}"
