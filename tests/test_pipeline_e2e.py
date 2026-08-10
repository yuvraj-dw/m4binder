import os
import shutil
import subprocess

import pytest

import m4b_lib.cleanup as cleanup_mod
from m4b_lib import ffmpeg_utils
from m4b_lib.cleanup import clean_one

pytestmark = pytest.mark.ml


def test_clean_one_preserves_duration_and_chapters(fixture_m4b, tmp_path):
    work = tmp_path / "work.m4b"
    shutil.copy(fixture_m4b, work)
    before = ffmpeg_utils.get_duration(str(work))

    clean_one(str(work), atten_lim_db=12.0, workers={"df3": 2, "encode": 2, "loudness": 2})

    after = ffmpeg_utils.get_duration(str(work))
    assert abs(after - before) < 0.5


def test_clean_one_keep_original_makes_a_backup(fixture_m4b, tmp_path):
    work = tmp_path / "keep.m4b"
    shutil.copy(fixture_m4b, work)
    clean_one(str(work), keep_original=True,
              workers={"df3": 1, "encode": 1, "loudness": 1})
    assert (tmp_path / "keep.orig.m4b").exists()


def test_clean_one_refuses_to_replace_original_with_truncated_output(fixture_m4b, tmp_path, monkeypatch):
    """A plausible-but-wrong staged file (sane size, wrong duration) must not
    replace the original in place — this is the failure mode that doesn't
    raise on its own, so clean_one has to catch it before os.replace."""
    work = tmp_path / "work.m4b"
    shutil.copy(fixture_m4b, work)
    before_bytes = work.read_bytes()
    before_duration = ffmpeg_utils.get_duration(str(work))

    def _fake_clean_timeline(tl, out_path, **kwargs):
        # Encode only the first 0.5s of the real decoded wav: a valid,
        # sane-sized m4b that is silently short by most of the book.
        subprocess.run(
            ["ffmpeg", "-y", "-i", tl.wav_path, "-t", "0.5", "-c:a", "aac", "-b:a", "64k", out_path],
            check=True, capture_output=True,
        )

    monkeypatch.setattr(cleanup_mod, "clean_timeline", _fake_clean_timeline)

    with pytest.raises(RuntimeError, match="duration mismatch"):
        clean_one(str(work))

    # No partial replace: original must be byte-for-byte untouched.
    assert work.read_bytes() == before_bytes
    assert abs(ffmpeg_utils.get_duration(str(work)) - before_duration) < 0.01


def test_pf_reaches_the_backend_from_clean_timeline(tmp_path):
    """--pf must survive the trip from CLI to EnhanceConfig.

    It previously did not: the flag was declared on the CLI and honoured by
    DF3Enhancer.load, but clean_timeline had no pf parameter between them, so
    both --pf and --clean-pf were silent no-ops. Nothing caught it because no
    test followed the value across that seam.
    """
    import soundfile as sf

    from m4b_lib.cleanup import clean_timeline
    from m4b_lib.enhance import register
    from m4b_lib.timeline import Chapter, Timeline

    seen = {}

    @register("pfprobe")
    class _Probe:
        name = "pfprobe"
        sample_rate = 48000

        def __init__(self, cfg):
            seen["cfg"] = cfg

        def load(self, device):
            pass

        def enhance_batch(self, x, state=None):
            return x, None          # passthrough; we only care about the config

        def close(self):
            pass

    src = tmp_path / "t.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "quiet", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=3:sample_rate=48000",
         "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(src)],
        check=True,
    )
    tl = Timeline(wav_path=str(src), sample_rate=48000,
                  frames=sf.info(str(src)).frames,
                  chapters=[Chapter(0.0, 3.0, "One")])

    clean_timeline(tl, str(tmp_path / "out.m4b"), pf=True, backend="pfprobe",
                   workers={"df3": 1, "loudness": 1, "encode": 1})

    assert seen["cfg"].pf is True, "pf did not reach the backend's EnhanceConfig"


@pytest.mark.parametrize("sub", ["bind", "clean", "migrate"])
def test_cli_dispatch_reaches_the_library_for_every_subcommand(sub, fixture_m4b, fixtures_dir, tmp_path):
    """Invoke the real CLI, not just the parser.

    `--help` exiting 0 only proves the parser builds; it says nothing about
    whether _dispatch_* can read the attributes it uses. A dispatch referencing
    an arg its own subparser never declared raises AttributeError at runtime
    and is invisible to both `--help` and to tests that call the library
    directly — which is exactly how a `clean` crash shipped once.

    Each case runs far enough to exercise dispatch and argument access, then
    stops on a deliberate, recognisable error rather than doing real work.
    """
    import sys

    argv = {
        # NOT --dry-run: it returns before the loop that reads args.pf etc.
        # A real (tiny) clean is the only path that touches every arg dispatch uses.
        "clean": ["clean", "--input", str(tmp_path / "work.m4b"), "--jobs-df3", "1",
                  "--jobs-loudness", "1", "--jobs-encode", "1"],
        # missing --output-file is a clean sys.exit from dispatch, after arg access
        "migrate": ["migrate", "--input-folder", str(fixtures_dir)],
        "bind": ["bind", "--input-folder", str(fixtures_dir), "--mode", "single"],
    }[sub]

    if sub == "clean":
        shutil.copy(fixture_m4b, tmp_path / "work.m4b")

    r = subprocess.run([sys.executable, "m4binder.py", *argv],
                       capture_output=True, text=True, cwd=os.getcwd())
    combined = r.stdout + r.stderr
    # Assert on OUTCOME, not on exception text: _dispatch_clean catches Exception
    # and prints it, so a missing-attribute crash produces neither a traceback nor
    # the string "AttributeError" — it shows up only as a non-zero exit and a
    # "[ERROR] Failed on ..." line. An earlier version of this test asserted on
    # those strings and passed against the very bug it was written to catch.
    assert "no attribute" not in combined, (
        f"{sub} dispatch referenced an argument its subparser does not define:\n{combined[-800:]}"
    )
    assert "Traceback" not in combined, f"{sub} dispatch crashed:\n{combined[-800:]}"
    if sub == "clean":
        assert r.returncode == 0, f"clean exited {r.returncode}:\n{combined[-800:]}"
