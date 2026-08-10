"""The two ways `clean` could destroy a file it was asked to improve.

`clean` overwrites in place. That is fine when the result is better and
catastrophic when it is not -- and there is a known class where it is not:
DeepFilterNet3 treats a music bed as noise, so a book with a score under the
narration comes out damaged. Peter's blind test put *untouched* first on the one
book in the set with music, ahead of every processed variant.

Both behaviours guarded here were the default until 2026-08-10:

1. `--keep-original` was OFF, so the damaged file replaced the only copy.
2. the directory walk matched the `.orig.m4b` backups a previous run had
   written, putting a second lossy generation on the file whose entire purpose
   is to be pristine.

Together they mean the documented quick-start command could destroy a
music-bearing book *and* its backup.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m4b_lib.cleanup import iter_targets  # noqa: E402


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00" * 16)
    return p


# --- 2. the directory walk must not eat its own backups ----------------------

def test_the_directory_walk_skips_orig_backups(tmp_path):
    """Reverting the filter makes this red: the backup gets a second encode."""
    _touch(tmp_path / "Book.m4b")
    _touch(tmp_path / "Book.orig.m4b")

    got = {os.path.basename(p) for p in iter_targets(str(tmp_path), "*.m4b")}
    assert got == {"Book.m4b"}, f"backup was queued for cleaning: {got}"


def test_the_recursive_walk_also_skips_backups(tmp_path):
    """`**` takes a different code path in iter_targets and needs its own guard."""
    _touch(tmp_path / "a" / "Book.m4b")
    _touch(tmp_path / "a" / "Book.orig.m4b")
    _touch(tmp_path / "b" / "Other.orig.1.m4b")     # the numbered-collision form

    got = {os.path.basename(p) for p in iter_targets(str(tmp_path), "**/*.m4b")}
    assert got == {"Book.m4b"}, f"backup was queued for cleaning: {got}"


def test_naming_a_backup_explicitly_still_works(tmp_path):
    """Skipping is for the sweep. An explicit path is an explicit instruction."""
    p = _touch(tmp_path / "Book.orig.m4b")
    assert list(iter_targets(str(p), "*.m4b")) == [str(p)]


def test_a_book_merely_containing_orig_is_not_skipped(tmp_path):
    """'Original Sin.m4b' must not be mistaken for a backup."""
    _touch(tmp_path / "Original Sin.m4b")
    _touch(tmp_path / "The Originals.m4b")
    got = {os.path.basename(p) for p in iter_targets(str(tmp_path), "*.m4b")}
    assert got == {"Original Sin.m4b", "The Originals.m4b"}, got


# --- 1. the original must survive by default ---------------------------------

def test_clean_keeps_the_original_by_default():
    """The signature default is the thing that matters: every caller inherits it.

    `clean_one` is reached from the CLI, from migrate, and from library code. A
    default of False means the safe path is the one you have to remember.
    """
    import inspect

    from m4b_lib.cleanup import clean_one

    default = inspect.signature(clean_one).parameters["keep_original"].default
    assert default is True, "clean_one still defaults to destroying the original"


def test_the_cli_exposes_an_opt_out(capsys):
    """Safe by default is only acceptable if the old behaviour is still reachable."""
    import m4binder

    parser = m4binder.build_parser()
    args = parser.parse_args(["clean", "--input", "x.m4b", "--no-keep-original"])
    assert args.keep_original is False

    args = parser.parse_args(["clean", "--input", "x.m4b"])
    assert args.keep_original is True, "CLI default still discards the original"


def test_keep_original_flag_is_still_accepted():
    """Existing scripts pass --keep-original explicitly; it must not error."""
    import m4binder

    args = m4binder.build_parser().parse_args(
        ["clean", "--input", "x.m4b", "--keep-original"])
    assert args.keep_original is True
