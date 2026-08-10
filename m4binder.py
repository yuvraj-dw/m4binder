#!/usr/bin/env python3
"""m4binder — audiobook tooling CLI.

Subcommands:
  bind   Convert mp3 chapter folders into m4b files (single or multiple mode).
  clean  Restore/clean an existing m4b file (basic DSP or ML).
  tag    Write tags and cover art onto an existing m4b, without re-encoding.
"""
import argparse
import os
import sys

from m4b_lib.bind import BindOptions, bind_single, bind_multiple


def _add_bind_parser(sub):
    p = sub.add_parser("bind", help="Convert mp3 chapter folders into m4b")
    p.add_argument("--mode", choices=["single", "multiple"], default="single")
    p.add_argument("--input-folder", required=True)
    p.add_argument("--output-file")
    p.add_argument("--output-folder")
    p.add_argument("--metadata-source", choices=["openlibrary", "none"], default="none",
                   help="Metadata source (google removed, use openlibrary or none)")
    p.add_argument("--title", default="")
    p.add_argument("--author", default="")
    p.add_argument("--bitrate", default="64k", help="Audio bitrate, e.g. 64k (validated ^\\d+[kM]?$)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    p.add_argument("--chapters-file",
                   help="ffmetadata file of chapter markers. Overrides the "
                        "default one-chapter-per-mp3 derivation.")
    p.add_argument("--audio-mode", choices=["transcode", "copy"],
                   default="transcode",
                   help="copy muxes the mp3 streams into the m4b untouched "
                        "(bit-exact, no encode); transcode re-encodes to AAC.")
    p.add_argument("--clean", action="store_true", help="After binding, clean the resulting m4b with best pipeline (ml-rust 12dB) — one-step migrate+clean")
    p.add_argument("--clean-atten-lim", default="12", help="Attenuation limit for --clean (default 12dB best, 0=no, 100=full, or auto for per-file SNR)")
    p.add_argument("--clean-pf", action="store_true", help="Enable post-filter for --clean")
    return p


def _add_migrate_parser(sub):
    p = sub.add_parser("migrate", help="One-step: mp3 chapters → cleaned m4b (bind + clean in one, single lossy encode)")
    p.add_argument("--input-folder", required=True, help="Folder containing mp3 chapters or subfolders")
    p.add_argument("--mode", choices=["single", "multiple"], default="single")
    p.add_argument("--output-file", help="Output m4b for single mode")
    p.add_argument("--output-folder", help="Output folder for multiple mode")
    p.add_argument("--metadata-source", choices=["openlibrary", "none"], default="none")
    p.add_argument("--title", default="")
    p.add_argument("--author", default="")
    p.add_argument("--bitrate", default="64k", help="Audio bitrate 64k mono standard")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    p.add_argument("--atten-lim", default="12", help="ML-Rust atten-lim dB 0-100 or auto (default 12 best per listening tests)")
    p.add_argument("--pf", action="store_true", help="Enable post-filter")
    # Deliberately still opt-in, unlike `clean`'s. migrate builds a NEW m4b from
    # mp3s and then cleans it; the source mp3s are untouched, so a bad clean
    # costs a re-run rather than the only copy. `clean` overwrites in place and
    # is the one that needs the safety net.
    p.add_argument("--keep-original", action="store_true", help="Keep original uncleaned as .orig.m4b")
    return p


def _add_clean_parser(sub):
    p = sub.add_parser("clean", help="Clean/restore an existing m4b — DeepFilterNet3 torch pipeline")
    p.add_argument("--input", required=True, help="Path to a .m4b file or a directory of them")
    p.add_argument("--pattern", default="*.m4b", help="Glob when --input is a directory")
    # ON by default since 2026-08-10. `clean` overwrites in place, and there is
    # a known class of book where the result is worse -- DeepFilterNet3 treats a
    # music bed as noise, and the blind test put the untouched file first on the
    # one book with music. A backup is cheap; an over-suppressed score is not
    # recoverable. --no-keep-original restores the old behaviour explicitly.
    p.add_argument("--keep-original", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Save original as <name>.orig.m4b before replacing "
                        "in place (default: on; --no-keep-original to disable)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    p.add_argument("--backend", default="df3",
                   help="Enhancement backend (default df3)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="Inference device (default cpu)")
    p.add_argument("--atten-lim", default="12",
                   help="Noise attenuation dB, or 'auto' to estimate per-file from SNR "
                        "(0 = passthrough, 12 = default, 100 = full)")
    p.add_argument("--pf", action="store_true",
                   help="Enable the model's post-filter (over-attenuates; usually leave off)")
    p.add_argument("--jobs-df3", type=int, default=None, help="Workers for enhancement")
    p.add_argument("--jobs-loudness", type=int, default=None, help="Workers for loudness")
    p.add_argument("--jobs-encode", type=int, default=None,
                   help="Workers for AAC encode. Above 1 produces audible joins — see README")
    p.add_argument("--dry-run", action="store_true", help="List files that would be cleaned")
    return p


def _dispatch_bind(args):
    import re
    if not re.match(r'^\d+[kKmM]?$', args.bitrate):
        sys.exit(f"Invalid --bitrate {args.bitrate!r}, expected like 64k or 128k")
    opts = BindOptions(
        input_folder=args.input_folder,
        output_file=args.output_file,
        output_folder=args.output_folder,
        metadata_source=args.metadata_source,
        title=args.title,
        author=args.author,
        bitrate=args.bitrate,
        overwrite=args.overwrite,
        chapters_file=args.chapters_file,
        audio_mode=args.audio_mode,
    )
    # Handle --clean flag for one-step migrate+clean
    if getattr(args, 'clean', False):
        # One-step: bind then clean the resulting m4b(s) with best pipeline
        from m4b_lib.migrate import bind_and_clean_single, bind_and_clean_multiple
        if args.mode == "single":
            if not args.output_file:
                sys.exit("--output-file is required in single mode")
            if os.path.exists(args.output_file) and not args.overwrite:
                sys.exit(f"Output {args.output_file} exists, use --overwrite to allow")
            bind_and_clean_single(opts, clean_atten_lim=args.clean_atten_lim, clean_pf=args.clean_pf)
        else:
            bind_and_clean_multiple(opts, clean_atten_lim=args.clean_atten_lim, clean_pf=args.clean_pf)
    else:
        if args.mode == "single":
            if not args.output_file:
                sys.exit("--output-file is required in single mode")
            if os.path.exists(args.output_file) and not args.overwrite:
                sys.exit(f"Output {args.output_file} exists, use --overwrite to allow")
            bind_single(opts)
        else:
            bind_multiple(opts)


def _dispatch_migrate(args):
    import re
    if not re.match(r'^\d+[kKmM]?$', args.bitrate):
        sys.exit(f"Invalid --bitrate {args.bitrate!r}")
    from m4b_lib.bind import BindOptions
    from m4b_lib.migrate import bind_and_clean_single, bind_and_clean_multiple
    opts = BindOptions(
        input_folder=args.input_folder,
        output_file=args.output_file,
        output_folder=args.output_folder,
        metadata_source=args.metadata_source,
        title=args.title,
        author=args.author,
        bitrate=args.bitrate,
        overwrite=args.overwrite,
    )
    if args.mode == "single":
        if not args.output_file:
            sys.exit("--output-file is required in single mode")
        bind_and_clean_single(opts, clean_atten_lim=args.atten_lim, clean_pf=args.pf, keep_original=args.keep_original)
    else:
        bind_and_clean_multiple(opts, clean_atten_lim=args.atten_lim, clean_pf=args.pf, keep_original=args.keep_original)


def _dispatch_clean(args):
    from m4b_lib import ffmpeg_utils
    from m4b_lib.cleanup import clean_one, iter_targets

    is_auto = args.atten_lim.strip().lower() == "auto"
    if not is_auto:
        # Fail fast on a bad spec before touching any file, not partway through a batch.
        try:
            float(args.atten_lim)
        except ValueError:
            sys.exit(f"Invalid --atten-lim {args.atten_lim!r}: expected a number or 'auto'")

    targets = list(iter_targets(args.input, args.pattern))
    if not targets:
        print(f"[WARN] No m4b files matched {args.pattern!r} in {args.input}")
        sys.exit(1)
    if args.dry_run:
        print(f"[DRY-RUN] Would clean {len(targets)} files:")
        for p in targets:
            print(f"  {p}")
        return

    workers = {k: v for k, v in (
        ("df3", args.jobs_df3), ("loudness", args.jobs_loudness),
        ("encode", args.jobs_encode)) if v}

    failed = []
    for path in targets:
        try:
            # "auto" is re-estimated per file (SNR varies book to book); a
            # fixed spec just re-parses the same float each time — cheap
            # either way, and it keeps one resolution path for both.
            atten = ffmpeg_utils.resolve_atten_lim(args.atten_lim, path)
            clean_one(path, keep_original=args.keep_original,
                      atten_lim_db=atten, pf=args.pf, backend=args.backend,
                      device=args.device, workers=workers)
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")
            failed.append(path)
    if failed:
        print(f"[WARN] {len(failed)}/{len(targets)} failed")
        sys.exit(1)


def _add_tag_parser(sub):
    p = sub.add_parser(
        "tag",
        help="Write tags/cover onto an existing m4b in place (no re-encode)",
    )
    p.add_argument("--file", required=True, help="The .m4b to tag")
    p.add_argument(
        "--metadata-json", required=True,
        help='JSON object of generic tag names, e.g. {"title": "...", '
             '"composer": "Narrator", "series": "X", "series-part": "2"}',
    )
    p.add_argument("--cover", help="Image file to embed as cover art")
    p.add_argument("--cover-format", choices=["jpeg", "png"], default="jpeg")


def _dispatch_tag(args):
    import json
    from pathlib import Path
    from m4b_lib.tagging import TaggingFailed, write_m4b_tags

    tags = json.loads(Path(args.metadata_json).read_text())
    cover = Path(args.cover).read_bytes() if args.cover else None
    try:
        write_m4b_tags(args.file, tags, cover, args.cover_format)
    except TaggingFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"tagged {args.file}")


def build_parser() -> argparse.ArgumentParser:
    """The full CLI. Split out of main() so tests can assert on defaults.

    Without this, the only way to check that e.g. --keep-original defaults on
    was to run the binary -- so nothing did, and the destructive default went
    unguarded.
    """
    parser = argparse.ArgumentParser(prog="m4binder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_bind_parser(sub)
    _add_clean_parser(sub)
    _add_migrate_parser(sub)
    _add_tag_parser(sub)
    return parser


def main():
    # Compat shim for old flat args (pre-subcommand): `m4binder.py --mode multiple --input-folder ...`
    # If first arg doesn't look like subcommand, assume 'bind' for backward compat with run_hugo.sh
    if len(sys.argv) > 1 and sys.argv[1] not in ("bind", "clean", "migrate", "-h", "--help"):
        if any(a.startswith("--mode") or a.startswith("--input-folder") for a in sys.argv[1:]):
            print("[DEPRECATED] flat args without subcommand, assuming 'bind' — please use 'm4binder.py bind ...'", file=sys.stderr)
            sys.argv.insert(1, "bind")

    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "bind":
        _dispatch_bind(args)
    elif args.cmd == "clean":
        _dispatch_clean(args)
    elif args.cmd == "migrate":
        _dispatch_migrate(args)
    elif args.cmd == "tag":
        _dispatch_tag(args)


if __name__ == "__main__":
    main()
