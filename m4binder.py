#!/usr/bin/env python3
"""m4binder — audiobook tooling CLI.

Subcommands:
  bind   Convert mp3 chapter folders into m4b files (single or multiple mode).
  clean  Restore/clean an existing m4b file (basic DSP or ML).
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
    p.add_argument("--metadata-source", choices=["openlibrary", "google", "none"], default="none")
    p.add_argument("--title", default="")
    p.add_argument("--author", default="")
    p.add_argument("--bitrate", default="64k")
    return p


def _add_clean_parser(sub):
    p = sub.add_parser("clean", help="Clean/restore an existing m4b file")
    p.add_argument("--input", required=True, help="Path to a .m4b file or a directory of them")
    p.add_argument("--pattern", default="*.m4b", help="Glob when --input is a directory")
    p.add_argument("--mode", choices=["basic", "ml"], default="basic")
    p.add_argument("--keep-original", action="store_true",
                   help="Save original as <name>.orig.m4b instead of replacing in place")
    p.add_argument("--skip-threshold-db", type=float, default=-35.0,
                   help="Skip files with mean volume below this dB (i.e. already-quiet)")
    return p


def _dispatch_bind(args):
    opts = BindOptions(
        input_folder=args.input_folder,
        output_file=args.output_file,
        output_folder=args.output_folder,
        metadata_source=args.metadata_source,
        title=args.title,
        author=args.author,
        bitrate=args.bitrate,
    )
    if args.mode == "single":
        if not args.output_file:
            sys.exit("--output-file is required in single mode")
        bind_single(opts)
    else:
        bind_multiple(opts)


def _dispatch_clean(args):
    # Lazy import so 'bind' users don't pay for cleanup dependencies
    from m4b_lib.cleanup import clean_one, iter_targets
    for path in iter_targets(args.input, args.pattern):
        clean_one(path, mode=args.mode, keep_original=args.keep_original,
                  skip_threshold_db=args.skip_threshold_db)


def main():
    parser = argparse.ArgumentParser(prog="m4binder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_bind_parser(sub)
    _add_clean_parser(sub)
    args = parser.parse_args()
    if args.cmd == "bind":
        _dispatch_bind(args)
    elif args.cmd == "clean":
        _dispatch_clean(args)


if __name__ == "__main__":
    main()
