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
    p.add_argument("--metadata-source", choices=["openlibrary", "none"], default="none",
                   help="Metadata source (google removed, use openlibrary or none)")
    p.add_argument("--title", default="")
    p.add_argument("--author", default="")
    p.add_argument("--bitrate", default="64k", help="Audio bitrate, e.g. 64k (validated ^\\d+[kM]?$)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
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
    p.add_argument("--keep-original", action="store_true", help="Keep original uncleaned as .orig.m4b")
    return p


def _add_clean_parser(sub):
    p = sub.add_parser("clean", help="Clean/restore an existing m4b file — best pipeline: ml-rust 12dB, HP70, two-pass -19 LUFS mono 64k")
    p.add_argument("--input", required=True, help="Path to a .m4b file or a directory of them")
    p.add_argument("--pattern", default="*.m4b", help="Glob when --input is a directory")
    p.add_argument("--keep-original", action="store_true",
                   help="Save original as <name>.orig.m4b instead of replacing in place")
    p.add_argument("--skip-threshold-db", type=float, default=-35.0,
                   help="Skip files with mean volume below this dB (i.e. already-quiet)")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing backup files")
    p.add_argument("--chunk-s", type=float, default=60.0, help="Chunk size seconds (default 60s, best)")
    p.add_argument("--overlap-s", type=float, default=2.0, help="Overlap seconds (default 2s, equal-power sin/cos)")
    p.add_argument("--device", default=None, help="Device override: cpu, cuda, or auto (default auto)")
    p.add_argument("--atten-lim", default="12", help="Attenuation limit dB 0-100 or auto (default 12 best per listening tests: 6->12 perceptible, 12->100 same, auto estimates per-file SNR)")
    p.add_argument("--pf", action="store_true", help="Enable post-filter (over-attenuates, keep off for audiobooks)")
    p.add_argument("--pf-beta", type=float, default=0.02, help="Post-filter beta (default 0.02, higher stronger)")
    p.add_argument("--dry-run", action="store_true", help="List files that would be cleaned without cleaning")
    p.add_argument("--jobs", type=int, default=1, help="Parallel jobs for batch clean (default 1)")
    p.add_argument("--auto", action="store_true", help="Use best values (same as default, kept for compatibility)")
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
    from m4b_lib.cleanup import clean_one, iter_targets
    targets = list(iter_targets(args.input, args.pattern))
    if not targets:
        print(f"[WARN] No m4b files matched pattern {args.pattern!r} in {args.input}")
        sys.exit(1)
    if args.dry_run:
        print(f"[DRY-RUN] Would clean {len(targets)} files with best pipeline (ml-rust 12dB, HP70, two-pass -19 LUFS mono):")
        for p in targets:
            print(f"  {p}")
        return

    if args.jobs and args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[INFO] Cleaning {len(targets)} files with {args.jobs} jobs (ThreadPool) — best: ml-rust 12dB")
        def _clean_one(p):
            try:
                # Clean now does auto best single set, no mode param needed — always ml-rust 12dB
                clean_one(p, keep_original=args.keep_original,
                          skip_threshold_db=args.skip_threshold_db,
                          chunk_s=args.chunk_s, overlap_s=args.overlap_s, device=args.device,
                          atten_lim_db=args.atten_lim, pf=args.pf, pf_beta=args.pf_beta)
                return (p, None)
            except Exception as e:
                return (p, e)
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(_clean_one, p): p for p in targets}
            failed = []
            for f in as_completed(futures):
                path, err = f.result()
                if err:
                    print(f"[ERROR] Failed on {path}: {err}")
                    failed.append(path)
            if failed:
                print(f"[WARN] {len(failed)}/{len(targets)} failed")
                sys.exit(1)
    else:
        for path in targets:
            try:
                clean_one(path, keep_original=args.keep_original,
                          skip_threshold_db=args.skip_threshold_db,
                          chunk_s=args.chunk_s, overlap_s=args.overlap_s, device=args.device,
                          atten_lim_db=args.atten_lim, pf=args.pf, pf_beta=args.pf_beta)
            except Exception as e:
                print(f"[ERROR] Failed on {path}: {e}")
                continue


def main():
    # Compat shim for old flat args (pre-subcommand): `m4binder.py --mode multiple --input-folder ...`
    # If first arg doesn't look like subcommand, assume 'bind' for backward compat with run_hugo.sh
    if len(sys.argv) > 1 and sys.argv[1] not in ("bind", "clean", "-h", "--help"):
        if any(a.startswith("--mode") or a.startswith("--input-folder") for a in sys.argv[1:]):
            print("[DEPRECATED] flat args without subcommand, assuming 'bind' — please use 'm4binder.py bind ...'", file=sys.stderr)
            sys.argv.insert(1, "bind")

    parser = argparse.ArgumentParser(prog="m4binder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_bind_parser(sub)
    _add_clean_parser(sub)
    _add_migrate_parser(sub)
    args = parser.parse_args()
    if args.cmd == "bind":
        _dispatch_bind(args)
    elif args.cmd == "clean":
        _dispatch_clean(args)
    elif args.cmd == "migrate":
        _dispatch_migrate(args)


if __name__ == "__main__":
    main()
