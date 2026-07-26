# m4binder — single best pipeline (ml-rust 12dB)

Audiobook toolkit: **bind** mp3 chapters → m4b, **clean** existing m4b, **migrate** one-step mp3 → cleaned m4b (single lossy encode). Hardened with 25 tests (TDD for H1/H3/H4), atomic writes, no-overwrite guards.

## Quick Start — Best Pipeline (No Knobs Needed)

```bash
# Clean existing m4b with best (ml-rust 12dB, HP70, two-pass -19 LUFS TP -2 LRA7 mono 64k)
m4binder.py clean --input "Some Book.m4b"

# Batch clean with auto atten-lim per-file SNR + keep original versioned
m4binder.py clean --input /mnt/md0/share/audiobooks/ --pattern "195*.m4b" --atten-lim auto --keep-original --jobs 4

# One-step: mp3 folder → cleaned m4b (single lossy encode, not double)
m4binder.py migrate --input-folder /path/to/Book/mp3s --output-file /path/to/Book.m4b

# Many books at once, one-step:
m4binder.py migrate --mode multiple --input-folder /mp3_library --output-folder /m4b_cleaned --overwrite

# Legacy bind still works, optional --clean flag for one-step
m4binder.py bind --mode multiple --input-folder /mp3_library --output-folder /m4bs --clean --clean-atten-lim 12 --overwrite
```

**What "best" means (locked from listening tests):**

- `6→12dB` perceptible, `12→100dB` same on this library → **12dB minimal full NR**, keeps room tone vs 100 breathless
- Across 73 Hugo books auto estimator gives 10-17dB, avg 13dB → 12 central sweet spot
- Rust binary 35M static musl, 60s chunk + 2s equal-power sin/cos crossfade device-aware, streaming via soundfile RAM ~11MB chunk + 200MB model (was full-load 6.9GB float32 for 10h → OOM)
- Decode mono 48k native (half RAM), loudnorm two-pass I=-19 TP=-2 LRA=7 dual_mono=true linear=true mono 64k AAC faststart (was single-pass -18/-1.5/11 stereo 32k/ch poor)
- HPF 70Hz (was 80 thin male), no LPF (let AAC handle), no compand boosting noise (old 0.3,1 6:-70,-60,-20 boosted hiss in pauses)
- Atomic: `mkstemp(dir=final_dir)` + size>1024 validate + `os.replace`, cover 20MB cap, chapters preserved
- Auto-install Rust: if `deep-filter` not in PATH, downloads v0.5.6 musl static from GitHub releases to `~/.local/bin/deep-filter` on first run (like model download to `~/.cache/DeepFilterNet/DeepFilterNet3`)

**Manual A/B you have in `before_after/`:**

- `before_after/cross/` — 21 files A/B/C/D/E/F/G: A_before, B_basic (single-window 0.27, fails when no silence), C_basic-v2 (adaptive 5 windows 0.32 + afftdn fallback), D_afftdn (pure adaptive), E_hybrid (de-hum 60/120/180 + afftdn, tunnel), F_ml/G_ml-rust (Rust 35M, identical, real upgrade)
- `before_after/ml_rust_knobs/` — 27 files 8 variants 100/30/20/12/6 + PF for 3 books, proving 12→100 same
- `before_after/auto_sample/` — 20 files 10 random Hugo books with auto atten-lim 10-17dB

```bash
cd before_after/cross && mpv *_A_before.m4b && mpv *_C_basic-v2.m4b && mpv *_F_ml.m4b
cd ../ml_rust_knobs && mpv *_A_before.m4b && mpv *_12_ml_rust.m4b && mpv *_100_ml_rust.m4b
```

## Bind — mp3 chapters → m4b

```bash
# Single book
m4binder.py bind --mode single --input-folder /path/to/book_mp3s --output-file /path/to/output.m4b --overwrite

# Many books, skips existing unless --overwrite
m4binder.py bind --mode multiple --input-folder /path/to/library --output-folder /path/to/m4bs/ --overwrite

# Legacy flat args still work via compat shim (deprecated):
m4binder.py --mode multiple --input-folder ... --output-folder ...
```

- Natural sort `1,2,10`, apostrophe-safe concat via mkstemp, atomic final, faststart, chapter generation from mp3 durations + ID3 titles, cover preserved as bytes, bitrate validated, output-folder defaults to input-folder.
- Metadata: `--metadata-source openlibrary|none` (google removed).

## Clean — restore existing m4b (best pipeline)

`clean` decodes m4b to wav, runs best DSP (ml-rust 12dB), loudnorm two-pass -19 LUFS, re-encodes to m4b with chapters+cover preserved, faststart, atomic replace same FS, backup versioning `.orig.m4b` → `.orig.1.m4b`.

Files quieter than `--skip-threshold-db` (default -35dB mean_volume) skipped. Probe scans whole file (was 5s).

### Knobs (advanced, defaults are best)

- `--atten-lim 12` or `auto` — 0=no NR, 100=full, 20-30 keeps room tone, 12 best per listening tests, `auto` estimates per-file SNR via adaptive silence -40/-35/-30/-25 up to 5 windows, noise max vs overall, SNR+3 clamped 6-30 (e.g., 1951→13dB, 1985→17dB, 2016→8dB)
- `--pf --pf-beta 0.02` — post-filter over-attenuates very noisy, keep off for audiobooks
- `--chunk-s 60 --overlap-s 2` — chunk size, equal-power crossfade
- `--device cpu|cuda|auto`
- `--keep-original` — save original as .orig.m4b versioned
- `--jobs 4` — ThreadPool to avoid fork-after-torch deadlock (was ProcessPool 72x warnings)

## Migrate — one-step mp3 → cleaned m4b (single lossy)

Optimal single lossy architecture: `mp3 decode→wav concat→clean wav→m4b encode once` (vs naive double lossy mp3→m4a→m4b→wav→m4b).

```bash
m4binder.py migrate --input-folder ./Dune/mp3s --output-file ./Dune.m4b --atten-lim 12
m4binder.py migrate --mode multiple --input-folder /mp3_lib --output-folder /m4b_cleaned --atten-lim auto --keep-original --overwrite
```

## Install

```bash
pip install -r requirements.txt          # mutagen==1.47.0 requests==2.32.3 openlibrary-client@3f13188
# System deps: ffmpeg >=7, ffprobe, deep-filter Rust binary (auto-installed on first run to ~/.local/bin/deep-filter if not found)
# Manual install Rust binary:
#   cargo install deep_filter
#   or download release v0.5.6 x86_64-unknown-linux-musl from https://github.com/Rikorose/DeepFilterNet/releases to ~/.local/bin/
#   or via our auto-install: python -m m4b_lib.cleanup_ml_rust (downloads if needed)

# First run downloads model ~10MB to ~/.cache/DeepFilterNet/DeepFilterNet3, needs internet
```

No torch dep. Old `requirements-ml.txt` with torch 2.6.0 2GB removed.

## Tests — 24 passed 1 skipped

```bash
pytest tests/ -v
# 10 smoke + 8 new coverage + 7 TDD H1/H3/H4 (atomic, no-overwrite, get_duration None) + 1 skipped (torch deleted)
```

- TDD proves atomic no-partial on failure, no-overwrite guard, get_duration None not 0.0
- Coverage: iter_targets batch/recursive/broken symlink, keep-original versioning .orig.1, cover+chapter preservation, sox silence window, apostrophe escaping, backslash escaping, zero-duration, parallel order + no-delete-preexisting
- ThreadPoolExecutor fixes fork deadlock warning (was 169 warnings → 1)

See PLAN.md for full roadmap and remaining P2 (de-hum auto-detect, --tmpdir, disk preflight, logging, split god-module).
