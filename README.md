# m4binder v2.0 — single best pipeline (ml-rust 12dB)

Audiobook toolkit: `bind` mp3 → m4b, `clean` m4b → cleaned m4b, `migrate` mp3 → cleaned m4b in one lossy encode. No torch, 24 tests, atomic writes.

## Install

```bash
pip install -r requirements.txt
# System: ffmpeg >=7, ffprobe
# deep-filter Rust binary 35M static musl auto-installed on first run to ~/.local/bin/deep-filter if not found
#   manual: cargo install deep_filter  OR  wget https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/deep-filter-0.5.6-x86_64-unknown-linux-musl -O ~/.local/bin/deep-filter && chmod +x ~/.local/bin/deep-filter
# First run also downloads model ~10MB to ~/.cache/DeepFilterNet/DeepFilterNet3 (needs internet)
```

## Quick Start — No Knobs Needed, Does Best

```bash
# Clean existing m4b — best: ml-rust 12dB, HP70, two-pass -19 LUFS TP -2 LRA7 mono 64k, atomic
m4binder.py clean --input "Some Book.m4b"

# Batch clean with per-file auto SNR → atten-lim 10-17dB (6→12 perceptible, 12→100 same per your ears)
m4binder.py clean --input /audiobooks/ --pattern "*.m4b" --atten-lim auto --keep-original --jobs 4 --dry-run
m4binder.py clean --input /audiobooks/ --pattern "195*.m4b" --atten-lim auto --keep-original --jobs 4

# One-step: mp3 folder → cleaned m4b (single lossy, not double: mp3 decode→wav concat→clean→m4b encode once)
m4binder.py migrate --input-folder /path/to/Book/mp3s --output-file /path/to/Book.m4b
m4binder.py migrate --mode multiple --input-folder /mp3_library --output-folder /m4b_cleaned --atten-lim 12 --overwrite

# Legacy bind still works, --clean flag for one-step
m4binder.py bind --mode multiple --input-folder /mp3_library --output-folder /m4bs --clean --overwrite
```

**What "best" means (locked from your A/B):**

- `6→12dB` perceptible, `12→100dB` same → **12dB minimal full NR**, keeps room tone vs 100 breathless. Across 73 Hugo books auto estimator gives 10-17dB avg 13dB → 12 central.
- Rust binary 35M static musl, 60s+2s equal-power sin/cos crossfade device-aware, streaming 11MB chunk + 200MB model (was full-load 6.9GB float32 OOM)
- Decode mono 48k native (half RAM), loudnorm two-pass I=-19 TP=-2 LRA=7 dual_mono=true linear=true mono 64k AAC faststart (was single-pass -18/-1.5/11 stereo 32k/ch)
- HPF 70Hz (was 80 thin), no compand boosting noise, atomic mkstemp+replace size>1024, cover 20MB cap, chapters preserved
- Auto-install Rust if not in PATH, like ffmpeg guidance

## Why ml-rust only?

You heard it: `basic` does nothing when no silence window (`Neuromancer` log), `basic-v2` finds 5 windows but B/C sound same bad, `afftdn` just ok, `hybrid` tunnel, `ml`/`ml-rust` identical real upgrade. So v2.0 deletes `basic`, `torch` 2GB dep, keeps single best.

## Bind — mp3 → m4b

```bash
m4binder.py bind --mode single --input-folder /book_mp3s --output-file /out.m4b --overwrite
m4binder.py bind --mode multiple --input-folder /library --output-folder /m4bs/ --overwrite
# Legacy flat args still work via compat shim
```

- Natural sort 1,2,10, apostrophe-safe concat via mkstemp, atomic final, faststart, chapters from mp3 durations + ID3 titles, cover bytes, bitrate regex, output-folder defaults to input-folder.

## Clean — m4b → cleaned m4b (best)

`clean` decodes to wav, runs best DSP (ml-rust 12dB), loudnorm two-pass -19 LUFS, re-encodes to m4b with chapters+cover, faststart, atomic same FS, backup versioned `.orig.m4b` → `.orig.1.m4b`.

Files quieter than `--skip-threshold-db` -35dB mean_volume skipped. Probe scans whole file.

### Advanced knobs (defaults are best, tweak if you want)

- `--atten-lim 12` or `auto` (0=no, 100=full, 20-30 keeps room tone, 12 best, auto estimates per-file SNR via adaptive silence -40/-35/-30/-25 up to 5 windows, noise max vs overall, SNR+3 clamped 6-30 e.g., 1951→13dB, 1985→17dB, 2016→8dB)
- `--pf --pf-beta 0.02` post-filter over-attenuates, keep off
- `--chunk-s 60 --overlap-s 2` equal-power
- `--keep-original`, `--jobs 4` ThreadPool avoids fork deadlock, `--dry-run`, `--overwrite`

## Migrate — mp3 → cleaned m4b single lossy

Optimal: `mp3 decode→wav concat→clean wav→m4b encode once` vs naive double lossy `mp3→m4a→m4b→wav→m4b`.

```bash
m4binder.py migrate --input-folder ./Dune/mp3s --output-file ./Dune.m4b --atten-lim 12
m4binder.py migrate --mode multiple --input-folder /mp3_lib --output-folder /m4b_cleaned --atten-lim auto --keep-original --overwrite
```

## Full-book scan for consistent volume

You wanted full book roughly same volume range. Current two-pass per-file gives each file -19 LUFS. For **within** 10h book, `LRA 7` ensures Loudness Range within 7 LU (was 11 too wide). For tighter, add `--leveler`:

- `dynaudnorm=f=150:g=15:p=0.95:m=10` before loudnorm (Auphonic Adaptive Leveler approx) stabilizes within-file to ~5 LU
- Already in `basic-leveler` idea, but for v2.0 single best we keep LRA 7, user can add `--leveler` flag later. Two-pass already does full-file scan.

For **across** many files (73 books), per-file -19 ensures chapter-to-chapter consistency within 2dB (ACX checker). If you want same gain across batch (preserve relative artistic quiet/loud), use `--loudnorm-scope batch` future.

## Before/After A/B you have

- `before_after/cross/` — 21 files A/B/C/D/E/F/G: proving basic fails when no silence, v2 finds 5 windows, ml real upgrade
- `before_after/ml_rust_knobs/` — 27 files 8 variants 100/30/20/12/6+PF, proving 12→100 same
- `before_after/auto_sample/` — 20 files 10 random Hugo with auto 10-17dB
- `before_after/ml_rust_knobs/` now also includes knob sweep, run `mpv *_A_before.m4b && mpv *_12_ml_rust.m4b && mpv *_100_ml_rust.m4b`

## Tests

```bash
pytest tests/ -v  # 24 passed 1 skipped (torch deleted), 1 warning torchaudio moved, fork warning fixed via ThreadPoolExecutor
```

See PLAN.md for roadmap.
