# m4binder + audio cleanup

Two subcommands sharing one codebase:

## bind — mp3 chapters → m4b

```bash
# Single book
python m4binder.py bind --mode single \
  --input-folder /path/to/book_mp3s \
  --output-file /path/to/output.m4b

# Many books (each subfolder is a book)
python m4binder.py bind --mode multiple \
  --input-folder /path/to/library \
  --output-folder /path/to/m4bs/
```

Optional metadata lookup with `--metadata-source openlibrary --title "Foo" --author "Bar"`.

## clean — restore an existing m4b

```bash
# One book, basic DSP (sox + ffmpeg, CPU only)
python m4binder.py clean --input '/path/to/Some Book.m4b'

# One book, ML mode (DeepFilterNet 3, uses GPU if available)
python m4binder.py clean --input '/path/to/Some Book.m4b' --mode ml

# Batch a directory
python m4binder.py clean --input /mnt/md0/share/audiobooks/ --pattern '195*.m4b' --mode ml

# Keep the original next to the cleaned file as Some\ Book.orig.m4b
python m4binder.py clean --input '/path/to/Some Book.m4b' --mode ml --keep-original
```

`clean` decodes the m4b to wav, runs the chosen pipeline, then re-encodes back to m4b
with chapter markers preserved. Files quieter than `--skip-threshold-db` (default -35dB)
are left untouched as already-clean.

### Pipelines

- **basic**: ffmpeg silencedetect → sox noiseprof/noisered (aggression 0.21)
  → highpass 80 Hz, lowpass 14 kHz → compander → ffmpeg loudnorm -18 LUFS
- **ml**: DeepFilterNet 3 inference (`pip install -r requirements-ml.txt`)
  → ffmpeg loudnorm -18 LUFS

## Install

```bash
pip install -r requirements.txt          # basic + bind
pip install -r requirements-ml.txt       # add ML cleanup
```

System deps: `ffmpeg`, `ffprobe`, `sox` (for basic mode).

## Tests

```bash
pytest tests/ -v
```

Smoke tests only — no full TDD coverage, manual spot-checks recommended for cleanup quality.
