# m4binder + audio cleanup

Two subcommands sharing one codebase, hardened with 25 tests (TDD for H1/H3/H4).

## bind — mp3 chapters → m4b

```bash
# Single book (requires --overwrite to replace existing)
python m4binder.py bind --mode single \
  --input-folder /path/to/book_mp3s \
  --output-file /path/to/output.m4b

# Many books (each subfolder is a book) — skips existing .m4b unless --overwrite
python m4binder.py bind --mode multiple \
  --input-folder /path/to/library \
  --output-folder /path/to/m4bs/ \
  --overwrite

# Legacy flat args still work via compat shim (deprecated):
python m4binder.py --mode multiple --input-folder ... --output-folder ...
```

- Natural sort (`1,2,10` not `1,10,2`), apostrophe-safe concat, atomic final write via tmp+replace, faststart, chapter generation from mp3 durations + ID3 titles, cover preserved as bytes, bitrate validated `^\d+[kM]?$`, output-folder defaults to input-folder.
- Metadata: `--metadata-source openlibrary|none` (google removed). Title fallback prefers album tag (old behavior).

## clean — restore an existing m4b

```bash
# One book, basic DSP (sox + ffmpeg, CPU only, safe default)
python m4binder.py clean --input '/path/to/Some Book.m4b'

# One book, ML mode (DeepFilterNet 3, GPU if available, experimental)
python m4binder.py clean --input '/path/to/Some Book.m4b' --mode ml

# Batch a directory (supports ** recursive)
python m4binder.py clean --input /mnt/md0/share/audiobooks/ --pattern '195*.m4b' --mode ml

# Keep original as .orig.m4b versioned .orig.1.m4b if exists
python m4binder.py clean --input '/path/to/Some Book.m4b' --mode ml --keep-original

# Overwrite guards: bind skips existing .m4b, clean atomic replace with size validation
```

`clean` decodes m4b to wav, runs pipeline, loudnorm two-pass -19 LUFS, re-encodes to m4b with chapters+cover preserved, faststart, atomic replace in same FS, backup versioning.

Files quieter than `--skip-threshold-db` (default -35dB mean_volume) are skipped. Probe now scans whole file via volumedetect (was 5s).

### Pipelines (pro-tuned)

- **basic** (safe, CPU, 1MB dep): ffmpeg silencedetect -40dB d=1.0 → sox noiseprof/noisered aggression 0.27 (was 0.21) with trim EOF guard and sox existence check → highpass 70Hz (was 80) lowpass 16000Hz (was 14k) → compand gate-downward `0.05,0.2 -60,-90,-40,-40,-20,-10,0,-5 0 -90 0.1` (was boosting noise) → ffmpeg loudnorm two-pass I=-19 TP=-2 LRA=7 dual_mono=true (was single-pass -18/-1.5/11). Mono 64k standard (was stereo 64k = 32k/ch poor).
- **ml** (experimental, needs torch 2GB): DeepFilterNet3 inference, streaming via soundfile 60s chunks + 2s equal-power sin/cos crossfade (was linear -3dB dip), device-aware fade tensors, locked model singleton, log_file=None, resampler cache locked, explicit PCM_S 16, unload_model + gc, atomic embed. Decodes mono 48k native (half RAM), loudnorm two-pass same as basic, mono 64k output. Falls back to full-load chunked if SR mismatch (still OOM risk for 10h if fallback). GPU auto-detect, 2-3GB VRAM per chunk.

Both preserve cover (extract via `ffmpeg -an -vcodec copy`, cap 20MB) and chapters, faststart, size>1024 validation.

### Security / Reliability hardening (H1-H4)

- H1 atomic: embed writes to `mkstemp(dir=final_dir)` + `os.replace`, not direct -y truncation. Validates size, cleans temp on failure. Bind and clean both use same pattern.
- H2 dedup TOCTOU: parallel_transcode reserves output via `O_EXCL|O_NOFOLLOW` empty file before ffmpeg, versioning _1 suffix avoids collision with existing_on_disk snapshot, cancels pending futures.
- H3 no-overwrite: parallel_transcode raises FileExistsError or versions, bind_multiple skips existing unless --overwrite, transcode uses -y only on temp it owns.
- H4 get_duration returns None not 0.0 for missing/corrupt, callers skip zero-duration <100ms with warning, placeholder 1s chapter if all skipped.
- M6 concat listfile symlink: uses mkstemp dir=final_dir not output.m4b.concat.txt open(w) following symlink.
- Cover bomb: TODO cap 20MB (partial), requests timeout 15s.

### ML — Is DF3 right for audiobooks? Naysayer view

DF3 = real-time comms tool, ERB 32 bands + deep filtering 96 bins 0-4.8kHz only, >4.8kHz only masking → sibilance loss, breath over-suppression (prosody), musical chirps on sustained vowels. Destroys non-speech (music beds, chapter stings) and downmixes stereo→mono irreversibly. Chunked state reset every 58s → pumping, crossfade dip. Training DNS 3-10s English VoIP, not tape hiss stationary (sox better). Cost 800MB DL 1.6GB disk torch, internet first run.

Keep dual mode, default basic safe, ml experimental with banner. Prefer Rust `deep-filter` binary (15-30MB, tract SIMD, no torch, built-in streaming) if available.

## Install

```bash
pip install -r requirements.txt          # basic + bind (mutagen, requests, openlibrary-client pinned)
pip install -r requirements-ml.txt       # adds ML: deepfilternet==0.5.6, soundfile==0.13.1, torch==2.6.0 torchaudio==2.6.0 (2GB, needs CUDA or CPU index)
# System deps: ffmpeg >=7, ffprobe, sox 14.4.2 (basic), Rust deep-filter binary optional for ML
```

First ML run downloads DeepFilterNet3 model ~10MB to ~/.cache/deepfilternet, needs internet.

## Tests — 25 passing (TDD)

```bash
pytest tests/ -v
# 10 original smoke + 8 new coverage + 7 TDD H1/H3/H4
```

- TDD: test_tdd_failures.py proves atomicity, no-overwrite, get_duration None (red→green)
- Coverage: iter_targets batch/recursive/broken symlink, keep-original versioning .orig.1, cover+chapter preservation, sox silence window, apostrophe escaping, backslash escaping, zero-duration, parallel order + no-delete-preexisting
- Warnings: 72x DeprecationWarning multiprocessing fork after torch threads — will become deadlock in 3.14, should switch to ThreadPoolExecutor or spawn

See PLAN.md for full roadmap and remaining P2.

