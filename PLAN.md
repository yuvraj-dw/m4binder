# m4binder + audio cleanup — Plan Doc

Generated 2026-05-14 after 4 adversarial reviews + TDD (25 tests passing). Working tree dirty vs HEAD (`feat/cleanup-merge`).

## 1. Executive Summary

`feat/cleanup-merge` rewrites monolithic `m4binder.py` (496 LOC) into package `m4b_lib` with subcommands:

- `bind` — mp3 folders → m4b with chapters, cover, faststart, natural sort, atomic final
- `clean` — existing m4b → restored m4b in-place, two modes:
  - `basic` — sox noiseprof/noisered + highpass/lowpass/compand + loudnorm (CPU, 1MB dep, safe)
  - `ml` — DeepFilterNet3 chunked + loudnorm (torch 2GB dep, GPU optional, aggressive)

**Current state post-P0+P1+TDD:**
- P0 criticals fixed: probe Optional, iter_targets recursive, concat utf-8/newline guard, embed mapping ordering, chapter restoration, bitrate threading, cover preservation, atomic replace, sox trim clipping, ML lock/log_file=None/resampler cache/chunked 60s+2s
- P1: parallel preserves order, deduplicates _1 suffix, avoids overwriting pre-existing via versioning, backslash escaping, zero-duration skip + placeholder, keep-original copy not move, probe regex n/a without dB + sci notation, ML streaming via soundfile bounding RAM for mono48k prod path
- H1/H3/H4 fixed via TDD 7 new tests (red→green): atomic embed via mkstemp+replace, no-overwrite guard for bind_multiple and parallel_transcode, get_duration returns None not 0.0
- 25 tests passing (10 original + 8 new coverage + 7 TDD), 35s
- Hugo 41G 73 books already converted and in `/mnt/md0/share/audiobooks/Hugo Awards/` — verified basic+ml preserve cover/chapters on 10s clip

**Not merge-ready per wide-net:** H2 dedup low but final m4b still overwrites, cover bomb no size cap, symlink loop via glob **, batch abort on first corrupt file, ENOSPC partial backup, fork after torch threads warning, loudness -18 TP -1.5 LRA11 not ACX compliant, compand boosts noise, HPF80 thin, LPF14k dull, linear crossfade -3dB dip every 58s, GPU fade device mismatch, remaining OOM fallback path.

## 2. Current Bugs Inventory (post-TDD)

### Critical/High still open (from 4th review)

- **H1-atomic final (fixed in TDD but need verify):** bind now atomic via tmp.m4b + replace + size >1024. Clean already atomic. Need to ensure `transcode_mp3_to_m4a` still uses -y on temp dir only (safe), final embed no -y on ultimate file (now via tmp). Verify cover_tmpdir makedirs order bug fixed (was mkstemp before makedirs when tmpdir=None with nonexist output dir).
- **H2-dedup collision:** `Foo_1.mp3`, `Foo.mp3`, `Foo.mp3` → `Foo_1.m4a`, `Foo.m4a`, `Foo_1.m4a` duplicate race. Our used_names set + existing_on_disk avoidance fixes most, but still `while out_name in used_names or out_name in existing_on_disk` loop should check filesystem after versioning. Still possible race if external process creates file between check and worker write (TOCTOU). Low for temp dir, but final m4b dir shared across CLI runs could collide. Mitigation: `O_EXCL` create or FileExistsError raise, not silent versioning.
- **H3-overwrite:** parallel now versions to _1 instead of overwriting, preserving original (good). bind_multiple now skips existing with WARN. Still need `--overwrite` flag for intentional re-encode. Also `transcode_mp3_to_m4a` still uses -y, should use -n or remove -y and rely on Python guard.
- **H4-get_duration None:** now returns None, callers handle None (create_chapters skip, sox pipeline guard `if file_dur and file_dur>0`). Need explicit None check in `_sox_pipeline` fallback (currently lacks text=True).
- **M6 concat listfile symlink attack:** `output_m4b + ".concat.txt"` open w follows symlink. Should use mkstemp dir=final_dir + O_NOFOLLOW|O_EXCL.
- **ML GPU fade device mismatch:** streaming `linspace` CPU vs CUDA enh → RuntimeError on 2nd chunk GPU. Full-load fixed with device=, streaming not. Fix: `torch.linspace(..., device=overlap_buf.device)`.
- **ML final flush dead code:** overlap_buf flush after loop unreachable but left, can extend output by 2s if logic changes. Should remove or assert.
- **ML thread safety:** `_enhance_chunk` reads globals unlocked, `unload_model` deletes under lock while inference may be running.

### Medium

- **Probe 5s window only** — silent intro + loud rest mis-classified. Should probe whole file or middle 30s.
- **Cover bomb:** `extract_embedded_cover` unlimited APIC, `requests.get` cover no size cap, no Content-Length check. Cap 20MB, magic FFD8/89504E47.
- **Glob symlink loop:** `glob.glob recursive=True` follows symlink dirs → `**/*` with symlink to `/` enumerates FS. Use `Path.rglob(follow_symlinks=False)`.
- **Batch abort:** `_dispatch_clean` loops without try/except — one corrupt m4b aborts remaining 72. Wrap per-file.
- **ENOSPC backup:** `copy2` partial backup left. Should copy via temp + replace + size verify.
- **Placeholder chapter:** all zero-duration → dummy 1s chapter while audio may be longer (transcode fails earlier anyway). Should raise ValueError if all skipped in bind.
- **Publisher/album missing:** old injected album=title, album_artist, publisher via ffmetadata. New only title/artist. Regression.
- **Google source:** choices includes google but raises NotImplementedError traceback. Remove from choices or friendly sys.exit.
- **Bitrate no validation:** `64k; rm -rf` not injection (list args) but invalid causes ffmpeg rc 234 cryptic. Validate `^\d+[kKmM]?$`.

## 3. Audio Engineering Gaps (ACX / Auphonic)

Current: `loudnorm I=-18 TP=-1.5 LRA=11`, HPF 80Hz, LPF 14kHz, compand `0.3,1 6:-70,-60,-20 -5 -90 0.2`, noisered 0.21, single-pass loudnorm.

Pro standard (ACX new):
- Target `I=-19 TP=-2 LRA=7 dual_mono=true` two-pass with measured_I/LRA/TP/thresh/offset + linear=true. Single-pass causes pumping + ISP clipping. Our -18 hot, -1.5 fails ACX < -3, LRA 11 inconsistent.
- Mono loudness bug: mono measured -18 single-channel = -15 dual-mono perceived → 3dB too hot. Need dual_mono flag.
- Resample: basic decode 2ch 44.1k pcm16 -> sox -> loudnorm 44.1k ok. ML decode 1ch 48k -> loudnorm 44.1k = 48k->44.1k extra resample after DF3 aliases artifacts. Keep ML at 48k throughout or use soxr high precision.
- HPF 80Hz thin male, should be 65-70Hz 2nd-order. LPF 14k dull + double filtered by AAC 64k ~15-16k. Raise to 16-17k or remove for ML.
- Compand boosts noise floor in pauses (-70,-60) +10dB, attack 0.3s too slow. Replace with `acompressor threshold=-18 ratio=2.5 attack=20 release=250` + gate `agate threshold=-40 ratio=10`.
- No de-hum 50/60Hz notch cascade, no de-click `adeclick`, no de-ess, no de-plosive.
- Noisered 0.21 under-cleans hiss dominant tapes (-40dB). Conservative safe, but 0.27-0.30 or two gentle passes better. Single window polluted if breath captured. Use multi-window average + `afftdn` adaptive.
- Bitrate: basic stereo 64k = 32k/ch low, ML mono 64k = 64k/ch higher quality inconsistent. Unify mono 64k default, detect true stereo via correlation <0.98, bump to 96k stereo if true stereo.
- Intermediate wav pcm_s16le can clip if gain >0 after compand. Use pcm_f32le for intermediates.

## 4. ML — Is DF3 Right Approach? Naysayer View

**Pros per web (GitHub, arXiv 2305.08227, noisereducerai):** Full-band 48k, PESQ 3.5-4.0+ STOI >0.95, latency 10-20ms, 1.1M params, handles non-stationary babble/keyboard better than sox/RNNoise (~2.2 PESQ). Good for podcasts/calls, short audio 80-100ms stable.

**Cons for audiobook archive:**
- Destroys non-speech: music beds, chapter stings, phone filters, reverb thoughts, stereo panned narrators. Intro/outro music warbles. Breath over-suppression (prosody), sibilance loss >5kHz (only ERB masking above 4.8k), musical chirps on sustained vowels fatiguing over 10h.
- Chunked state reset: GRU hidden reset per 60s chunk → transient every 58s, linear crossfade dip -3dB hole every 58s, phase misalignment comb filtering.
- Training bias: DNS English VoIP 3-10s, not tape hiss steady-state (perfect for sox), no wow/flutter, 50Hz hum, vinyl. Multilingual sibilance suffers.
- Cost: torch 2GB wheel, 2-3GB VRAM per chunk, model download 100MB internet first run, OOM fallback path 6.9GB float32 for 10h if SR mismatch.
- Legality: model MIT but training data provenance unknown, ACX may forbid AI-altered for some contracts.

**Verdict:** DF3 good for non-stationary noise, better than RNNoise, but sox noiseprof more transparent for stationary hiss (majority of Hugo tapes). Keep dual mode, default basic safe, ml opt-in with warning, prefer Rust `deep-filter` binary (15-30MB, tract SIMD, no torch, built-in streaming) over Python torch wrapper to avoid 2GB dep and OOM.

## 5. Security / Reliability / Perf for 73 books 41G

- **Security:** Cover bomb 20MB cap + magic check, OpenLibrary cover Content-Length check, symlink attack listfile mkstemp O_NOFOLLOW, iter_targets ** glob loop detection, no timeout on ffmpeg/sox subprocess (add timeout=300)
- **Reliability:** Batch abort wrap per-file try/except in _dispatch_clean, disk-full preflight (need max temp = 3*wav ~19GB basic 10h, 10GB ml, check df -h), atomic backup via temp+replace+size verify, 0-byte file handling skip with warning, 1000 chapters ffprobe 1000 calls slow (100s) — consider batch
- **Perf:** 41GB at 64k = ~1423h audio. Basic: decode 50x RT (28h) + sox 5x (284h) + loudnorm 10x (142h) = ~454h CPU single core, ~3 days 8 cores if parallelized. ML GPU 10x RT = 142h GPU + 170h CPU overhead. CPU-only ML 0.5-1x = 1423-2846h (60-120 days). Need --jobs parallel for clean, use mono for basic to halve temp/CPU.
- **UX:** Missing --overwrite flag (now skips existing), keep-original versioning undocumented (.orig.1), skip-threshold misnamed (mean_volume not noise floor), pattern default *.m4b double-filtering, google traceback, no --dry-run/--jobs

## 6. Maintenance

- **Deps:** git+https openlibrary-client unpinned supply chain risk, mutagen/requests unpinned, deepfilternet unpinned 0.5.6, torch==2.6.0 pins CUDA 2GB but no CPU index, soundfile unpinned 0.13.1 libsndfile binary may break 3.13. Fix: hash-pinned requirements.txt, pip-tools/uv lock, pyproject.toml python_requires >=3.11,<3.13, torch CPU extra for CI
- **Python matrix:** Venv 3.12.12, system 3.13.5, no CI 3.11/3.13. torchaudio.backend.common.AudioMetaData warning will become ImportError in torch 2.8. ProcessPoolExecutor fork after torch threads → DeprecationWarning 72x, will raise in 3.14 (PEP 684). Fix: ThreadPoolExecutor for IO-bound ffmpeg or spawn context
- **Docs:** README says smoke only, now 25 tests including TDD. Missing mono ML output, chunk params, cache location ~/.cache/deepfilter..., atomicity, .orig versioning, overwrite skip behavior, bitrate semantics. Need update.
- **CLI breaking:** Old flat args `m4binder.py --mode multiple ...` used in run_hugo.sh/run_music.sh, new requires subcommand `bind`. Add compat shim: if first arg not in (bind,clean,-h,--help): insert bind + deprecation warn, or update scripts. Also single mode now requires --output-file (old auto folderName.m4b).
- **Tech debt:** print vs logging, ffmpeg_utils god-module (transcode+concat+ffmetadata+cover+probe+chapters), BindOptions mixes single/multiple, magic numbers scattered, cover always jpg even if png, publisher/album missing, google choice, requests no retry/UA.

## 7. Roadmap

### Phase 0 — Merge blockers (current)

- [x] P0 fixes: probe Optional, iter_targets recursive, concat utf-8/newline, embed mapping, chapter restoration, bitrate threading, cover preservation, atomic replace, sox trim clipping, ML lock/log_file=None/chunked
- [x] P1 fixes: parallel order preserved, dedup _1, no-delete pre-existing, backslash escaping, zero-duration skip, keep-original copy not move, probe regex n/a no dB, ML streaming soundfile
- [x] H1/H3/H4 TDD red→green: atomic tmp.m4b+replace+size>1024, no-overwrite guard bind_multiple skip, get_duration None not 0.0 — 7 new tests
- [ ] Fix remaining H1 order bug: cover_tmpdir makedirs before mkstemp when tmpdir=None + nonexist output dir (found secondary review)
- [ ] Fix H2 TOCTOU: replace -y with existence check + O_EXCL, add --overwrite flag to bind/clean, convert final embed to not rely on -y
- [ ] Fix ML GPU fade device mismatch: linspace device=overlap_buf.device, equal-power sin/cos crossfade
- [ ] Remove dead _atomic_embed_wrapper in bind.py
- [ ] Pin deps: openlibrary-client @ sha, mutagen==1.47.0, requests==2.32.3, deepfilternet==0.5.6, soundfile==0.13.1, torch cpu index for CI

### Phase 1 — Audio engineering compliance (1-2 weeks)

- [ ] Loudness: I=-19 TP=-2 LRA=7 dual_mono=true two-pass with measured_I/LRA/TP/thresh/offset, linear=true, docs ACX vs Auphonic -19 mono == -16 stereo
- [ ] HPF 70Hz, LPF 16kHz (or none for ML), remove compand or replace with acompressor 2.5:1 + agate, bump noisered 0.21→0.27 or add afftdn option nr=20 nf=-30
- [ ] De-hum notch cascade 60/120/180 or 50/100/150 + width 2 g=-40/-30/-20, add --hum auto|50|60 flag
- [ ] Bitrate unify mono 64k default, detect true stereo via astats correlation <0.98 → 96k stereo
- [ ] Intermediate pcm_f32le not pcm_s16le to avoid clipping
- [ ] Add --lpf/--hpf cli flags, --no-lpf

### Phase 2 — ML hardening (2-3 weeks)

- [ ] Streaming validation: sf.info(out).frames == total_frames ±overlap, fail if drift, remove dead final flush
- [ ] Thread safety: lock around enhance, resampler cache locked, unload_model gc+empty_cache per batch, document single-threaded
- [ ] Make streaming always (resample per chunk via soxr), never full-load torch.cat all chunks — incremental writer even in fallback
- [ ] Add Rust deep-filter binary backend: detection which deep-filter, subprocess wrapper cleanup_ml_rust.py 30 LOC, make requirements-ml.txt optional legacy
- [ ] Expose --chunk-s/--overlap-s cli, --device cpu|cuda, doc VRAM 2-3GB per 60s, model download size cache
- [ ] Fix loudnorm resample chain: keep ML at 48k final (48k AAC allowed) to avoid 48→44.1 double resample

### Phase 3 — Security / Reliability / UX (1 week)

- [ ] Cover bomb: cap 20MB, check Content-Length, magic, dimensions via ffprobe
- [ ] Symlink loop: Path.rglob follow_symlinks=False + visited inodes set, limit glob to max files
- [ ] Batch abort: try/except per file in _dispatch_clean, log continue, exit non-zero if any failed
- [ ] ENOSPC: copy backup via temp+replace+size verify, free-space preflight need max temp 190GB basic 100h book
- [ ] Subprocess timeout=300 for ffmpeg/ffprobe/sox
- [ ] --overwrite flag for bind/clean, --dry-run, --jobs N parallel for clean, --tmpdir option
- [ ] Listfile mkstemp O_NOFOLLOW|O_EXCL, not output.m4b.concat.txt
- [ ] Fix probe 5s window: remove -t 5 or -t 60 or scan middle, add probe_duration param

### Phase 4 — Maintenance / Docs / CI

- [ ] pyproject.toml + uv lock, python_requires >=3.11,<3.13, CI matrix 3.11/3.12 unit + ml job cached model ~/.cache/torch ~/.cache/deepfilternet
- [ ] Replace ProcessPoolExecutor with ThreadPoolExecutor for ffmpeg (IO-bound) or spawn context to fix fork warning
- [ ] Logging: logging.getLogger("m4binder") replace prints, --verbose/--quiet
- [ ] Split ffmpeg_utils god-module: transcode.py, chapters.py, probe.py, cover.py
- [ ] BindOptions split to BindSingleOptions/BindMultipleOptions, validate bitrate regex ^\d+[kKmM]?$
- [ ] Remove google from choices or implement with sys.exit friendly, add publisher/album/album_artist metadata
- [ ] README update: remove smoke only, document mono ML, chunk params, cache, atomicity, .orig versioning, skip-existing behavior, bitrate, system deps versions, ACX compliance
- [ ] MIGRATION.md for old flat args → subcommand, update run_hugo.sh/run_music.sh to use bind subcommand
- [ ] .gitignore add *.m4b *.orig.m4b, add CHANGELOG

## 8. Merge Checklist

Before merging feat/cleanup-merge → main:

- [ ] All P0+P1+H1/H3/H4 fixes committed (currently 8 modified files unstaged + 2 untracked tests)
- [ ] 25 tests green, plus 4 new tests for channel count, long ML streaming duration, sox EOF guard, probe None + backup .2 + google/bitrate (from review)
- [ ] Hugo 10s clip with cover: basic and ml both preserve cover/chapters, validated ffprobe Audio/Video/Chapters
- [ ] run_hugo.sh / run_music.sh updated to `m4binder.py bind ...` or compat shim added
- [ ] Requirements pinned with hash, torch CPU index for CI
- [ ] README updated, PLAN.md added
- [ ] .gitignore updated

## 9. Decision — Should we keep ML?

**Naysayer case strongest for archival principle least destructive first:** Sox basic with conservative 0.21 is more transparent on good tapes, preserves breaths/music, 0 deps, 18h -j4 for 73 books. DF3 better for non-stationary (page turns, AC cycling, babble) but alters voice timbre (breathless, metallic tail, lisped sibilance >5kHz, -3dB dip every 58s, mono'd, double-resampled).

**Recommendation:** Keep dual mode, default basic, ml opt-in with banner: "ML uses DeepFilterNet3 torch 2GB GPU 2-3GB VRAM slower may alter timbre use --keep-original spot-check 5min". Prefer Rust deep-filter binary if available to avoid torch OOM. For Hugo archive, run basic whole batch, manually ml only subset <10% that fails.

## 10. References

- DeepFilterNet GitHub: real-time, 48kHz, low complexity, frame 20ms, latency 10-20ms
- DF3 paper arXiv:2305.08227 Perceptually Motivated Real-Time Speech Enhancement
- noisereducerai.com comparison: DF3 PESQ 3.5-4.0+ STOI >0.95, short audio 80-100ms stable, DF2 3.17-3.5 PESQ 0.944, RNNoise 3.88 PESQ 0.92 (lighter)
- GitHub issues: no artifact search results (restricted)
- ACX: RMS -23 to -18, peak < -3, noise floor <-60, LUFS -19 typical, Auphonic -19 mono == -16 stereo, dual_mono compensation +3LU
