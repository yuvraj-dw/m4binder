# m4binder

Audiobook toolkit: `bind` mp3 folders → m4b, `clean` an existing m4b in place,
`migrate` mp3 folders → cleaned m4b in a single lossy encode, `tag` write
metadata and cover art onto a finished m4b without re-encoding it.

Cleaning runs [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) over
the whole book, then normalises loudness to broadcast targets. On a 24-core
machine a 2.9-hour audiobook cleans in about 8 minutes.

## Install

```bash
pip install -e .                 # bind only — no ML dependencies
pip install -e ".[ml]"           # adds clean/migrate (torch, deepfilternet)
```

Needs `ffmpeg` and `ffprobe` (>= 7) on PATH.

torch ships as separate CPU and CUDA builds from different indexes — they are
not a base package plus an extra. On a machine without an NVIDIA GPU, install
the CPU wheel first and it will be kept:

```bash
pip install torch==2.6.0+cpu torchaudio==2.6.0+cpu \
  --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml]"
```

The DeepFilterNet3 model (~10MB) downloads to `~/.cache` on first run.

## Quick start

```bash
# Clean an m4b in place. Defaults are the tuned settings; no knobs needed.
m4binder.py clean --input "Some Book.m4b"

# A directory of them
m4binder.py clean --input /audiobooks/ --pattern "*.m4b"

# mp3 folder -> cleaned m4b, one lossy encode rather than two
m4binder.py migrate --input-folder ./Dune/mp3s --output-file ./Dune.m4b

# Bind without cleaning
m4binder.py bind --mode multiple --input-folder /library --output-folder /m4bs

# Tag a finished m4b — no re-encode, audio untouched
m4binder.py tag --file "Some Book.m4b" --metadata-json tags.json --cover cover.jpg
```

`clean` **overwrites the file in place**, keeping the source as
`<name>.orig.m4b` first. That backup is **on by default** — pass
`--no-keep-original` to skip it and save the disk.

The default is on because there is a known class of book this pipeline makes
*worse*: DeepFilterNet3 treats a music bed as noise, and in the round-9 blind
test the untouched file took first place on the one book with music, ahead of
every processed variant. The damage is silent — the book still plays. A backup
is cheap and reversible; an over-suppressed score is neither.

## What cleaning does

Decode to mono 48k → DeepFilterNet3 at `atten_lim=12` → EBU R128 loudness
measurement → single AAC encode at 64k with the loudness gain applied → mux,
preserving chapters and cover art.

**Both defaults on this line are now contradicted by a blind test (F-79, F-80),
2026-08-08.** Seven attenuations on five books: on every noisy book the ranking is
monotone in attenuation and `12` places **5th of 7**, while on the one book with a
music bed *untouched* wins outright and `12` places 4th. The right value is
content-dependent and 12 is wrong in both directions. Separately, `--pf` on top of
24 dB beat plain 24 dB on **4 of 4** books — the "over-attenuates, leave off" note
this table carried had no measurement behind it. Neither default has been changed
yet; that is a product call, and it is four books and one listener.

**Why `atten_lim=12`.** The dB figure is the noise attenuation: 12 dB passes
25.1% of the original signal, so the residual hiss sits 12 dB down. Peter reports
0→6 and 6→12 as audible steps and no difference past 12 — one listener's
recollection, not a controlled test. What *is* controlled is F-28's blind test on
three books: 12 > 24 > 29, i.e. quality falls as attenuation rises, and on 2 of
the 3 books the setting was inaudible at all. Higher
values put a larger share of *model output* in the mix, which increases exposure
to the model's own artifacts (it treats music beds and chapter stings as noise),
so 12 is the least model exposure that makes the hiss inaudible.

**Loudness** targets `I=-19 LUFS`, `TP=-2 dBTP`, `dual_mono=true`, mono 64k.
(These are ACX-*like*, not ACX-compliant: ACX requires peaks at or below −3 dBFS
and this ceiling is −2 dBTP. Called "ACX-compatible" here until 2026-08-07. The
parameters are deliberate; only the compliance claim was wrong.) The gain is capped so true peak never exceeds the ceiling, which
means a quiet-but-peaky recording is attenuated rather than amplified.

## Options

### `clean` / `migrate`

| Flag | Default | Notes |
|---|---|---|
| `--atten-lim` | `12` | dB, or `auto`. **`auto` is not recommended — see below** |
| `--pf` | off | Model post-filter. **Try turning it on** — see below |
| `--keep-original` | **on** | Save the source as `.orig.m4b` before replacing. `--no-keep-original` to disable |
| `--backend` | `df3` | Enhancement backend |
| `--device` | `cpu` | `cuda` is tested and is the *faster* path (F-29, F-31) |
| `--jobs-df3` | 12 | Enhancement workers |
| `--jobs-loudness` | 8 | Loudness measurement workers |
| `--jobs-encode` | 1 | See Known issues before raising |
| `--dry-run` | | List what would be cleaned |

Worker defaults come from an early ad-hoc sweep on this box: enhancement and
loudness are memory-bandwidth-bound and saturate around 8-12 workers, and pushing
to 24 made each worker 3x slower for only 26% more aggregate throughput.

**The later, properly-repeated benchmark does not reproduce that 26%.** On
`before_after/bench2060` (3 runs each, 300 s of audio, same box) the aggregate DF3
speedups were w1 6.9x, w4 25.4x, **w8 42.6x, w12 38.3x, w24 42.0x** — 8 → 24 is
−1.5%, and the shipped default of 12 is nominally the *worst* of {8, 12, 24}. The
run-to-run spread at w24 is 32-67x, i.e. wider than the differences between
settings, so the honest reading is that **8, 12 and 24 are indistinguishable at
this sample size** and none of them is a measured optimum. Do not treat these
defaults as tuned; they are reasonable and untuned. Re-benchmarking on a longer
book is an open thread (F-31).

### `bind`

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `single` | `single` = one book; `multiple` = one book per subfolder |
| `--input-folder` | | required |
| `--output-file` / `--output-folder` | | per `--mode` |
| `--title` / `--author` | | written into the container |
| `--bitrate` | `64k` | ignored when `--audio-mode copy` |
| `--chapters-file` | derived | an ffmetadata file of chapter markers — see below |
| `--audio-mode` | `transcode` | `copy` muxes the mp3s in untouched — see below |
| `--overwrite` | off | allow replacing existing output |
| `--clean` | off | bind, then clean in one step |

**`--chapters-file` — when one mp3 is not one chapter.** By default `bind` derives
chapter markers from the mp3 boundaries, using each file's duration and ID3
title. That is right when the files *are* the chapters and wrong when they are
arbitrary parts, which is what Libby/OverDrive loans give you — a book split into
equal-length chunks that cut mid-sentence. Pass an
[ffmetadata](https://ffmpeg.org/ffmpeg-formats.html#Metadata-1) file to supply the
real markers instead:

```ini
;FFMETADATA1
[CHAPTER]
TIMEBASE=1/1000
START=0
END=754000
title=Chapter One
```

The file is handed to ffmpeg as-is; a malformed one surfaces as an ffmpeg error
rather than a friendly one.

**`--audio-mode copy` — skip the re-encode.** The default transcodes the mp3s to
AAC at `--bitrate`. If the source is already a lossy 64k mp3, that is a second
generation of loss to reach the same bitrate, and `copy` muxes the mp3 streams
into the m4b container untouched — bit-exact and much faster.

The cost is compatibility: **mp3-in-MP4 is legal but not universal**, and players
that expect an audiobook to be AAC (Apple Books among them) may refuse it. Use it
when you control the player, not for a library you want to open anywhere.

Two container details this path handles, both of which bit during development and
are easy to reintroduce:

- ffmpeg infers the `ipod` muxer from the `.m4b` extension and that muxer
  **refuses to mux an mp3 stream** (`Could not find tag for codec mp3`). `copy`
  pins `-f mp4`, which accepts mp3 and aac alike.
- plain `-f mp4` then defaults the brand to a generic `isom`, so `copy` also pins
  `-brand "M4A "`. The brand is how iTunes/Apple Books recognise an
  audiobook-shaped file, and losing it is silent — the book still builds, still
  plays, still passes CI. `tests/test_bind_options.py` asserts both `major_brand`
  and `compatible_brands` on the **default** path for exactly this reason.

## Tagging a finished m4b

```bash
m4binder.py tag --file "Some Book.m4b" --metadata-json tags.json \
    --cover cover.jpg --cover-format jpeg
```

`--metadata-json` is a JSON object of **generic** tag names; the mapping to MP4
atoms lives in `m4b_lib/tagging.py` so callers never touch `\xa9nam` or
`----:com.apple.iTunes:`:

```json
{
  "title": "Some Book",
  "artist": "The Author",
  "composer": "The Narrator",
  "series": "A Series",
  "series-part": "2",
  "isbn": "9780000000000"
}
```

Known names (`title`, `artist`, `album`, `album_artist`, `composer`, `date`,
`genre`, `comment`, `description`, `copyright`, …) go to their standard atoms;
**anything else becomes an iTunes freeform atom**, which is how arbitrary keys
like `isbn` survive at all. `composer` is where audiobook players look for the
narrator. Empty values are skipped rather than written blank — an empty tag looks
authoritative and is worse than an absent one.

`series` and `series-part` are written **twice**, to a standard atom and a
freeform one, because readers disagree about where a series index lives. A
non-numeric index like `"1.5"` cannot go in the integer movement atom and
survives via the freeform copy alone.

**It edits atoms in place with mutagen, not ffmpeg**, for two reasons found the
hard way:

- ffmpeg forces an either/or. `-movflags use_metadata_tags` is required for
  freeform atoms and **silently drops cover art** — exit 0, no warning. Omitting
  it keeps the cover and discards every non-standard key. Two passes do not help:
  the second drops what the first wrote.
- re-muxing rewrites hundreds of megabytes to change a few hundred bytes. In
  place, a 477 MB book tags in milliseconds and the audio is byte-identical by
  construction rather than by verification.

**Caveat:** the save is in place, so a failure partway through — power loss, a
full disk — can leave the file damaged. There is no temp-file-then-rename here as
there is in the bind path. Tag a copy if the only copy is precious.

## Performance

Measured on 2.9 hours of real audiobook, 24 cores:

| Stage | Time | Realtime factor |
|---|---|---|
| decode | 0.2 min | 1082x |
| enhance (DF3) | 4.0 min | 44x |
| loudness | 0.9 min | 201x |
| encode + mux | 3.4 min | 52x |
| **total** | **8.4 min** | **21x** |

A 30-hour audiobook is roughly 90 minutes.

**The encoder is the wall, not the model.** Enhancement can be made infinitely
fast and the pipeline still only reaches 40.1x, because the AAC encode is serial
at 52x. Measured alternatives to the 44x enhance row above, on an RTX 2060 SUPER:

| enhancement | overall | full ~19,700 h library |
|---|---|---|
| DF3, CPU pool (12 workers) | 19.6x | 41.9 d |
| **DF3 on the GPU** | **23.0x** | **35.7 d** |
| enhancement infinitely fast | **40.1x** | **20.5 d** |

**15.8 of the 20.5-day floor is pure encode.** The 52x figure was inherited
unverified for most of this project's life. Re-measured 2026-08-07 at the
shipped default (`--jobs-encode 1`, one stream part): **encode + mux runs at
62.6x**, encode alone at 64.7x, and a monolithic ffmpeg call at 66.8x — so there
is no per-part overhead worth removing (F-52,
`before_after/encode_bench/results.json`). **52x is pessimistic by ~20%**, which
means every library estimate on this page overstates the encode wall rather than
understating it. The 4.0-minute enhance row may still have been taken on a
CUDA-visible box, in which case it was a GPU measurement mislabelled as CPU
(F-29).

## Known issues

**Parallel encoding leaves audible joins.** `--jobs-encode` above 1 encodes
segments separately and concatenates them. Every AAC encoder emits priming
samples and pads its final frame, so each join carries a −40 dB dip about 35ms
wide — audible in narration. The default is 1 (a single continuous encode),
which is correct by construction. It costs about 3.4 min per 2.9 h of audio; the shipped
single-part encode measures 62.6x (F-52) — roughly 35 min on a 30-hour book. (`scheduler.py:41`
says "55 min rather than 5 on a 30h book"; that comment is the stale one.) Do
not raise it until the join is gapless.

**`--device` was broken in both directions until 2026-07-29, and the fix matters
even if you never passed the flag.** `--device cuda` was a hard `TypeError` —
DF3 runs its analysis on the host, so moving the input tensor to the GPU crashed
it. Worse, `--device cpu` *silently ran on the GPU*: `init_df()` moves the model
to `get_device()`, which auto-selects `cuda:0` whenever one is visible, and 12
workers at ~0.66 GB each exhausted an 8 GB card partway through a book. That is
the **default** invocation. It was latent only because this venv holds
`torch+cpu`, while `requirements-ml.txt` pins bare `torch==2.6.0` — the CUDA
build on Linux. Both fixed; see F-29.

**The GPU path is now the faster one, and it is not serial by batch.** Measured on
an RTX 2060 SUPER: DF3 at 54.1x against 38.3x for the 12-worker CPU pool — 41.9
→ 35.7 days for the full library. `_run_stream` builds batch 1 and GPU jobs
run serially, so the GPU never sees a batch > 1 — `scheduler.py:8` says so
directly now, having previously claimed the opposite. The measured
numbers are what actually ships. (`scheduler.py:8` now states this correctly;
the "claim" described above was fixed after F-31 and no longer exists in the
code.)

**Already-quiet files are not skipped.** Every file gets a full lossy
generation regardless of whether it needs cleaning.

**Do not use `--atten-lim auto`. Use the default of 12.** Its range is also wider
than this README used to say — the Options table claimed "10-17 typical", but the
estimator clamps to **6-30** and measured 6/11/30 (min/median/max) over 60 books;
running it over the 10 books in `before_after/auto_sample/` gives 7, 10, 10, 10,
14, 15, 17, 19, 20, 27. Four of ten sit outside "10-17", and the top of the range
is where it sounds worst.

A blind listening test
on 2026-07-29 settled it: across three books at 12 / 24 / 29 dB, two were
*indistinguishable* at every setting, and on the one where the setting was
audible — The Dispossessed, which has music — the ranking was strictly **12 > 24 >
29**, and `auto` chose 29. It picked the worst available option on the only book
that could tell the difference.

The cause is not a bug left in the estimator. `auto` keys off the **gap floor**,
and F-6 had already measured that gap floor is *anti-correlated* with Peter's
judgement on this exact book: a loud, compressed master reads as noisy on every
level-based measure. The two fixes below made `auto` faithfully implement a bad
feature. A better gap measurement would make it more precisely wrong.

The flag is kept, since an explicit value is sometimes wanted and the estimator is
at least self-consistent now, but it is not the recommended path and must not
become the default. History of the two fixes follows.

**The sign was inverted, and the measurement was contaminated.** The sign —
`optimal = snr + margin` meant attenuation *rose* with signal-to-noise ratio, so
a clean file asked for more processing than a noisy one; it is now the shortfall
against a target SNR, so a noisier file gets a larger limit. And the noise
measurement was contaminated: each silence window was measured with `-ss/-t`,
which bounds the muxer and not the filter, so `volumedetect` averaged in the
speech that follows the gap. On a quiet gap that read −33 dB where the gap was
actually −88 dB, which flattened every file to roughly the same SNR. Windows are
now measured with `atrim`, and the estimate tracks a synthesised SNR to within
about 1 dB from 15 to 60 dB.

What is *not* proven: the target the estimator aims at (`target_snr_db`, 51 dB)
is an anchor chosen so that a median library book — about 42 dB of gap SNR —
lands on the listening-validated default of 12 dB. Nobody has listened to the
files at the values it now picks. It still decodes at 16 kHz, which is deaf
above 8 kHz; it still samples a single 60-second window from a file whose noise
can vary by 46 dB; it measures the noise floor in the *gaps*, which says little
about hiss under speech; and on a file so noisy that `silencedetect` finds no
window at all it falls back to a hardcoded 20 dB rather than to the aggressive
end. An explicit value is still the predictable choice.

~~**Cleaning a directory also re-cleans its own backups.**~~ **Fixed
2026-08-10.** The directory sweep matched the `.orig.m4b` files a previous run
had written and cleaned those too, putting a second lossy generation on the file
whose whole purpose is to be pristine. `iter_targets` now skips `.orig.m4b` and
the `.orig.N.m4b` collision form. A backup named explicitly (`clean --input
Book.orig.m4b`) is still honoured — skipping is for the sweep, not for a direct
instruction — and a book actually called `Original Sin.m4b` is not mistaken for
one.

**DeepFilterNet3 destroys music.** It was trained to treat everything
non-speech as noise, so intro themes, chapter stings and music beds are removed
along with the hiss. Measured on one audiobook's music passage: 29 dB gone from
the low band, 47 dB from the midrange, 0.21 correlation with the original. At
the default `atten-lim 12` the 25% dry blend masks most of it; at higher
settings it does not.

**Books longer than 12.43 hours were silently truncated** (fixed). Intermediate
audio was written as WAV, whose 32-bit RIFF size fields cap a file at 4 GiB —
exactly 44739.24 s of mono 16-bit 48 kHz. Three separate books produced output
of exactly 44739.3 s. Now written as Wave64. If you cleaned a long book with an
earlier version, check its duration.

## Tests

```bash
pytest tests/ -v
```

ML-dependent tests are marked `ml` and skip cleanly if the extra is not
installed. CI installs it and fails if they skip.

## Before/after comparisons

The A/B sets behind the tuning decisions — the `atten-lim` sweep the 12 dB
default came from, the `auto` estimator sample, the parallel-encode join defect,
and the algorithm comparison — moved to
[m4binder-research](https://github.com/patricker/m4binder-research) along with
the rest of the measurement work. They were never tracked here; `before_after/`
has always been gitignored.
