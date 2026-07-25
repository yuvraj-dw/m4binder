"""One-step migrate: mp3 chapters → cleaned m4b in single lossy encode.

Two architectures:
- A) Simple (bind then clean) — double lossy, reuses existing code, 20 LOC
- B) Optimal single lossy (implemented here) — mp3 decode → wav concat → clean wav → m4b encode once

B is better quality: cleaner sees raw tape hiss not AAC artifacts, single AAC encode.

For v2.0 single best set: ml-rust 12dB is default, no fallback.
"""

import os
import tempfile
from typing import List

from m4b_lib import ffmpeg_utils, metadata
from m4b_lib.bind import BindOptions, _natural_key
from m4b_lib.cleanup import _decode_to_wav, _loudness_normalize, _sox_pipeline_v2, _afftdn_pipeline, _hybrid_pipeline


def _list_mp3s(folder: str) -> List[str]:
    return sorted(
        (os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp3")),
        key=lambda p: _natural_key(os.path.basename(p)),
    )


def _bind_and_clean_one(input_folder: str, output_m4b: str, bind_opts: BindOptions,
                        clean_atten_lim: str = "12", clean_pf: bool = False,
                        keep_original: bool = False) -> None:
    """One-step: mp3 folder -> cleaned m4b, single lossy encode.

    Steps:
    1. List mp3s natural sort
    2. Resolve metadata (title/author/cover)
    3. Parallel decode mp3s -> wavs (1ch 48k for ml-rust best)
    4. Concat wavs -> combined.wav
    5. Create chapters ffmetadata from mp3 durations
    6. Clean combined.wav via ml-rust best (12dB) -> cleaned.wav
    7. Loudnorm two-pass -19/-2/7 mono -> normalized.wav
    8. Embed chapters+cover+normalized -> final m4b atomic

    If keep_original True and output exists, preserves original as .orig.m4b (not applicable for migrate first time).
    """
    mp3s = _list_mp3s(input_folder)
    if not mp3s:
        raise ValueError(f"No mp3 files in {input_folder}")

    meta = metadata.resolve_metadata(
        bind_opts.metadata_source, bind_opts.title, bind_opts.author, first_mp3=mp3s[0]
    )

    # Use same FS as output for atomic replace safety
    output_parent = os.path.dirname(os.path.abspath(output_m4b))
    os.makedirs(output_parent, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="m4b_migrate_", dir=output_parent) as tmp:
        wav_dir = os.path.join(tmp, "wavs")
        os.makedirs(wav_dir, exist_ok=True)

        # Parallel decode mp3s -> wavs, 1ch 48k for ml-rust best
        # Reuse ffmpeg_utils decode but for mp3 input: we can use _decode_to_wav which works for any container
        wavs = []
        for mp3 in mp3s:
            base = os.path.splitext(os.path.basename(mp3))[0]
            wav_path = os.path.join(wav_dir, base + ".wav")
            _decode_to_wav(mp3, wav_path, channels=1, sample_rate=48000)
            wavs.append(wav_path)

        combined_wav = os.path.join(tmp, "combined.wav")
        # Concat wavs via ffmpeg concat demuxer (need listfile)
        # Reuse concat logic but for wav
        listfile = os.path.join(tmp, "wav_concat.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for w in wavs:
                safe = w.replace("\\", "\\\\").replace("'", r"'\''")
                f.write(f"file '{safe}'\n")

        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", combined_wav],
            check=True, capture_output=True, timeout=300,
        )

        # Chapters from mp3 durations
        chapters_ini = os.path.join(tmp, "chapters.ini")
        ffmpeg_utils.create_chapters_ffmetadata(mp3s, chapters_ini)

        # Clean combined wav with best pipeline (ml-rust 12dB)
        cleaned_wav = os.path.join(tmp, "cleaned.wav")
        # Use Rust backend directly for best
        try:
            from m4b_lib.cleanup_ml_rust import _find_binary, rust_enhance
            binary = _find_binary()
            if binary is None:
                raise FileNotFoundError("deep-filter binary not found")
            # Handle auto string if passed
            atten = clean_atten_lim
            if isinstance(atten, str) and atten.lower() == "auto":
                # Estimate optimal from combined wav
                optimal, info = ffmpeg_utils.estimate_optimal_atten_lim(combined_wav)
                if optimal is not None:
                    print(f"  [auto] SNR {info.get('snr'):.1f}dB -> atten-lim {optimal}dB")
                    atten = optimal
                else:
                    atten = 12.0
            else:
                try:
                    atten = float(atten)
                except:
                    atten = 12.0
            print(f"  cleaning with ml-rust atten-lim {atten}dB")
            rust_enhance(combined_wav, cleaned_wav, atten_lim_db=atten, pf=clean_atten_lim if isinstance(clean_atten_lim, bool) else clean_pf)
        except Exception as e:
            print(f"  Rust backend failed ({e}), falling back to afftdn")
            _afftdn_pipeline(combined_wav, cleaned_wav)

        # Loudnorm two-pass
        normalized_wav = os.path.join(tmp, "normalized.wav")
        _loudness_normalize(cleaned_wav, normalized_wav, channels=1)

        # Embed to final m4b
        cleaned_m4b_tmp = os.path.join(tmp, "cleaned.m4b")
        ffmpeg_utils.embed_chapters_and_meta(
            normalized_wav, chapters_ini=chapters_ini, cover_bytes=meta.cover_bytes,
            output_m4b=cleaned_m4b_tmp, tmpdir=tmp, copy_audio=False,
            title=meta.title, author=meta.author, bitrate=bind_opts.bitrate,
        )

        # Validate and atomic replace
        if not os.path.exists(cleaned_m4b_tmp) or os.path.getsize(cleaned_m4b_tmp) < 1024:
            raise RuntimeError(f"migrate produced invalid output size {os.path.getsize(cleaned_m4b_tmp) if os.path.exists(cleaned_m4b_tmp) else 'missing'}")

        # Handle keep_original if output exists and requested
        if keep_original and os.path.exists(output_m4b):
            import shutil
            backup = os.path.splitext(output_m4b)[0] + ".orig.m4b"
            if os.path.exists(backup):
                base, ext = os.path.splitext(backup)
                i = 1
                while os.path.exists(f"{base}.{i}{ext}"):
                    i += 1
                backup = f"{base}.{i}{ext}"
            shutil.copy2(output_m4b, backup)
            print(f"  original kept at {backup}")

        os.replace(cleaned_m4b_tmp, output_m4b)

    print(f"Created cleaned audiobook: {output_m4b}")


def bind_and_clean_single(bind_opts: BindOptions, clean_atten_lim: str = "12", clean_pf: bool = False, keep_original: bool = False) -> None:
    if not bind_opts.output_file:
        raise ValueError("bind_and_clean_single requires output_file")
    _bind_and_clean_one(bind_opts.input_folder, bind_opts.output_file, bind_opts, clean_atten_lim, clean_pf, keep_original)


def bind_and_clean_multiple(bind_opts: BindOptions, clean_atten_lim: str = "12", clean_pf: bool = False, keep_original: bool = False) -> None:
    out_dir = bind_opts.output_folder or bind_opts.input_folder
    os.makedirs(out_dir, exist_ok=True)
    for entry in sorted(os.listdir(bind_opts.input_folder)):
        sub = os.path.join(bind_opts.input_folder, entry)
        if not os.path.isdir(sub):
            continue
        if not any(f.lower().endswith(".mp3") for f in os.listdir(sub)):
            print(f"[WARN] No MP3 files in subfolder: {sub}. Skipping.")
            continue
        out = os.path.join(out_dir, entry + ".m4b")
        if os.path.exists(out) and not bind_opts.overwrite:
            print(f"[WARN] Output {out} exists, skipping (use --overwrite)")
            continue
        print(f"[INFO] Migrating (bind+clean) subfolder: {sub} -> {out} (atten-lim {clean_atten_lim}dB)")
        try:
            _bind_and_clean_one(sub, out, bind_opts, clean_atten_lim, clean_pf, keep_original)
        except Exception as e:
            print(f"[ERROR] Failed on {sub}: {e}")
        else:
            print(f"[INFO] Finished {out}")
