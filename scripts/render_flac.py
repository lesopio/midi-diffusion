import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pretty_midi


def _find_midi_files(midi_dir: Path) -> List[Path]:
    files: List[Path] = []
    for root, _, names in os.walk(midi_dir):
        for name in names:
            if name.lower().endswith((".mid", ".midi")):
                files.append(Path(root) / name)
    return sorted(files)


def _safe_out_path(out_dir: Path, stem: str) -> Path:
    out_path = out_dir / f"{stem}.flac"
    if not out_path.exists():
        return out_path
    suffix = 1
    while True:
        candidate = out_dir / f"{stem}_{suffix}.flac"
        if not candidate.exists():
            return candidate
        suffix += 1


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-6:
        return audio
    return audio / peak


def _write_flac(audio: np.ndarray, sample_rate: int, out_path: Path) -> Optional[str]:
    try:
        import soundfile as sf  # type: ignore

        sf.write(str(out_path), audio, sample_rate, format="FLAC")
        return None
    except Exception as exc:
        return str(exc)


def _write_wav_then_convert(audio: np.ndarray, sample_rate: int, out_path: Path) -> Optional[str]:
    try:
        from scipy.io import wavfile  # type: ignore

        wav_path = out_path.with_suffix(".wav")
        audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        wavfile.write(str(wav_path), sample_rate, audio_i16)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return "ffmpeg not found; wrote wav instead"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(wav_path),
            str(out_path),
        ]
        result = os.system(" ".join(cmd))
        if result != 0:
            return "ffmpeg conversion failed"
        wav_path.unlink(missing_ok=True)
        return None
    except Exception as exc:
        return str(exc)


def _render_one(path: Path, out_path: Path, sample_rate: int, soundfont: Optional[str], normalize: bool) -> Optional[str]:
    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        return f"load failed: {exc}"
    try:
        if soundfont:
            audio = pm.fluidsynth(fs=sample_rate, sf2_path=soundfont)
        else:
            audio = pm.fluidsynth(fs=sample_rate)
    except Exception as exc:
        return f"fluidsynth failed: {exc}"
    if normalize:
        audio = _normalize_audio(audio)
    err = _write_flac(audio, sample_rate, out_path)
    if err is None:
        return None
    err2 = _write_wav_then_convert(audio, sample_rate, out_path)
    if err2 is None:
        return None
    return f"flac write failed: {err}; wav fallback failed: {err2}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi-file", type=str, default="")
    parser.add_argument("--midi-dir", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="outputs_flac")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    midi_file = args.midi_file.strip()
    midi_dir = args.midi_dir.strip()
    if not midi_file and not midi_dir:
        print("[render] missing --midi-file or --midi-dir")
        return 1

    if midi_file and midi_dir:
        print("[render] provide either --midi-file or --midi-dir")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    soundfont = args.soundfont.strip() or os.environ.get("SOUNDFONT", "")
    if soundfont:
        sf_path = Path(soundfont)
        if not sf_path.exists():
            print(f"[render] soundfont not found: {sf_path}")
            return 1
        soundfont = str(sf_path)

    if midi_file:
        files = [Path(midi_file)]
    else:
        files = _find_midi_files(Path(midi_dir))

    if not files:
        print("[render] no midi files found")
        return 1

    total = len(files)
    success = 0
    for idx, path in enumerate(files, start=1):
        if not path.exists():
            print(f"[render] missing: {path}")
            continue
        out_path = _safe_out_path(out_dir, path.stem)
        err = _render_one(path, out_path, args.sample_rate, soundfont or None, args.normalize)
        if err is None:
            success += 1
            print(f"[render] {idx}/{total} ok -> {out_path}")
        else:
            print(f"[render] {idx}/{total} failed: {path} ({err})")

    print(f"[render] done: {success}/{total}")
    return 0 if success > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
