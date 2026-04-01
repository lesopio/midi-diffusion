import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import pretty_midi
import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from midigen.data import validate_midi_header


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("midi_prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def find_midi_files(input_dir: Path) -> List[Path]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fn in filenames:
            if fn.lower().endswith((".mid", ".midi")):
                files.append(Path(root) / fn)
    return files


def try_load_pretty_midi(path: Path) -> Tuple[pretty_midi.PrettyMIDI | None, str | None]:
    try:
        return pretty_midi.PrettyMIDI(str(path)), None
    except Exception as exc:
        return None, str(exc)


def repair_with_mido(src_path: Path, temp_dir: Path) -> Tuple[Path | None, str | None]:
    try:
        mid = mido.MidiFile(str(src_path))
        repaired = temp_dir / (src_path.stem + "_repaired.mid")
        mid.save(str(repaired))
        return repaired, None
    except Exception as exc:
        return None, str(exc)


def clean_pretty_midi(pm: pretty_midi.PrettyMIDI, min_note_length: float) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, int]]:
    stats = {
        "removed_drums": 0,
        "fixed_negative_start": 0,
        "fixed_zero_length": 0,
        "clamped_pitch": 0,
        "clamped_velocity": 0,
        "dropped_empty_instrument": 0,
        "dropped_invalid_notes": 0,
    }

    cleaned = pretty_midi.PrettyMIDI(initial_tempo=pm.estimate_tempo() or 120.0)
    try:
        cleaned.time_signature_changes = list(pm.time_signature_changes)
        cleaned.key_signature_changes = list(pm.key_signature_changes)
        cleaned.lyrics = list(pm.lyrics)
        cleaned.text_events = list(getattr(pm, "text_events", []))
    except Exception:
        pass
    try:
        times, tempi = pm.get_tempo_changes()
        cleaned._PrettyMIDI__tempo_changes = (times, tempi)
    except Exception:
        pass
    try:
        cleaned.resolution = pm.resolution
    except Exception:
        pass
    for inst in pm.instruments:
        if inst.is_drum:
            stats["removed_drums"] += 1
            continue
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=False, name=inst.name)
        for note in inst.notes:
            start = max(0.0, float(note.start))
            end = float(note.end)
            if start != note.start:
                stats["fixed_negative_start"] += 1
            if end <= start:
                end = start + min_note_length
                stats["fixed_zero_length"] += 1
            pitch = int(note.pitch)
            velocity = int(note.velocity)
            if pitch < 0:
                pitch = 0
                stats["clamped_pitch"] += 1
            elif pitch > 127:
                pitch = 127
                stats["clamped_pitch"] += 1
            if velocity < 1:
                velocity = 1
                stats["clamped_velocity"] += 1
            elif velocity > 127:
                velocity = 127
                stats["clamped_velocity"] += 1
            if end <= start:
                stats["dropped_invalid_notes"] += 1
                continue
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start,
                    end=end,
                )
            )
        new_inst.notes.sort(key=lambda n: (n.start, n.pitch))
        try:
            new_inst.control_changes = list(inst.control_changes)
            new_inst.pitch_bends = list(inst.pitch_bends)
        except Exception:
            pass
        if new_inst.notes:
            cleaned.instruments.append(new_inst)
        else:
            stats["dropped_empty_instrument"] += 1

    return cleaned, stats


def _process_file(
    src_path_str: str,
    out_path_str: str,
    min_note_length: float,
    max_size_mb: int,
) -> dict:
    src_path = Path(src_path_str)
    out_path = Path(out_path_str)
    result = {
        "source": str(src_path),
        "output": str(out_path),
        "status": "error",
        "message": "",
        "repaired": False,
        "header_ok": False,
        "loaded": False,
        "local_stats": None,
    }
    try:
        try:
            size = src_path.stat().st_size
            if size <= 0 or size > max_size_mb * 1024 * 1024:
                result["status"] = "skipped"
                result["message"] = f"Skip (size invalid): {src_path}"
                return result
        except Exception:
            pass

        header_ok = validate_midi_header(str(src_path), max_size_mb=max_size_mb)
        result["header_ok"] = header_ok

        pm, err = try_load_pretty_midi(src_path)
        repaired = False
        if pm is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                repaired_path, rep_err = repair_with_mido(src_path, Path(tmpdir))
                if repaired_path is None:
                    result["status"] = "error"
                    result["message"] = f"Load failed: {src_path} ({err}); repair failed ({rep_err})"
                    return result
                pm, err = try_load_pretty_midi(repaired_path)
                if pm is None:
                    result["status"] = "error"
                    result["message"] = f"Load failed after repair: {src_path} ({err})"
                    return result
                repaired = True

        result["repaired"] = repaired
        result["loaded"] = True
        cleaned, local_stats = clean_pretty_midi(pm, min_note_length=min_note_length)
        result["local_stats"] = local_stats

        if not cleaned.instruments:
            result["status"] = "empty"
            result["message"] = f"Skip (empty after clean): {src_path}"
            return result

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.write(str(out_path))
        result["status"] = "saved"
        if repaired and not header_ok:
            result["message"] = f"Header invalid, repaired via mido: {src_path}"
        elif repaired:
            result["message"] = f"Repaired and saved: {src_path} -> {out_path}"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["message"] = f"Error processing {src_path}: {exc}"
        return result


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def pack_zip(zip_path: Path, root_dir: Path, extra_files: List[Path]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        midi_dir = root_dir / "midi"
        for file_path in midi_dir.rglob("*"):
            if file_path.is_file():
                rel = file_path.relative_to(root_dir)
                zf.write(file_path, arcname=str(rel))
        for extra in extra_files:
            if extra.exists():
                rel = extra.relative_to(root_dir)
                zf.write(extra, arcname=str(rel))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="data/midi")
    parser.add_argument("--output-dir", type=str, default="data/cleaned_dataset")
    parser.add_argument("--zip-path", type=str, default="data/cleaned_dataset.zip")
    parser.add_argument("--min-note-length", type=float, default=0.05)
    parser.add_argument("--max-size-mb", type=int, default=10)
    parser.add_argument("--keep-structure", action="store_true")
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    zip_path = Path(args.zip_path)
    midi_out_dir = output_dir / "midi"
    midi_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "prepare.log"

    logger = setup_logger(log_path)
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Zip path: {zip_path}")

    files = find_midi_files(input_dir)
    if not files:
        logger.error("No MIDI files found.")
        sys.exit(1)

    out_map: Dict[Path, Path] = {}
    if args.keep_structure:
        for path in files:
            rel = path.relative_to(input_dir)
            out_map[path] = midi_out_dir / rel
    else:
        stem_counts: Dict[str, int] = {}
        for path in files:
            stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1
        for path in files:
            out_path = midi_out_dir / f"{path.stem}.mid"
            if stem_counts.get(path.stem, 0) > 1 or out_path.exists():
                digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
                out_path = midi_out_dir / f"{path.stem}_{digest}.mid"
            out_map[path] = out_path

    stats = {
        "total_files": len(files),
        "valid_header": 0,
        "loaded": 0,
        "repaired": 0,
        "saved": 0,
        "skipped": 0,
        "empty_after_clean": 0,
        "errors": 0,
    }
    repair_stats = {
        "removed_drums": 0,
        "fixed_negative_start": 0,
        "fixed_zero_length": 0,
        "clamped_pitch": 0,
        "clamped_velocity": 0,
        "dropped_empty_instrument": 0,
        "dropped_invalid_notes": 0,
    }

    manifest = []
    workers = max(1, int(args.workers))
    with tqdm(total=len(files), desc="Processing MIDI", unit="file") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for path in files:
                out_path = out_map[path]
                futures.append(
                    executor.submit(
                        _process_file,
                        str(path),
                        str(out_path),
                        args.min_note_length,
                        args.max_size_mb,
                    )
                )
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)

                if result.get("header_ok"):
                    stats["valid_header"] += 1

                status = result.get("status")
                if status == "skipped":
                    stats["skipped"] += 1
                elif status == "empty":
                    stats["empty_after_clean"] += 1
                elif status == "error":
                    stats["errors"] += 1
                elif status == "saved":
                    stats["saved"] += 1
                else:
                    stats["errors"] += 1

                if result.get("loaded"):
                    stats["loaded"] += 1

                if result.get("repaired"):
                    stats["repaired"] += 1

                local_stats = result.get("local_stats") or {}
                for k in repair_stats:
                    repair_stats[k] += local_stats.get(k, 0)

                msg = result.get("message")
                if msg:
                    logger.info(msg)

                if status == "saved":
                    manifest.append(
                        {
                            "source": result.get("source"),
                            "output": result.get("output"),
                            "repaired": bool(result.get("repaired")),
                            "local_fixes": local_stats,
                        }
                    )

    summary = {
        "stats": stats,
        "repairs": repair_stats,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }

    write_manifest(output_dir / "manifest.json", {
        "items": manifest,
        "summary": summary,
    })
    write_manifest(output_dir / "stats.json", summary)

    if not args.skip_zip:
        extra = [
            output_dir / "manifest.json",
            output_dir / "stats.json",
            output_dir / "prepare.log",
        ]
        pack_zip(zip_path, output_dir, extra)
        logger.info(f"Packed zip: {zip_path}")

    logger.info(f"Summary: {stats}")
    logger.info(f"Repairs: {repair_stats}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
