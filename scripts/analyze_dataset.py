import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from midigen.data import build_time_grid, midi_to_pianoroll, scan_midi_files


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + 1e-8)
    q = q / (q.sum() + 1e-8)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log((a[mask] + 1e-12) / (b[mask] + 1e-12))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def analyze_dir(data_dir: str, steps_per_beat: int, max_seq_len: int | None, sample_limit: int) -> dict:
    files, stats = scan_midi_files(data_dir)
    if sample_limit > 0:
        files = files[:sample_limit]

    lengths = []
    tempi = []
    pitch_hist = np.zeros(128, dtype=np.int64)
    polyphony = []
    note_counts = []

    for path in tqdm(files, desc=f"Analyze {data_dir}", unit="file"):
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception:
            continue
        try:
            tempo = pm.estimate_tempo()
            if tempo and tempo > 0:
                tempi.append(float(tempo))
        except Exception:
            pass

        grid = build_time_grid(pm, steps_per_beat)
        steps = max(1, len(grid) - 1)
        if max_seq_len is not None:
            steps = min(steps, max_seq_len)
        lengths.append(steps)

        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if 0 <= note.pitch <= 127:
                    pitch_hist[note.pitch] += 1
        try:
            pr = midi_to_pianoroll(pm, steps_per_beat)
            if max_seq_len is not None:
                pr = pr[:max_seq_len]
            if pr.size > 0:
                polyphony.extend(pr.sum(axis=1).tolist())
                note_counts.append(int(pr.sum()))
        except Exception:
            pass

    def _summary(arr):
        if not arr:
            return {}
        a = np.array(arr, dtype=np.float64)
        return {
            "count": int(len(a)),
            "mean": float(np.mean(a)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    return {
        "file_stats": stats,
        "lengths": _summary(lengths),
        "tempi": _summary(tempi),
        "polyphony": _summary(polyphony),
        "note_counts": _summary(note_counts),
        "pitch_hist": pitch_hist.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--compare-dir", type=str, default="")
    parser.add_argument("--steps-per-beat", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--out", type=str, default="dataset_report.json")
    args = parser.parse_args()

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    report = {"primary": analyze_dir(args.data_dir, args.steps_per_beat, max_seq_len, args.sample_limit)}

    if args.compare_dir:
        report["compare"] = analyze_dir(args.compare_dir, args.steps_per_beat, max_seq_len, args.sample_limit)
        p = np.array(report["primary"]["pitch_hist"], dtype=np.float64)
        q = np.array(report["compare"]["pitch_hist"], dtype=np.float64)
        report["pitch_js_divergence"] = js_divergence(p, q)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
