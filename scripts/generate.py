import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from midigen.sample import generate_tokens, save_midi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max-voices", type=int, default=4)
    parser.add_argument("--steps-per-beat", type=int, default=4)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokens = generate_tokens(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        steps=args.steps,
        device=device,
        max_voices=args.max_voices,
        steps_per_beat=args.steps_per_beat,
        temperature=args.temperature,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        out_midi = out_dir / f"sample_{i}.mid"
        save_midi(tokens[i], str(out_midi), steps_per_beat=args.steps_per_beat, tempo=args.tempo)
        torch.save(tokens[i].cpu(), out_dir / f"sample_{i}.pt")
        print(f"[generate] saved midi: {out_midi}")


if __name__ == "__main__":
    main()
