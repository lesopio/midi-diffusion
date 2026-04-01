import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from midigen.train import train_main


def extract_dataset(zip_path: Path, extract_dir: Path, force: bool) -> Path:
    midi_dir = extract_dir / "midi"
    if midi_dir.exists() and not force:
        return midi_dir
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))
    if not midi_dir.exists():
        raise RuntimeError(f"Zip does not contain midi/ folder: {zip_path}")
    return midi_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-zip", type=str, default="")
    parser.add_argument("--extract-dir", type=str, default="data/packed_dataset")
    parser.add_argument("--force-extract", action="store_true")
    args, rest = parser.parse_known_args()

    if args.data_zip:
        zip_path = Path(args.data_zip)
        extract_dir = Path(args.extract_dir)
        midi_dir = extract_dataset(zip_path, extract_dir, args.force_extract)
        if "--data-dir" not in rest:
            rest = rest + ["--data-dir", str(midi_dir)]

    sys.argv = [sys.argv[0]] + rest
    train_main()
