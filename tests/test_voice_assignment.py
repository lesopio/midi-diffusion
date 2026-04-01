import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from midigen.data import assign_voices


def test_assign_voices_keeps_continuity():
    prev = [60, 64, None, None]
    active = [60, 65]
    assigned = assign_voices(active, prev, max_voices=4)
    assert assigned.get(0) == 60
