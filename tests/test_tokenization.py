import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from midigen.data import (
    pianoroll_to_polyphonic_tokens,
    PITCH_RANGE,
    PITCH_OFFSET,
    REST_TOKEN,
)


def test_tokenization_basic():
    pr = np.zeros((4, PITCH_RANGE), dtype=np.uint8)
    pr[0, 0] = 1
    pr[1, 2] = 1
    pr[3, 5] = 1

    tokens = pianoroll_to_polyphonic_tokens(pr, max_voices=2, max_len=4)
    assert tokens.shape == (4, 2)
    assert tokens[0, 0] == PITCH_OFFSET
    assert tokens[1, 0] == PITCH_OFFSET + 2
    assert tokens[2, 0] == REST_TOKEN
