#!/usr/bin/env python3
"""Load a saved CTC checkpoint's model, audio extractor, and text targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from ctc_targets import load_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="A checkpoints/step-* directory")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    targets, metadata = load_targets(checkpoint)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint / "feature_extractor")
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint / "model")
    if model.config.vocab_size != len(targets.vocab):
        raise ValueError("Model head and saved CTC target vocabulary have different sizes")
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "target": targets.kind,
        "vocab_size": len(targets.vocab),
        "sampling_rate": extractor.sampling_rate,
        "target_metadata": metadata,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
