"""Shared transcript normalization and waveform/CTC batching."""

from __future__ import annotations

import unicodedata

import torch


SAMPLE_RATE = 16_000


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def padded_batch(rows: list[dict], feature_extractor, targets, device):
    features = feature_extractor(
        [row["waveform"] for row in rows],
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    labels = [targets.encode(row["text"]) if hasattr(targets, "encode") else
              [targets.get("|" if char.isspace() else char, targets["<unk>"]) for char in row["text"]]
              for row in rows]
    max_len = max(map(len, labels))
    label_tensor = torch.full((len(rows), max_len), -100, dtype=torch.long)
    for i, ids in enumerate(labels):
        label_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return (
        features.input_values.to(device),
        features.attention_mask.to(device) if "attention_mask" in features else None,
        label_tensor.to(device),
    )
