#!/usr/bin/env python3
"""Prepare a private, exploratory manifest from the pinned local ASR1K Parquet file.

ASR1K does not identify speakers. This adapter is for pipeline checks only.
Rows sharing the same normalized transcript stay in one partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from ctc_common import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("datasets/asr1k-bd6c1be/data/train-00000-of-00001.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/asr1k_exploratory.csv"))
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    source = args.parquet.resolve()
    parquet = pq.ParquetFile(source)
    rows = []
    seen_ids = set()
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["path", "audio", "sentence"])
        for offset, row in enumerate(table.to_pylist()):
            item_id = row["path"]
            if item_id in seen_ids:
                raise ValueError(f"Duplicate utterance ID: {item_id}")
            seen_ids.add(item_id)
            text = normalize(row["sentence"])
            payload = row["audio"]["bytes"]
            if not text or not payload:
                raise ValueError(f"Missing transcript or audio: {item_id}")
            info = sf.info(io.BytesIO(payload))
            if info.samplerate != 16000 or info.frames <= 0:
                raise ValueError(f"Expected nonempty 16 kHz audio: {item_id}")
            # The shared trainer understands these direct Parquet audio references.
            location = "parquet:" + json.dumps({"path": str(source), "row_group": group, "row": offset})
            digest = hashlib.sha256(f"{args.seed}|{text}".encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % 10
            partition = "train" if bucket < 8 else "dev" if bucket == 8 else "test"
            rows.append({"utterance_id": item_id, "partition": partition,
                         "audio_path": "" if partition == "test" else location,
                         "transcript": "" if partition == "test" else text,
                         "duration_seconds": info.frames / info.samplerate,
                         "speaker_id": ""})
    if not {"train", "dev", "test"}.issubset({row["partition"] for row in rows}):
        raise ValueError("Source is too small for the exploratory split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"manifest": str(args.output), "counts": {
        part: sum(row["partition"] == part for row in rows)
        for part in ("train", "dev", "test")}, "speaker_status": "unknown",
        "purpose": "exploratory_pipeline_check"}))


if __name__ == "__main__":
    main()
