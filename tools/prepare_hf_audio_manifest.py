#!/usr/bin/env python3
"""Materialize a pinned audio/text/duration Hub dataset into the common CTC manifest.

This creates an exploratory split with exact audio and normalized transcript
groups kept together. It cannot certify speaker independence without IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

from datasets import load_dataset
import pyarrow.parquet as pq
import soundfile as sf

from ctc_common import SAMPLE_RATE, normalize


FIELDS = ["utterance_id", "partition", "audio_path", "transcript", "duration_seconds",
          "speaker_id", "source_audio_path"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--parquet-root", type=Path,
                        help="Optional mounted/downloaded bucket root containing data/*.parquet")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-percent", type=int, default=80)
    parser.add_argument("--dev-percent", type=int, default=10)
    args = parser.parse_args()
    if not (0 < args.train_percent < 100 and 0 < args.dev_percent < 100 and
            args.train_percent + args.dev_percent < 100):
        raise ValueError("Train/dev percentages must leave a nonempty test share")
    config = {"repo_id": args.repo_id, "revision": args.revision, "seed": args.seed,
              "train_percent": args.train_percent, "dev_percent": args.dev_percent,
              "audio_dir": str(args.audio_dir.resolve()),
              "parquet_root": str(args.parquet_root.resolve()) if args.parquet_root else None}
    summary_path = args.output.with_suffix(".summary.json")
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        sealed = args.output.with_name(args.output.stem + "_sealed_test.csv")
        if (saved.get("config") != config or not args.output.is_file() or
                file_hash(args.output) != saved.get("manifest_sha256") or
                not sealed.is_file() or file_hash(sealed) != saved.get("sealed_test_sha256")):
            raise ValueError("Existing manifest has different source/config or changed contents; use a new output path")
        with args.output.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row["partition"] != "test" and not Path(row["audio_path"]).is_file():
                    raise FileNotFoundError(f"Materialized audio missing: {row['audio_path']}")
        with sealed.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if not Path(row["audio_path"]).is_file():
                    raise FileNotFoundError(f"Sealed-test audio missing: {row['audio_path']}")
        print(json.dumps({"reused_manifest": str(args.output), "summary": saved}, ensure_ascii=False))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    parquet_inventory = None
    if args.parquet_root:
        parquet_paths = sorted((args.parquet_root / "data").glob("*.parquet"))
        if not parquet_paths:
            parquet_paths = sorted(args.parquet_root.glob("*.parquet"))
        if not parquet_paths:
            raise ValueError(f"No Parquet shards under {args.parquet_root}/data or {args.parquet_root}")
        if (args.repo_id == "IAmNotAnanth/sinhala-ctc-111h" and
                args.revision == "a59ffd444b39ff3c9fca387f2d4879201511f93b"):
            expected = {f"train-{index:05d}-of-00026.parquet" for index in range(26)}
            if {path.name for path in parquet_paths} != expected:
                raise ValueError("Mounted bucket does not contain exactly the 26 pinned dataset shards")
            if sum(path.stat().st_size for path in parquet_paths) != 12_724_390_010:
                raise ValueError("Mounted bucket shard sizes differ from the pinned dataset inventory")
        parquet_inventory = [{"name": path.name, "bytes": path.stat().st_size}
                             for path in parquet_paths]

        def local_rows():
            for path in parquet_paths:
                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=64, columns=["audio", "text", "duration"]):
                    yield from batch.to_pylist()

        source = local_rows()
    else:
        source = load_dataset(args.repo_id, split="train", revision=args.revision,
                              streaming=True).decode(False)
    rows = []
    parent = []
    audio_first: dict[str, int] = {}
    text_first: dict[str, int] = {}
    seen_ids: set[str] = set()
    excluded = Counter()
    duration_mismatches = 0
    channels = Counter()

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = root(left), root(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for source_index, row in enumerate(source):
        text = normalize(row["text"] or "")
        audio = row["audio"]
        payload = audio.get("bytes") if isinstance(audio, dict) else None
        if not text or not payload:
            excluded["empty_text_or_audio"] += 1
            continue
        info = sf.info(io.BytesIO(payload))
        if info.samplerate != SAMPLE_RATE or info.frames <= 0:
            raise ValueError(f"Unexpected sample rate/empty audio at source row {source_index}: {info}")
        channels[info.channels] += 1
        actual_seconds = info.frames / info.samplerate
        claimed_seconds = float(row["duration"])
        if abs(actual_seconds - claimed_seconds) > max(0.05, 0.01 * actual_seconds):
            duration_mismatches += 1
        audio_hash = hashlib.sha256(payload).hexdigest()
        source_path = str(audio.get("path") or "")
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(source_path).stem)[:80] or "audio"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        utterance_id = f"{stem}-{audio_hash[:12]}-{text_hash[:8]}"
        if utterance_id in seen_ids:
            excluded["duplicate_audio_text"] += 1
            continue
        seen_ids.add(utterance_id)
        suffix = {"WAV": "wav", "FLAC": "flac", "OGG": "ogg"}.get(info.format, "bin")
        audio_path = args.audio_dir / audio_hash[:2] / f"{audio_hash}.{suffix}"
        if not audio_path.exists():
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = audio_path.with_suffix(".tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, audio_path)
        index = len(rows)
        parent.append(index)
        if audio_hash in audio_first:
            union(index, audio_first[audio_hash])
        else:
            audio_first[audio_hash] = index
        if text in text_first:
            union(index, text_first[text])
        else:
            text_first[text] = index
        rows.append({"utterance_id": utterance_id, "audio_path": str(audio_path.resolve()),
                     "transcript": text, "duration_seconds": f"{actual_seconds:.6f}",
                     "speaker_id": "", "source_audio_path": source_path})
        if (source_index + 1) % 5000 == 0:
            print(json.dumps({"source_rows_seen": source_index + 1, "retained": len(rows)},
                             ensure_ascii=False), flush=True)
    if not rows:
        raise ValueError("No usable audio/text rows were read")

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[root(index)].append(index)
    partitions = {}
    for members in components.values():
        key = min(rows[index]["utterance_id"] for index in members)
        bucket = int.from_bytes(hashlib.sha256(f"{args.seed}|{key}".encode()).digest()[:8], "big") % 100
        partition = ("train" if bucket < args.train_percent else
                     "dev" if bucket < args.train_percent + args.dev_percent else "test")
        for index in members:
            partitions[index] = partition
    counts = Counter(partitions.values())
    if not counts["train"] or not counts["dev"] or not counts["test"]:
        raise ValueError(f"Split has an empty partition: {counts}")

    temporary_manifest = args.output.with_suffix(".tmp")
    sealed_test = args.output.with_name(args.output.stem + "_sealed_test.csv")
    temporary_test = sealed_test.with_suffix(".tmp")
    hours = defaultdict(float)
    with temporary_manifest.open("w", encoding="utf-8", newline="") as stream, \
         temporary_test.open("w", encoding="utf-8", newline="") as test_stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        test_writer = csv.DictWriter(test_stream, fieldnames=FIELDS)
        writer.writeheader()
        test_writer.writeheader()
        for index, row in enumerate(rows):
            partition = partitions[index]
            hours[partition] += float(row["duration_seconds"]) / 3600
            full = {"partition": partition, **row}
            if partition == "test":
                test_writer.writerow(full)
                full = {**full, "audio_path": "", "transcript": "",
                        "duration_seconds": "", "source_audio_path": ""}
            writer.writerow(full)
    os.replace(temporary_test, sealed_test)
    os.replace(temporary_manifest, args.output)
    summary = {"config": config, "manifest_sha256": file_hash(args.output),
               "sealed_test_sha256": file_hash(sealed_test),
               "rows": dict(counts), "hours": dict(hours), "components": len(components),
               "unique_audio_hashes": len(audio_first), "unique_normalized_texts": len(text_first),
               "excluded": dict(excluded), "duration_mismatches": duration_mismatches,
               "channel_counts": dict(channels), "sample_rate": SAMPLE_RATE,
               "speaker_ids_available": False, "speaker_overlap": "unknown",
               "local_parquet_content_hashes_verified": False if args.parquet_root else None,
               "hub_revision_verified_for_input": False if args.parquet_root else True,
               "local_parquet_inventory": parquet_inventory,
               "exact_audio_or_normalized_text_cross_partition": False,
               "sealed_test_path": str(sealed_test.resolve())}
    temporary_summary = summary_path.with_suffix(".tmp")
    temporary_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    os.replace(temporary_summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
