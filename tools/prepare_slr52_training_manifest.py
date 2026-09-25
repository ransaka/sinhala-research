#!/usr/bin/env python3
"""Join SLR52 metadata, an explicit speaker split, and extracted FLAC paths.

The output contains transcripts and stays under the Git-ignored data directory.
Test audio is never opened. Train/dev FLAC headers are checked without decoding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from build_split_manifest import read_metadata, transcript_key


FIELDS = [
    "utterance_id", "speaker_id", "source_split", "partition", "audio_path",
    "transcript", "duration_seconds", "sample_rate", "text_novel_eval_eligible",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True,
                        help="Extracted archive root containing train/ and test/ FLAC directories.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Write inside Git-ignored data/ or another private location.")
    args = parser.parse_args()

    metadata = {
        row["utterance_id"]: row
        for row in read_metadata([args.train_metadata, args.test_metadata], include_transcript_key=False)
    }
    split_rows = {}
    with args.split_manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"utterance_id", "speaker_id", "source_split", "partition"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Split manifest must include {sorted(required)}")
        for row in reader:
            key = row["utterance_id"].strip()
            if key in split_rows:
                raise ValueError(f"Duplicate split ID: {key}")
            split_rows[key] = row
    if metadata.keys() != split_rows.keys():
        missing = metadata.keys() - split_rows.keys()
        extra = split_rows.keys() - metadata.keys()
        raise ValueError(f"Metadata/split ID mismatch: {len(missing)} missing, {len(extra)} extra")

    root = args.audio_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    hours = defaultdict(float)
    speakers = defaultdict(set)
    speaker_partition = {}
    missing_audio = []
    transcript_empty = []
    rows = []
    for key in sorted(metadata):
        info = metadata[key]
        split = split_rows[key]
        if key != Path(key).name or not key or "/" in key or "\\" in key:
            raise ValueError(f"Unsafe utterance ID: {key!r}")
        if info["speaker_id"] != split["speaker_id"] or info["source_split"] != split["source_split"]:
            raise ValueError(f"Metadata/split mismatch for {key}")
        partition = split["partition"]
        if partition not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown partition {partition!r} for {key}")
        speaker = info["speaker_id"]
        earlier_partition = speaker_partition.setdefault(speaker, partition)
        if earlier_partition != partition:
            raise ValueError(f"Speaker {speaker} crosses {earlier_partition} and {partition}")
        source_split = info["source_split"]
        if source_split not in {"train", "test"}:
            raise ValueError(f"Unknown source split {source_split!r} for {key}")
        path = root / source_split / f"{key}.flac"
        text = transcript_key(info["reference"]) if partition != "test" else ""
        if partition != "test" and not text:
            transcript_empty.append(key)
        duration = ""
        sample_rate = ""
        if partition != "test":
            if not path.is_file():
                missing_audio.append(str(path))
            else:
                audio = sf.info(path)
                if audio.samplerate != 16_000 or audio.frames <= 0:
                    raise ValueError(f"Unexpected audio header for {path}: {audio}")
                duration = f"{audio.duration:.6f}"
                sample_rate = str(audio.samplerate)
                hours[partition] += audio.duration / 3600
        counts[partition] += 1
        speakers[partition].add(speaker)
        rows.append({
            "utterance_id": key,
            "speaker_id": speaker,
            "source_split": source_split,
            "partition": partition,
            "audio_path": str(path) if partition != "test" else "",
            "transcript": text if partition != "test" else "",
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "text_novel_eval_eligible": split.get("text_novel_eval_eligible", ""),
        })
    if transcript_empty or missing_audio:
        raise ValueError(f"Empty transcripts: {transcript_empty[:5]} ({len(transcript_empty)} total); "
                         f"missing train/dev audio: {missing_audio[:5]} ({len(missing_audio)} total)")
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_metadata_sha256": {p.name: sha256(p) for p in (args.train_metadata, args.test_metadata)},
        "split_manifest_sha256": sha256(args.split_manifest),
        "training_manifest_sha256": sha256(args.output),
        "audio_root": str(root),
        "rows": dict(counts),
        "speakers": {name: len(value) for name, value in speakers.items()},
        "train_dev_hours_from_headers": dict(hours),
        "train_dev_audio_headers_verified": True,
        "test_audio_opened": False,
        "test_reference_exported": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
