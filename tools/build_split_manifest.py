#!/usr/bin/env python3
"""Build a deterministic speaker-grouped split and text-novel eval masks.

Input is the pair of SLR52 metadata CSVs with columns 0=file ID, 1=speaker ID,
2=reference text, split. The script never copies audio or writes references.

The speaker assignment is provisional until duration balance, audio integrity,
corpus rights, and supervisor review are complete. Transcript novelty is exact
after NFC plus whitespace collapse; it is a benchmark mask, not text cleaning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

CSV_FIELD_LIMIT = 2**31 - 1
SPLIT_NAMES = ("train", "dev", "test")
TARGETS = (0.80, 0.10, 0.10)
SPLIT_SEED = "sinhala-asr-speaker-split-v1"


def transcript_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_metadata(paths: list[Path], *, include_transcript_key: bool = True) -> list[dict[str, str]]:
    csv.field_size_limit(CSV_FIELD_LIMIT)
    records: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"0", "1", "2", "split"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected metadata columns in {path}")
            for row in reader:
                utterance_id = row["0"].strip()
                speaker_id = row["1"].strip()
                if not utterance_id or not speaker_id:
                    raise ValueError(f"Missing utterance/speaker ID in {path}")
                if utterance_id in seen_ids:
                    raise ValueError(f"Duplicate utterance ID: {utterance_id}")
                seen_ids.add(utterance_id)
                records.append({
                    "utterance_id": utterance_id,
                    "speaker_id": speaker_id,
                    "reference": row["2"],
                    "source_split": row["split"].strip(),
                    "transcript_key": transcript_key(row["2"]) if include_transcript_key else "",
                })
    return records


def assign_speakers(records: list[dict[str, str]]) -> dict[str, str]:
    by_speaker: dict[str, int] = Counter(r["speaker_id"] for r in records)
    total = len(records)
    target_rows = [ratio * total for ratio in TARGETS]
    assigned_rows = [0, 0, 0]
    assignment: dict[str, str] = {}
    ordered = sorted(
        by_speaker.items(),
        key=lambda item: (
            -item[1],
            hashlib.sha256(f"{SPLIT_SEED}|{item[0]}".encode()).hexdigest(),
        ),
    )
    for speaker_id, row_count in ordered:
        # Greedily fill the split furthest below its target. Ties prefer train,
        # then dev, then test, which makes the procedure fully deterministic.
        split_index = max(
            range(3),
            key=lambda i: ((target_rows[i] - assigned_rows[i]) / target_rows[i], -i),
        )
        assignment[speaker_id] = SPLIT_NAMES[split_index]
        assigned_rows[split_index] += row_count
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Immutable HF dataset commit")
    args = parser.parse_args()

    records = read_metadata([args.train_metadata, args.test_metadata])
    speaker_split = assign_speakers(records)
    rows_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    text_sets: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    text_counts: dict[str, Counter[str]] = {name: Counter() for name in SPLIT_NAMES}
    for record in records:
        split = speaker_split[record["speaker_id"]]
        record["partition"] = split
        rows_by_split[split].append(record)
        text_sets[split].add(record["transcript_key"])
        text_counts[split][record["transcript_key"]] += 1

    # For dev/test, retain only transcript strings absent from every other
    # partition. This gives text-novel subsets for tuning and final evaluation.
    other_texts = {
        "dev": text_sets["train"] | text_sets["test"],
        "test": text_sets["train"] | text_sets["dev"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "speaker_grouped_manifest.csv"
    fields = [
        "utterance_id", "speaker_id", "source_split", "partition",
        "text_novel_eval_eligible",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda r: r["utterance_id"]):
            split = record["partition"]
            eligible = split == "train" or record["transcript_key"] not in other_texts[split]
            writer.writerow({
                "utterance_id": record["utterance_id"],
                "speaker_id": record["speaker_id"],
                "source_split": record["source_split"],
                "partition": split,
                "text_novel_eval_eligible": str(eligible).lower(),
            })

    counts = Counter(r["partition"] for r in records)
    speakers = {
        split: len({r["speaker_id"] for r in records if r["partition"] == split})
        for split in SPLIT_NAMES
    }
    eligible_rows = {
        split: sum(
            r["partition"] == split
            and (split == "train" or r["transcript_key"] not in other_texts[split])
            for r in records
        )
        for split in SPLIT_NAMES
    }
    train_seen_rows = {
        split: sum(
            count for key, count in text_counts[split].items()
            if key in text_sets["train"]
        )
        for split in ("dev", "test")
    }
    summary = {
        "status": "provisional_metadata_only",
        "source_repo": "Ransaka/SinhalaASR",
        "source_revision": args.source_revision,
        "source_metadata": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.train_metadata, args.test_metadata)
        },
        "source_rows": len(records),
        "speakers": speakers,
        "rows": dict(counts),
        "row_fractions": {s: counts[s] / len(records) for s in SPLIT_NAMES},
        "text_novel_eval_eligible_rows": eligible_rows,
        "text_novel_eval_eligible_fraction_of_partition": {
            s: eligible_rows[s] / counts[s] for s in SPLIT_NAMES
        },
        "unique_transcript_counts": {s: len(text_sets[s]) for s in SPLIT_NAMES},
        "shared_unique_transcripts": {
            "train_dev": len(text_sets["train"] & text_sets["dev"]),
            "train_test": len(text_sets["train"] & text_sets["test"]),
            "dev_test": len(text_sets["dev"] & text_sets["test"]),
        },
        "dev_test_rows_with_train_seen_transcript": train_seen_rows,
        "transcript_key": "Unicode NFC then collapse all whitespace runs",
        "split_seed": SPLIT_SEED,
        "assignment": "descending speaker row count; deterministic SHA-256 tie-break; greedy target deficit for 80/10/10",
        "duration_balanced": False,
        "audio_integrity_verified": False,
        "supervisor_approved": False,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    summary_path = args.output_dir / "speaker_grouped_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
