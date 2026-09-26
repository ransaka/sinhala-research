#!/usr/bin/env python3
"""Compare two greedy-CTC prediction files on exactly the same utterances.

The current trainer stores NFC/whitespace-normalized references. Consequently
"emitted" scores below use those stored references, not original corpus text.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    """The trainer's nfc_ws_v1 scoring profile; no punctuation or case changes."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edits(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    """Return substitution, deletion, insertion counts at minimum edit cost."""
    previous = [(0, 0, index) for index in range(len(hypothesis) + 1)]
    for ref_index, ref_token in enumerate(reference, 1):
        current = [(0, ref_index, 0)]
        for hyp_index, hyp_token in enumerate(hypothesis, 1):
            match = previous[hyp_index - 1]
            if ref_token == hyp_token:
                current.append(match)
                continue
            substitution = (match[0] + 1, match[1], match[2])
            deletion = (previous[hyp_index][0], previous[hyp_index][1] + 1,
                        previous[hyp_index][2])
            insertion = (current[-1][0], current[-1][1], current[-1][2] + 1)
            current.append(min((substitution, deletion, insertion), key=lambda value: sum(value)))
        previous = current
    return previous[-1]


def read_predictions(path: Path, partition: str) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("partition") != partition:
                continue
            for field in ("utterance_id", "reference", "hypothesis", "speaker_id"):
                if not isinstance(row.get(field), str):
                    raise ValueError(f"{path}:{line_number}: missing string {field}")
            identifier = row["utterance_id"]
            if identifier in rows:
                raise ValueError(f"{path}: duplicate {partition} ID {identifier}")
            rows[identifier] = row
    if not rows:
        raise ValueError(f"{path}: no {partition} predictions")
    return rows


def counts(row: dict, *, normalized: bool) -> dict[str, tuple[int, int, int, int]]:
    reference, hypothesis = row["reference"], row["hypothesis"]
    if normalized:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    words_ref, words_hyp = reference.split(), hypothesis.split()
    return {
        "wer": (*edits(words_ref, words_hyp), len(words_ref)),
        "cer": (*edits(list(reference), list(hypothesis)), len(reference)),
    }


def rate(rows: list[dict[str, tuple[int, int, int, int]]], metric: str) -> dict:
    sums = [sum(row[metric][column] for row in rows) for column in range(4)]
    return rate_from_sums(sums, metric)


def rate_from_sums(sums: list[int], metric: str) -> dict:
    if sums[3] == 0:
        raise ValueError(f"Zero reference {metric} denominator")
    return {"rate": sum(sums[:3]) / sums[3], "substitutions": sums[0],
            "deletions": sums[1], "insertions": sums[2], "reference_units": sums[3]}


def percentile(values: list[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (index - low)


def compare(first: dict[str, dict], second: dict[str, dict], replicates: int, seed: int) -> dict:
    if first.keys() != second.keys():
        missing_first = sorted(second.keys() - first.keys())[:10]
        missing_second = sorted(first.keys() - second.keys())[:10]
        raise ValueError(f"Different utterance IDs; only second: {missing_first}; only first: {missing_second}")
    for identifier in first:
        for field in ("reference", "speaker_id"):
            if first[identifier][field] != second[identifier][field]:
                raise ValueError(f"Mismatched {field} for {identifier}")
    ids = sorted(first)
    speakers = [first[identifier]["speaker_id"] for identifier in ids]
    if all(speakers):
        cluster_kind = "speaker"
        groups = defaultdict(list)
        for identifier, speaker in zip(ids, speakers):
            groups[speaker].append(identifier)
        clusters = list(groups.values())
    else:
        cluster_kind = "utterance" if not any(speakers) else "utterance_partial_speaker_ids"
        clusters = [[identifier] for identifier in ids]
    output = {"utterances": len(ids), "clusters": len(clusters),
              "bootstrap_cluster": cluster_kind, "normalization": "nfc_ws_v1",
              "bootstrap_replicates": replicates, "bootstrap_seed": seed, "scores": {}}
    rng = random.Random(seed)
    for profile in ("emitted", "normalized"):
        normalized = profile == "normalized"
        left = {identifier: counts(first[identifier], normalized=normalized) for identifier in ids}
        right = {identifier: counts(second[identifier], normalized=normalized) for identifier in ids}
        output["scores"][profile] = {}
        for metric in ("wer", "cer"):
            left_score = rate(list(left.values()), metric)
            right_score = rate(list(right.values()), metric)
            cluster_totals = [
                ([sum(left[identifier][metric][column] for identifier in cluster) for column in range(4)],
                 [sum(right[identifier][metric][column] for identifier in cluster) for column in range(4)])
                for cluster in clusters
            ]
            deltas = []
            for _ in range(replicates):
                sampled = [rng.choice(cluster_totals) for _ in clusters]
                left_sums = [sum(pair[0][column] for pair in sampled) for column in range(4)]
                right_sums = [sum(pair[1][column] for pair in sampled) for column in range(4)]
                try:
                    delta = rate_from_sums(right_sums, metric)["rate"] - rate_from_sums(
                        left_sums, metric)["rate"]
                except ValueError:
                    continue  # A bootstrap sample can contain no reference words.
                deltas.append(delta)
            if not deltas:
                raise ValueError(f"No valid bootstrap samples for {profile} {metric}")
            deltas.sort()
            output["scores"][profile][metric] = {
                "first": left_score, "second": right_score,
                "delta_second_minus_first": right_score["rate"] - left_score["rate"],
                "delta_95pct_percentile_interval": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
                "valid_bootstrap_replicates": len(deltas),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="First run's eval_predictions/step-*.jsonl")
    parser.add_argument("second", type=Path, help="Second run's matching prediction file")
    parser.add_argument("--partition", choices=("train", "dev", "test"), default="dev")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, help="Write the JSON result to this path")
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be positive")
    result = compare(read_predictions(args.first, args.partition),
                     read_predictions(args.second, args.partition),
                     args.bootstrap_replicates, args.seed)
    result.update({"first_path": str(args.first), "second_path": str(args.second),
                   "partition": args.partition,
                   "reference_note": "References are as stored by the trainer after NFC and whitespace normalization; emitted is not corpus-raw."})
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
