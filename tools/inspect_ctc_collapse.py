#!/usr/bin/env python3
"""Inspect greedy CTC predictions on real ASR1K utterances and save plots."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from train_ctc_pilot import decode, encode, load_rows, padded_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("datasets/asr1k-bd6c1be/data/train-00000-of-00001.parquet"))
    parser.add_argument("--encoder", type=Path, default=Path("checkpoints/xls-r-300m-1a640f3"))
    parser.add_argument("--run", type=Path, default=Path("outputs/ctc-overfit-4x200"))
    parser.add_argument("--dev-examples", type=int, default=4)
    args = parser.parse_args()
    destination = args.run / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)

    config = json.loads((args.run / "run_config.json").read_text())
    vocab = json.loads((args.run / "vocab.json").read_text())
    rows = load_rows(args.parquet)
    order = list(range(len(rows)))
    random.Random(config["seed"]).shuffle(order)
    train = [rows[i] for i in order[: config["split"]["train_pool"]]][: config["pilot_train_examples"]]
    dev_start = config["split"]["train_pool"]
    dev = [rows[i] for i in order[dev_start : dev_start + args.dev_examples]]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Wav2Vec2ForCTC.from_pretrained(args.run / "checkpoint").to(device).eval()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.encoder)
    inverse_vocab = {value: key for key, value in vocab.items()}
    predictions = []

    for partition, subset in (("train", train), ("dev", dev)):
        for index, row in enumerate(subset):
            inputs, mask, labels = padded_batch([row], feature_extractor, vocab, device)
            with torch.no_grad():
                output = model(input_values=inputs, attention_mask=mask, labels=labels)
            frame_count = int(model._get_feat_extract_output_lengths(mask.sum(-1))[0])
            logits = output.logits[0, :frame_count].float().cpu()
            probabilities = logits.softmax(-1).numpy()
            greedy_ids = probabilities.argmax(-1).tolist()
            blank = probabilities[:, 0]
            max_nonblank = probabilities[:, 1:].max(-1)
            nonblank_id = probabilities[:, 1:].argmax(-1) + 1
            target_ids = encode(row["text"], vocab)
            prediction = {
                "partition": partition,
                "utterance_id": row["id"],
                "reference": row["text"],
                "hypothesis": decode(greedy_ids, vocab),
                "seconds": row["seconds"],
                "target_codepoints": len(target_ids),
                "output_frames": frame_count,
                "loss_per_target_codepoint": float(output.loss),
                "blank_argmax_fraction": float(np.mean(np.array(greedy_ids) == 0)),
                "blank_probability_mean": float(blank.mean()),
                "blank_probability_p10_p50_p90": [float(x) for x in np.percentile(blank, [10, 50, 90])],
                "maximum_nonblank_probability_mean": float(max_nonblank.mean()),
                "top_nonblank_id": int(Counter(nonblank_id.tolist()).most_common(1)[0][0]),
            }
            prediction["top_nonblank_character"] = inverse_vocab[prediction["top_nonblank_id"]]
            predictions.append(prediction)

            times = np.arange(frame_count) * row["seconds"] / frame_count
            figure, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
            axes[0].plot(times, blank, label="CTC blank", linewidth=1)
            axes[0].plot(times, max_nonblank, label="largest nonblank character", linewidth=1)
            axes[0].set(ylabel="Frame probability", ylim=(0, 1), title=f"{partition} {index + 1}: {row['id']}")
            axes[0].legend(loc="upper right")
            axes[1].scatter(times, greedy_ids, s=3, label="greedy token ID")
            axes[1].set(xlabel="Audio time (seconds)", ylabel="Greedy token ID", ylim=(-1, len(vocab)))
            figure.savefig(destination / f"{partition}_{index + 1:02d}_posterior.png", dpi=160)
            plt.close(figure)

    with (destination / "predictions.jsonl").open("w") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    loss = config["train_loss_by_epoch"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    updates_per_epoch = config["steps"] / len(loss)
    ax.plot(np.arange(1, len(loss) + 1) * updates_per_epoch, loss, linewidth=1)
    ax.set(xlabel="Optimizer updates", ylabel="CTC loss (mean per target code point)",
           title="Four real training utterances: optimization loss")
    ax.grid(alpha=0.2)
    fig.savefig(destination / "train_loss.png", dpi=160)
    plt.close(fig)

    summary = {
        "checkpoint": str(args.run / "checkpoint"),
        "train_examples_inspected": len(train),
        "train_empty_hypotheses": sum(not item["hypothesis"] for item in predictions if item["partition"] == "train"),
        "dev_examples_inspected": len(dev),
        "dev_empty_hypotheses": sum(not item["hypothesis"] for item in predictions if item["partition"] == "dev"),
        "final_two_update_epoch_loss": loss[-1],
        "mean_train_blank_argmax_fraction": float(np.mean([item["blank_argmax_fraction"] for item in predictions if item["partition"] == "train"])),
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
