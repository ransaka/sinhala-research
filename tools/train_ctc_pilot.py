#!/usr/bin/env python3
"""One-condition local CTC pilot on the pinned SinhalaASR-1000 revision.

This is a plumbing/throughput experiment, not a benchmark result. The source has
no speaker IDs, so its deterministic row split cannot establish speaker
independence. It trains a fresh code-point CTC head over a pretrained XLS-R
encoder and leaves the held-out 10% partition untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as package_metadata
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from jiwer import cer, wer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from ctc_common import SAMPLE_RATE, normalize, padded_batch


DATASET_REVISION = "bd6c1be581c7831612730d3ae209caa3fe38bfa2"
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"


def load_rows(parquet_path: Path) -> list[dict]:
    table = pq.read_table(parquet_path, columns=["audio", "sentence"])
    rows = []
    for audio, sentence in zip(table.column("audio").to_pylist(), table.column("sentence").to_pylist()):
        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate}")
        text = normalize(sentence)
        if not text:
            continue
        rows.append({"id": audio["path"], "waveform": waveform, "text": text, "seconds": len(waveform) / sample_rate})
    if len(rows) != table.num_rows:
        raise ValueError(f"Only decoded {len(rows)} of {table.num_rows} rows")
    return rows


def build_vocab(rows: list[dict]) -> dict[str, int]:
    chars = {ch for row in rows for ch in row["text"] if not ch.isspace()}
    return {"<pad>": 0, "<unk>": 1, "|": 2, **{ch: i + 3 for i, ch in enumerate(sorted(chars))}}


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get("|" if ch.isspace() else ch, vocab["<unk>"]) for ch in text]


def ctc_feasibility(rows, vocab, model, device):
    waveform_lengths = torch.tensor([len(row["waveform"]) for row in rows], device=device)
    output_lengths = model._get_feat_extract_output_lengths(waveform_lengths).detach().cpu().tolist()
    target_lengths = [len(encode(row["text"], vocab)) for row in rows]
    required_lengths = []
    for row in rows:
        ids = encode(row["text"], vocab)
        adjacent_repeats = sum(left == right for left, right in zip(ids, ids[1:]))
        required_lengths.append(len(ids) + adjacent_repeats)
    infeasible = [need > available for need, available in zip(required_lengths, output_lengths)]
    return {
        "examples": len(rows),
        "infeasible_count": sum(infeasible),
        "target_codepoints_p50_p90_max": [float(np.percentile(target_lengths, 50)), float(np.percentile(target_lengths, 90)), max(target_lengths)],
        "output_frames_p50_p90_min": [float(np.percentile(output_lengths, 50)), float(np.percentile(output_lengths, 90)), min(output_lengths)],
        "min_ctc_frames_p50_p90_max": [float(np.percentile(required_lengths, 50)), float(np.percentile(required_lengths, 90)), max(required_lengths)],
    }


def decode(ids: list[int], vocab: dict[str, int]) -> str:
    inverse = {value: key for key, value in vocab.items()}
    out: list[str] = []
    previous = None
    for token_id in ids:
        if token_id != previous and token_id != 0:
            token = inverse.get(token_id, "<unk>")
            out.append(" " if token == "|" else ("�" if token == "<unk>" else token))
        previous = token_id
    return "".join(out)


@torch.no_grad()
def evaluate(model, rows, feature_extractor, vocab, device, batch_size):
    model.eval()
    references, hypotheses = [], []
    losses = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, attention_mask, labels = padded_batch(batch, feature_extractor, vocab, device)
        output = model(input_values=inputs, attention_mask=attention_mask)
        logits = output.logits
        input_lengths = model._get_feat_extract_output_lengths(
            attention_mask.sum(-1) if attention_mask is not None else torch.full(
                (len(batch),), inputs.shape[-1], device=device
            )
        )
        pred_ids = logits.argmax(-1).detach().cpu().tolist()
        input_lengths = input_lengths.detach().cpu().tolist()
        for row, ids, length in zip(batch, pred_ids, input_lengths):
            references.append(row["text"])
            hypotheses.append(decode(ids[:length], vocab))
    return {
        "examples": len(rows),
        "wer": float(wer(references, hypotheses)),
        "cer": float(cer(references, hypotheses)),
        "references": references,
        "hypotheses": hypotheses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("datasets/asr1k-bd6c1be/data/train-00000-of-00001.parquet"))
    parser.add_argument("--encoder", type=Path, default=Path("checkpoints/xls-r-300m-1a640f3"))
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Load CTC weights from a previous pilot checkpoint; optimizer starts fresh.")
    parser.add_argument("--output", type=Path, default=Path("outputs/asr1k-xlsr-codepoint-pilot"))
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overfit-examples", type=int, default=0,
                        help="Repeat this many training examples and report train-set decoding during optimization (0 disables).")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many optimizer updates (0 means complete requested epochs).")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="In overfit mode, decode the repeated training set every N updates.")
    parser.add_argument("--disable-regularization", action="store_true",
                        help="For tiny overfit diagnosis, disable SpecAugment, dropout, and layerdrop.")
    parser.add_argument("--disable-spec-augment", action="store_true",
                        help="For tiny overfit diagnosis, disable only SpecAugment.")
    parser.add_argument("--disable-dropout", action="store_true",
                        help="For tiny overfit diagnosis, disable dropout and layerdrop but retain SpecAugment.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    raw_hash = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    rows = load_rows(args.parquet)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    train_pool = [rows[i] for i in order[: int(len(order) * 0.8)]]
    dev = [rows[i] for i in order[int(len(order) * 0.8) : int(len(order) * 0.9)]]
    train = train_pool[: min(args.train_examples, len(train_pool))]
    if args.overfit_examples:
        train = train[: min(args.overfit_examples, len(train))]
        if len(train) < args.overfit_examples:
            raise ValueError("Requested more overfit examples than the train partition contains")
    vocab = build_vocab(train)
    args.output.mkdir(parents=True, exist_ok=True)
    split_rows = []
    train_ids = {row["id"] for row in train}
    train_pool_ids = {row["id"] for row in train_pool}
    dev_ids = {row["id"] for row in dev}
    for row in rows:
        split_rows.append(
            {
                "utterance_id": row["id"],
                "partition": "train" if row["id"] in train_ids else ("train_unused" if row["id"] in train_pool_ids else ("dev" if row["id"] in dev_ids else "test_sealed")),
                "duration_seconds": row["seconds"],
            }
        )
    (args.output / "split_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in split_rows)
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.encoder)
    regularization_overrides = (
        {
            "apply_spec_augment": False,
            "mask_time_prob": 0.0,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "feat_proj_dropout": 0.0,
            "final_dropout": 0.0,
            "activation_dropout": 0.0,
            "layerdrop": 0.0,
        }
        if args.disable_regularization else {}
    )
    if args.disable_spec_augment:
        regularization_overrides.update({"apply_spec_augment": False, "mask_time_prob": 0.0})
    if args.disable_dropout:
        regularization_overrides.update({
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "feat_proj_dropout": 0.0,
            "final_dropout": 0.0,
            "activation_dropout": 0.0,
            "layerdrop": 0.0,
        })
    model, loading_info = Wav2Vec2ForCTC.from_pretrained(
        args.resume_checkpoint or args.encoder,
        vocab_size=len(vocab),
        pad_token_id=0,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
        **regularization_overrides,
    )
    model.freeze_feature_encoder()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("lm_head.")
    ]
    head_parameters = [parameter for parameter in model.lm_head.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )

    train_seconds = sum(row["seconds"] for row in train)
    dev_seconds = sum(row["seconds"] for row in dev)
    duration_stats = np.asarray([row["seconds"] for row in rows])
    run = {
        "dataset_id": "Ransaka/SinhalaASR-1000",
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": raw_hash,
        "dataset_rows_decoded": len(rows),
        "dataset_speakers": "unavailable in source metadata; split is not speaker-disjoint",
        "model_id": "facebook/wav2vec2-xls-r-300m",
        "model_revision": MODEL_REVISION,
        "initial_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        "optimizer_resumed": False,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "model_vocab_size": len(vocab),
        "model_loading_info": {
            "missing_keys": sorted(loading_info.get("missing_keys", []))[:12],
            "unexpected_keys": sorted(loading_info.get("unexpected_keys", []))[:12],
            "mismatched_keys": [str(x) for x in sorted(loading_info.get("mismatched_keys", []))[:12]],
        },
        "architecture": "Wav2Vec2ForCTC; pretrained XLS-R encoder; new random CTC head; feature convolution frozen",
        "target_units": "NFC Sinhala Unicode code points; whitespace mapped to word delimiter",
        "split": {"algorithm": "seeded row shuffle", "seed": args.seed, "train_pool": len(train_pool), "dev": len(dev), "test": len(rows) - len(train_pool) - len(dev), "speaker_disjoint": False},
        "pilot_train_examples": len(train),
        "overfit_diagnostic": bool(args.overfit_examples),
        "max_steps": args.max_steps,
        "disable_regularization": args.disable_regularization,
        "disable_spec_augment": args.disable_spec_augment,
        "disable_dropout": args.disable_dropout,
        "pilot_dev_examples": len(dev),
        "pilot_train_audio_seconds": train_seconds,
        "pilot_dev_audio_seconds": dev_seconds,
        "duration_seconds_p50_p90_max": [float(np.percentile(duration_stats, 50)), float(np.percentile(duration_stats, 90)), float(duration_stats.max())],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "encoder_lr": args.encoder_lr,
        "new_ctc_head_lr": args.head_lr,
        "seed": args.seed,
        "device": str(device),
        "ctc_feasibility": {
            "train": ctc_feasibility(train, vocab, model, device),
            "dev": ctc_feasibility(dev, vocab, model, device),
        },
        "torch_version": torch.__version__,
        "torchaudio_version": package_metadata.version("torchaudio"),
        "transformers_version": package_metadata.version("transformers"),
        "datasets_version": package_metadata.version("datasets"),
        "sinlib_version": package_metadata.version("sinlib"),
    }
    (args.output / "run_config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    (args.output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n")

    initial = evaluate(model, dev, feature_extractor, vocab, device, args.batch_size)
    initial.pop("references")
    initial.pop("hypotheses")
    run["initial_dev_metrics"] = initial

    step_count = 0
    start_time = time.perf_counter()
    epoch_losses = []
    epoch_metrics = []
    overfit_snapshots = []
    epoch = 0
    while (args.max_steps and step_count < args.max_steps) or (not args.max_steps and epoch < args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train)
        loss_total = 0.0
        seen = 0
        for offset in range(0, len(train), args.batch_size):
            batch = train[offset : offset + args.batch_size]
            inputs, attention_mask, labels = padded_batch(batch, feature_extractor, vocab, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_values=inputs, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CTC loss at step {step_count}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
            step_count += 1
            if args.overfit_examples and step_count % args.eval_every == 0:
                snapshot = evaluate(model, train, feature_extractor, vocab, device, args.batch_size)
                record = {
                    "overfit_train": True,
                    "epoch": epoch + 1,
                    "step": step_count,
                    "loss": float(loss.detach().cpu()),
                    "train_wer": snapshot["wer"],
                    "train_cer": snapshot["cer"],
                    "sample_hypothesis": snapshot["hypotheses"][0],
                    "sample_reference": snapshot["references"][0],
                }
                overfit_snapshots.append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                model.train()
            if step_count % 20 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"epoch={epoch + 1} step={step_count} examples={seen}/{len(train)} loss={loss.item():.4f} elapsed_s={elapsed:.1f}", flush=True)
            if args.max_steps and step_count >= args.max_steps:
                break
        epoch_loss = loss_total / max(seen, 1)
        epoch_losses.append(epoch_loss)
        if not args.overfit_examples:
            metrics = evaluate(model, dev, feature_extractor, vocab, device, args.batch_size)
            metrics.pop("references")
            metrics.pop("hypotheses")
            metrics["epoch"] = epoch + 1
            metrics["train_loss"] = epoch_loss
            epoch_metrics.append(metrics)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
        epoch += 1

    elapsed = time.perf_counter() - start_time
    run.update(
        {
            "steps": step_count,
            "train_loss_by_epoch": epoch_losses,
            "epoch_metrics": epoch_metrics,
            "overfit_snapshots": overfit_snapshots,
            "wall_seconds": elapsed,
            "audio_seconds_per_wall_hour": train_seconds * args.epochs / elapsed * 3600,
            "seconds_per_update": elapsed / max(step_count, 1),
        }
    )
    final_metrics = evaluate(model, dev, feature_extractor, vocab, device, args.batch_size)
    if args.overfit_examples:
        final_train = evaluate(model, train, feature_extractor, vocab, device, args.batch_size)
        train_references = final_train.pop("references")
        train_hypotheses = final_train.pop("hypotheses")
        run["final_train_metrics"] = final_train
        (args.output / "train_hypotheses.jsonl").write_text(
            "".join(json.dumps({"reference": r, "hypothesis": h}, ensure_ascii=False) + "\n"
                    for r, h in zip(train_references, train_hypotheses))
        )
    refs, hyps = final_metrics.pop("references"), final_metrics.pop("hypotheses")
    run["dev_metrics"] = final_metrics
    (args.output / "hypotheses.jsonl").write_text(
        "".join(json.dumps({"reference": r, "hypothesis": h}, ensure_ascii=False) + "\n" for r, h in zip(refs, hyps))
    )
    (args.output / "run_config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    model.save_pretrained(args.output / "checkpoint")
    print(json.dumps({"completed": True, "wall_seconds": elapsed, "audio_seconds_per_wall_hour": run["audio_seconds_per_wall_hour"], "dev_wer": final_metrics["wer"], "dev_cer": final_metrics["cer"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
