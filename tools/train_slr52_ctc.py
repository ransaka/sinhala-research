#!/usr/bin/env python3
"""Stream real SLR52 FLAC files into a resumable XLS-R CTC run.

The prepared manifest must come from prepare_slr52_training_manifest.py. This
script reads only train/dev rows. A train limit is for recipe/throughput pilots;
leave it at zero for the complete training partition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as package_metadata
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from jiwer import cer, wer
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from ctc_common import SAMPLE_RATE, normalize, padded_batch
from ctc_targets import build_targets, load_targets


MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[dict], list[dict], dict]:
    train, dev = [], []
    all_counts = {"train": 0, "dev": 0, "test": 0}
    seen_ids = set()
    speaker_partition = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"utterance_id", "speaker_id", "partition", "audio_path", "transcript", "duration_seconds"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must include {sorted(required)}")
        for row in reader:
            item_id = row["utterance_id"]
            if item_id in seen_ids:
                raise ValueError(f"Duplicate utterance ID: {item_id}")
            seen_ids.add(item_id)
            part = row["partition"]
            if part not in all_counts:
                raise ValueError(f"Unexpected partition: {part}")
            all_counts[part] += 1
            speaker = row["speaker_id"]
            previous = speaker_partition.setdefault(speaker, part)
            if previous != part:
                raise ValueError(f"Speaker {speaker} crosses partitions")
            if part == "test":
                continue  # Test text and audio are never read.
            text = normalize(row["transcript"])
            duration = float(row["duration_seconds"])
            audio_path = Path(row["audio_path"])
            if not text or duration <= 0 or not audio_path.is_file():
                raise ValueError(f"Invalid train/dev row or missing audio: {item_id}")
            item = {"id": item_id, "speaker_id": speaker, "text": text,
                    "audio_path": audio_path, "seconds": duration}
            (train if part == "train" else dev).append(item)
    if not train or not dev:
        raise ValueError("Manifest must contain nonempty train and dev partitions")
    return train, dev, all_counts


def choose_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return rows
    ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}|{row['id']}".encode()).digest())
    return ordered[:limit]


def load_audio(row: dict) -> dict:
    waveform, rate = sf.read(row["audio_path"], dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if rate != SAMPLE_RATE or len(waveform) == 0:
        raise ValueError(f"Unexpected audio in {row['audio_path']}: {rate} Hz, {len(waveform)} samples")
    return {**row, "waveform": waveform}


def batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield [load_audio(row) for row in rows[start : start + batch_size]]


@torch.no_grad()
def evaluate(model, rows, feature_extractor, targets, device, batch_size):
    model.eval()
    references, hypotheses, predictions = [], [], []
    blank_frames = frame_count = 0
    for batch in batches(rows, batch_size):
        inputs, mask, _ = padded_batch(batch, feature_extractor, targets, device)
        logits = model(input_values=inputs, attention_mask=mask).logits
        lengths = model._get_feat_extract_output_lengths(mask.sum(-1)).cpu().tolist()
        ids = logits.argmax(-1).cpu().tolist()
        for row, sequence, length in zip(batch, ids, lengths):
            valid = sequence[:length]
            hypothesis = targets.decode(valid)
            references.append(row["text"])
            hypotheses.append(hypothesis)
            predictions.append({"utterance_id": row["id"], "reference": row["text"],
                                "hypothesis": hypothesis, "speaker_id": row["speaker_id"]})
            blank_frames += valid.count(0)
            frame_count += length
    metrics = {"examples": len(rows), "wer": float(wer(references, hypotheses)),
               "cer": float(cer(references, hypotheses)),
               "empty_hypotheses": sum(not item for item in hypotheses),
               "blank_argmax_fraction": blank_frames / frame_count}
    return metrics, predictions


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(model, optimizer, output: Path, step: int, state: dict, keep: int) -> None:
    folder = output / "checkpoints" / f"step-{step:08d}"
    temporary = output / "checkpoints" / f".step-{step:08d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "model")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "state": state,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }, temporary / "trainer_state.pt")
    if folder.exists():
        shutil.rmtree(folder)
    temporary.rename(folder)
    write_json(output / "latest_checkpoint.json", {"path": str(folder.resolve()), "step": step})
    old = sorted((output / "checkpoints").glob("step-*"))
    for path in old[:-keep]:
        shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--encoder", type=Path, default=Path("checkpoints/xls-r-300m-1a640f3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["codepoint", "sentencepiece", "sinlib"], default="codepoint")
    parser.add_argument("--sp-vocab-size", type=int, default=512,
                        help="SentencePiece unigram vocabulary size, including its unknown unit")
    parser.add_argument("--resume", type=Path, help="Path to this trainer's step-* checkpoint directory")
    parser.add_argument("--train-limit", type=int, default=0, help="0 uses every train row")
    parser.add_argument("--dev-limit", type=int, default=512, help="0 uses every dev row")
    parser.add_argument("--train-eval-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bucket-size", type=int, default=100,
                        help="Sort small shuffled pools by duration to reduce padding; 0 disables.")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--regularization", choices=["none", "pretrained"], default="none",
                        help="Model SpecAugment/dropout/layerdrop setting; AdamW weight decay stays 0.01.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if min(args.batch_size, args.max_steps, args.eval_every, args.checkpoint_every,
           args.keep_checkpoints, args.train_eval_examples, args.log_every) <= 0:
        raise ValueError("Batch size, steps, intervals, and retained checkpoints must be positive")
    if min(args.train_limit, args.dev_limit) < 0:
        raise ValueError("Limits must be nonnegative")
    if args.bucket_size < 0:
        raise ValueError("Bucket size must be nonnegative")
    if args.sp_vocab_size < 32:
        raise ValueError("SentencePiece vocabulary size must be at least 32")

    manifest_hash = file_hash(args.manifest)
    train_all, dev_all, counts = read_manifest(args.manifest)
    train = choose_rows(train_all, args.train_limit, args.seed)
    dev = choose_rows(dev_all, args.dev_limit, args.seed + 1)
    train_eval = choose_rows(train, args.train_eval_examples, args.seed + 2)
    if args.resume:
        if not args.output.is_dir():
            raise ValueError("Resume output directory does not exist")
        targets, target_meta = load_targets(args.output)
        if targets.kind != args.target:
            raise ValueError("Resume target type differs from the saved run")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"Output directory is not empty: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "checkpoints").mkdir(exist_ok=True)
        targets, target_meta = build_targets(args.target, train, args.output, args.sp_vocab_size)
    target_audit = {"train": targets.audit(train, "train", strict=True),
                    "dev": targets.audit(dev, "dev")}
    signature = {
        "manifest_sha256": manifest_hash, "target_metadata": target_meta,
        "target_type": args.target, "sp_vocab_size": args.sp_vocab_size,
        "train_limit": args.train_limit, "dev_limit": args.dev_limit,
        "train_eval_examples": args.train_eval_examples, "batch_size": args.batch_size,
        "bucket_size": args.bucket_size,
        "regularization": args.regularization, "seed": args.seed,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every,
        "keep_checkpoints": args.keep_checkpoints, "log_every": args.log_every,
    }
    if args.resume:
        old = json.loads((args.output / "run_config.json").read_text())
        if old["signature"] != signature:
            raise ValueError("Resume arguments or manifest/vocabulary differ from the saved run")
        if args.resume.parent.resolve() != (args.output / "checkpoints").resolve():
            raise ValueError("Resume checkpoint must belong to the specified output directory")
        latest = json.loads((args.output / "latest_checkpoint.json").read_text())
        if args.resume.resolve() != Path(latest["path"]).resolve():
            raise ValueError("Resume from the latest checkpoint to preserve metric and best-model history")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.encoder)
    overrides = {"apply_spec_augment": False, "mask_time_prob": 0.0,
                 "hidden_dropout": 0.0, "attention_dropout": 0.0,
                 "feat_proj_dropout": 0.0, "final_dropout": 0.0,
                 "activation_dropout": 0.0, "layerdrop": 0.0} if args.regularization == "none" else {}
    source = args.resume / "model" if args.resume else args.encoder
    model = Wav2Vec2ForCTC.from_pretrained(
        source, vocab_size=len(targets.vocab), pad_token_id=0, ctc_loss_reduction="mean",
        ctc_zero_infinity=True, ignore_mismatched_sizes=False if args.resume else True,
        **overrides,
    )
    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    encoder_parameters = [p for name, p in model.named_parameters()
                          if p.requires_grad and not name.startswith("lm_head.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": list(model.lm_head.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)

    run_config = {
        "dataset": "SLR52 extracted FLAC with explicit speaker split",
        "signature": signature, "source_manifest": str(args.manifest.resolve()),
        "encoder": str(args.encoder.resolve()), "encoder_revision": MODEL_REVISION,
        "target_units": args.target,
        "target_audit": target_audit,
        "normalization": "NFC and whitespace collapse, no reference correction",
        "train_rows": len(train), "dev_rows_scored": len(dev),
        "train_eval_rows": len(train_eval), "manifest_partitions": counts,
        "train_hours_from_headers": sum(row["seconds"] for row in train) / 3600,
        "dev_hours_from_headers": sum(row["seconds"] for row in dev) / 3600,
        "vocab_size": len(targets.vocab), "device": str(device),
        "initial_max_steps": args.max_steps,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "test_partition_opened": False,
        "torch_version": torch.__version__,
        "transformers_version": package_metadata.version("transformers"),
        "soundfile_version": package_metadata.version("soundfile"),
        "jiwer_version": package_metadata.version("jiwer"),
    }
    if not args.resume:
        write_json(args.output / "run_config.json", run_config)
        write_json(args.output / "selected_ids.json", {
            "train": [row["id"] for row in train], "dev": [row["id"] for row in dev],
            "train_eval": [row["id"] for row in train_eval],
        })
    state = {"step": 0, "epoch": 0, "batch_offset": 0,
             "best_dev_cer": float("inf"), "processed_audio_seconds": 0.0}
    if args.resume:
        saved = torch.load(args.resume / "trainer_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "mps" and saved["mps_rng"] is not None:
            torch.mps.set_rng_state(saved["mps_rng"])
        if args.max_steps <= state["step"]:
            raise ValueError("--max-steps must exceed the checkpoint's completed step")
        metrics_file = args.output / "metrics.jsonl"
        if metrics_file.exists():
            lines = metrics_file.read_text(encoding="utf-8").splitlines()
            if lines and json.loads(lines[-1])["step"] > state["step"]:
                raise ValueError("Metrics extend past the latest checkpoint; use a new output directory")

    with (args.output / "run_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "resume" if args.resume else "start",
                                 "checkpoint": str(args.resume) if args.resume else None,
                                 "target_max_steps": args.max_steps,
                                 "unix_time": time.time()}, ensure_ascii=False) + "\n")

    start = time.perf_counter()
    metrics_path = args.output / "metrics.jsonl"

    def run_evaluation() -> None:
        train_metrics, train_predictions = evaluate(model, train_eval, extractor, targets, device, args.batch_size)
        dev_metrics, dev_predictions = evaluate(model, dev, extractor, targets, device, args.batch_size)
        record = {"step": state["step"], "epoch": state["epoch"] + 1,
                  "train": train_metrics, "dev": dev_metrics,
                  "elapsed_seconds_this_process": time.perf_counter() - start,
                  "processed_audio_seconds_total": state["processed_audio_seconds"]}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        prediction_path = args.output / "eval_predictions" / f"step-{state['step']:08d}.jsonl"
        prediction_path.parent.mkdir(exist_ok=True)
        with prediction_path.open("w", encoding="utf-8") as stream:
            for partition, predictions in (("train", train_predictions), ("dev", dev_predictions)):
                for item in predictions:
                    stream.write(json.dumps({"partition": partition, **item}, ensure_ascii=False) + "\n")
        if dev_metrics["cer"] < state["best_dev_cer"]:
            state["best_dev_cer"] = dev_metrics["cer"]
            model.save_pretrained(args.output / "best_dev_model")
            write_json(args.output / "best_dev_metrics.json", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    while state["step"] < args.max_steps:
        order = list(range(len(train)))
        random.Random(args.seed + state["epoch"]).shuffle(order)
        if args.bucket_size:
            for offset in range(0, len(order), args.bucket_size):
                order[offset : offset + args.bucket_size] = sorted(
                    order[offset : offset + args.bucket_size], key=lambda index: train[index]["seconds"]
                )
        for batch_index in range(state["batch_offset"], (len(order) + args.batch_size - 1) // args.batch_size):
            indices = order[batch_index * args.batch_size : (batch_index + 1) * args.batch_size]
            batch = [load_audio(train[i]) for i in indices]
            inputs, mask, labels = padded_batch(batch, extractor, targets, device)
            output_lengths = model._get_feat_extract_output_lengths(mask.sum(-1)).cpu().tolist()
            for row, available in zip(batch, output_lengths):
                target_ids = targets.encode(row["text"])
                needed = len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:]))
                if needed > available:
                    raise ValueError(f"CTC-infeasible utterance {row['id']}: needs {needed}, has {available}")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_values=inputs, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {state['step'] + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            state["step"] += 1
            state["batch_offset"] = batch_index + 1
            state["processed_audio_seconds"] += sum(row["seconds"] for row in batch)
            if state["step"] % args.log_every == 0:
                print(json.dumps({"step": state["step"], "epoch": state["epoch"] + 1,
                                  "loss": float(loss.detach().cpu()),
                                  "elapsed_seconds_this_process": time.perf_counter() - start},
                                 ensure_ascii=False), flush=True)
            if state["step"] % args.eval_every == 0 or state["step"] == args.max_steps:
                run_evaluation()
            if (state["step"] % args.checkpoint_every == 0
                    or state["step"] % args.eval_every == 0
                    or state["step"] == args.max_steps):
                save_checkpoint(model, optimizer, args.output, state["step"], state, args.keep_checkpoints)
            if state["step"] >= args.max_steps:
                break
        if state["batch_offset"] >= (len(order) + args.batch_size - 1) // args.batch_size:
            state["epoch"] += 1
            state["batch_offset"] = 0
    print(json.dumps({"completed": True, "step": state["step"],
                      "best_dev_cer": state["best_dev_cer"],
                      "wall_seconds_this_process": time.perf_counter() - start},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
