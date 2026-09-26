#!/usr/bin/env python3
"""Stream manifest-listed audio into a resumable XLS-R CTC run.

The manifest is prepared by a source adapter. This script reads only train/dev
rows. A train limit is for recipe/throughput pilots; zero uses the full train set.
"""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from functools import lru_cache
import hashlib
import importlib.metadata as package_metadata
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import pyarrow.parquet as pq
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


def read_manifest(path: Path) -> tuple[list[dict], list[dict], dict, dict]:
    train, dev = [], []
    all_counts = {"train": 0, "dev": 0, "test": 0}
    seen_ids = set()
    speaker_partition = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"utterance_id", "partition", "audio_path", "transcript", "duration_seconds"}
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
            speaker = row.get("speaker_id", "").strip()
            if speaker:
                previous = speaker_partition.setdefault(speaker, part)
                if previous != part:
                    raise ValueError(f"Speaker {speaker} crosses partitions")
            if part == "test":
                continue  # Test text and audio are never read.
            text = normalize(row["transcript"])
            duration = float(row["duration_seconds"])
            audio_path = row["audio_path"]
            source_path = (Path(json.loads(audio_path[len("parquet:"):])["path"])
                           if audio_path.startswith("parquet:") else Path(audio_path))
            if not text or duration <= 0 or not source_path.is_file():
                raise ValueError(f"Invalid train/dev row or missing audio: {item_id}")
            item = {"id": item_id, "speaker_id": speaker, "text": text,
                    "audio_path": audio_path, "seconds": duration}
            (train if part == "train" else dev).append(item)
    if not train or not dev:
        raise ValueError("Manifest must contain nonempty train and dev partitions")
    known_speaker_rows = sum(bool(row["speaker_id"]) for row in train + dev)
    speaker_status = {"known_train_dev_rows": known_speaker_rows,
                      "train_dev_rows": len(train) + len(dev),
                      "split": "provided_ids_disjoint" if known_speaker_rows == len(train) + len(dev)
                      else "speaker_unknown" if known_speaker_rows == 0 else "speaker_partially_known"}
    return train, dev, all_counts, speaker_status


def choose_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return rows
    ordered = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}|{row['id']}".encode()).digest())
    return ordered[:limit]


@lru_cache(maxsize=8)
def parquet_file(path: str):
    return pq.ParquetFile(path)


@lru_cache(maxsize=2)
def parquet_audio_group(path: str, group: int):
    return parquet_file(path).read_row_group(group, columns=["audio"]).column(0)


def load_audio(row: dict) -> dict:
    location = row["audio_path"]
    if location.startswith("parquet:"):
        reference = json.loads(location[len("parquet:"):])
        cell = parquet_audio_group(reference["path"], reference["row_group"])[reference["row"]].as_py()
        payload = cell.get("bytes") if isinstance(cell, dict) else None
        if not payload:
            raise ValueError(f"Parquet audio bytes missing for {row['id']}")
        waveform, rate = sf.read(io.BytesIO(payload), dtype="float32")
    else:
        waveform, rate = sf.read(location, dtype="float32")
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


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def save_checkpoint(model, optimizer, output: Path, step: int, state: dict, keep: int,
                    rank: int, world_size: int, device: torch.device) -> None:
    rank_rng = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
    }
    all_rng = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(rank_rng, all_rng, dst=0)
    else:
        all_rng = [rank_rng]
    if rank != 0:
        dist.barrier()
        return
    folder = output / "checkpoints" / f"step-{step:08d}"
    temporary = output / "checkpoints" / f".step-{step:08d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary / "model")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "state": state,
        "rank_rng": all_rng,
    }, temporary / "trainer_state.pt")
    if folder.exists():
        shutil.rmtree(folder)
    temporary.rename(folder)
    write_json(output / "latest_checkpoint.json", {"path": str(folder.resolve()), "step": step})
    old = sorted((output / "checkpoints").glob("step-*"))
    for path in old[:-keep]:
        shutil.rmtree(path)
    if world_size > 1:
        dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-id", default="unspecified",
                        help="Source dataset identifier recorded in the run signature")
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
    parser.add_argument("--no-best-model", action="store_true",
                        help="Record best dev metric without storing an extra model copy")
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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise ValueError("Distributed training requires CUDA GPUs and torchrun")
        if args.regularization != "none":
            raise ValueError("Distributed training currently requires --regularization none")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=12))
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else
                          "cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    global_batch_size = args.batch_size * world_size

    manifest_hash = file_hash(args.manifest)
    train_all, dev_all, counts, speaker_status = read_manifest(args.manifest)
    train = choose_rows(train_all, args.train_limit, args.seed)
    dev = choose_rows(dev_all, args.dev_limit, args.seed + 1)
    train_eval = choose_rows(train, args.train_eval_examples, args.seed + 2)
    if args.resume:
        if not args.output.is_dir():
            raise ValueError("Resume output directory does not exist")
    else:
        if rank == 0:
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError(f"Output directory is not empty: {args.output}")
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "checkpoints").mkdir(exist_ok=True)
            build_targets(args.target, train, args.output, args.sp_vocab_size)
        if world_size > 1:
            dist.barrier()
    targets, target_meta = load_targets(args.output)
    if targets.kind != args.target:
        raise ValueError("Resume target type differs from the saved run")
    target_audit = {"train": targets.audit(train, "train", strict=True),
                    "dev": targets.audit(dev, "dev")}
    signature = {
        "manifest_sha256": manifest_hash, "dataset_id": args.dataset_id,
        "target_metadata": target_meta,
        "encoder_path": str(args.encoder.resolve()),
        "feature_extractor_sha256": file_hash(args.encoder / "preprocessor_config.json"),
        "target_type": args.target, "sp_vocab_size": args.sp_vocab_size,
        "world_size": world_size, "global_batch_size": global_batch_size,
        "train_limit": args.train_limit, "dev_limit": args.dev_limit,
        "train_eval_examples": args.train_eval_examples, "batch_size": args.batch_size,
        "bucket_size": args.bucket_size,
        "regularization": args.regularization, "seed": args.seed,
        "encoder_lr": args.encoder_lr, "head_lr": args.head_lr,
        "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every,
        "keep_checkpoints": args.keep_checkpoints, "log_every": args.log_every,
        "no_best_model": args.no_best_model,
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

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)
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
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    encoder_parameters = [p for name, p in model.named_parameters()
                          if p.requires_grad and not name.startswith("lm_head.")]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.encoder_lr},
        {"params": list(model.lm_head.parameters()), "lr": args.head_lr},
    ], weight_decay=0.01)
    raw_model = model
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    run_config = {
        "dataset": args.dataset_id,
        "speaker_status": speaker_status,
        "audio_storage": "mounted_parquet" if train[0]["audio_path"].startswith("parquet:") else "files",
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
        "world_size": world_size, "per_gpu_batch_size": args.batch_size,
        "global_batch_size": global_batch_size,
        "incomplete_global_batch_policy": "pad from shuffled order" if world_size > 1 else "keep",
        "repeated_rows_per_epoch": ((-len(train)) % global_batch_size) if world_size > 1 else 0,
        "initial_max_steps": args.max_steps,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "test_partition_opened": False,
        "torch_version": torch.__version__,
        "transformers_version": package_metadata.version("transformers"),
        "soundfile_version": package_metadata.version("soundfile"),
        "jiwer_version": package_metadata.version("jiwer"),
    }
    if not args.resume and rank == 0:
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
        rng = saved["rank_rng"][rank]
        random.setstate(rng["python_rng"])
        np.random.set_state(rng["numpy_rng"])
        torch.set_rng_state(rng["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda_rng"], device)
        if device.type == "mps":
            torch.mps.set_rng_state(rng["mps_rng"])
        if args.max_steps <= state["step"]:
            raise ValueError("--max-steps must exceed the checkpoint's completed step")
        for log_name in ("metrics.jsonl", "train_log.jsonl"):
            log_file = args.output / log_name
            if log_file.exists():
                lines = log_file.read_text(encoding="utf-8").splitlines()
                if lines and json.loads(lines[-1])["step"] > state["step"]:
                    raise ValueError(f"{log_name} extends past the latest checkpoint; use a new output directory")

    if rank == 0:
        append_jsonl(args.output / "run_events.jsonl", {
            "event": "resume" if args.resume else "start",
            "checkpoint": str(args.resume) if args.resume else None,
            "target_max_steps": args.max_steps,
            "unix_time": time.time(),
        })

    start = time.perf_counter()
    metrics_path = args.output / "metrics.jsonl"

    def run_evaluation() -> None:
        if rank == 0:
            train_metrics, train_predictions = evaluate(raw_model, train_eval, extractor, targets, device, args.batch_size)
            dev_metrics, dev_predictions = evaluate(raw_model, dev, extractor, targets, device, args.batch_size)
            record = {"step": state["step"], "epoch": state["epoch"] + 1,
                      "train": train_metrics, "dev": dev_metrics,
                      "elapsed_seconds_this_process": time.perf_counter() - start,
                      "processed_audio_seconds_total": state["processed_audio_seconds"]}
            append_jsonl(metrics_path, record)
            prediction_path = args.output / "eval_predictions" / f"step-{state['step']:08d}.jsonl"
            prediction_path.parent.mkdir(exist_ok=True)
            with prediction_path.open("w", encoding="utf-8") as stream:
                for partition, predictions in (("train", train_predictions), ("dev", dev_predictions)):
                    for item in predictions:
                        stream.write(json.dumps({"partition": partition, **item}, ensure_ascii=False) + "\n")
            if dev_metrics["cer"] < state["best_dev_cer"]:
                state["best_dev_cer"] = dev_metrics["cer"]
                if not args.no_best_model:
                    raw_model.save_pretrained(args.output / "best_dev_model")
                write_json(args.output / "best_dev_metrics.json", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        if world_size > 1:
            best = [state["best_dev_cer"] if rank == 0 else None]
            dist.broadcast_object_list(best, src=0)
            state["best_dev_cer"] = best[0]

    while state["step"] < args.max_steps:
        order = list(range(len(train)))
        random.Random(args.seed + state["epoch"]).shuffle(order)
        if args.bucket_size:
            for offset in range(0, len(order), args.bucket_size):
                order[offset : offset + args.bucket_size] = sorted(
                    order[offset : offset + args.bucket_size], key=lambda index: train[index]["seconds"]
                )
        epoch_batches = (len(order) + global_batch_size - 1) // global_batch_size
        for batch_index in range(state["batch_offset"], epoch_batches):
            global_indices = order[batch_index * global_batch_size : (batch_index + 1) * global_batch_size]
            if world_size > 1 and len(global_indices) < global_batch_size:
                missing = global_batch_size - len(global_indices)
                global_indices += [order[(state["epoch"] + i) % len(order)] for i in range(missing)]
            indices = global_indices[rank * args.batch_size : (rank + 1) * args.batch_size]
            batch = [load_audio(train[i]) for i in indices]
            inputs, mask, labels = padded_batch(batch, extractor, targets, device)
            output_lengths = raw_model._get_feat_extract_output_lengths(mask.sum(-1)).cpu().tolist()
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
            state["processed_audio_seconds"] += sum(train[i]["seconds"] for i in global_indices)
            if state["step"] % args.log_every == 0 or state["step"] == args.max_steps:
                log_loss = loss.detach().clone()
                if world_size > 1:
                    dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
                    log_loss /= world_size
                if rank == 0:
                    log_record = {
                        "step": state["step"], "epoch": state["epoch"] + 1,
                        "loss": float(log_loss.cpu()),
                        "elapsed_seconds_this_process": time.perf_counter() - start,
                        "processed_audio_seconds_total": state["processed_audio_seconds"],
                    }
                    append_jsonl(args.output / "train_log.jsonl", log_record)
                    print(json.dumps(log_record, ensure_ascii=False), flush=True)
            if state["step"] % args.eval_every == 0 or state["step"] == args.max_steps:
                run_evaluation()
            if state["step"] % args.checkpoint_every == 0 or state["step"] == args.max_steps:
                save_checkpoint(raw_model, optimizer, args.output, state["step"], state,
                                args.keep_checkpoints, rank, world_size, device)
            if state["step"] >= args.max_steps:
                break
        if state["batch_offset"] >= epoch_batches:
            state["epoch"] += 1
            state["batch_offset"] = 0
    if rank == 0:
        completed = {"completed": True, "step": state["step"],
                     "best_dev_cer": state["best_dev_cer"],
                     "wall_seconds_this_process": time.perf_counter() - start}
        append_jsonl(args.output / "run_events.jsonl", {
            "event": "complete", "step": state["step"],
            "best_dev_cer": state["best_dev_cer"], "unix_time": time.time(),
        })
        print(json.dumps(completed, ensure_ascii=False), flush=True)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
