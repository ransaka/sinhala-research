#!/usr/bin/env python3
"""Score cached multilingual Whisper large-v3 on the fixed ASR1K dev partition."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import torch
from jiwer import cer, wer
from transformers import AutoProcessor, WhisperForConditionalGeneration


DATASET_REVISION = "bd6c1be581c7831612730d3ae209caa3fe38bfa2"
MODEL_ID = "openai/whisper-large-v3"
MODEL_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"
SAMPLE_RATE = 16_000


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=Path("datasets/asr1k-bd6c1be/data/train-00000-of-00001.parquet"))
    parser.add_argument("--output", type=Path, default=Path("outputs/asr1k-whisper-large-v3-dev100"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(args.parquet, columns=["audio", "sentence"])
    raw_hash = hashlib.sha256(args.parquet.read_bytes()).hexdigest()
    order = list(range(table.num_rows))
    random.Random(13).shuffle(order)
    dev_indices = order[800:900][: args.limit]

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.float16,
    ).to(device).eval()
    started = time.perf_counter()
    results = []

    for n, idx in enumerate(dev_indices, start=1):
        audio = table.column("audio")[idx].as_py()
        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sample_rate}")
        reference = normalize(table.column("sentence")[idx].as_py())
        inputs = processor.feature_extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_features=inputs.input_features.to(device=device, dtype=torch.float16),
                language="si",
                task="transcribe",
                num_beams=1,
                max_length=args.max_length,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=1.05,
            )
        hypothesis = normalize(
            processor.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )
        results.append(
            {
                "utterance_id": audio["path"],
                "reference": reference,
                "hypothesis": hypothesis,
                "duration_seconds": len(waveform) / sample_rate,
            }
        )
        if n % 10 == 0:
            print(f"decoded={n}/{len(dev_indices)} elapsed_s={time.perf_counter() - started:.1f}", flush=True)

    refs = [row["reference"] for row in results]
    hyps = [row["hypothesis"] for row in results]
    run = {
        "dataset_id": "Ransaka/SinhalaASR-1000",
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": raw_hash,
        "split": "same seeded row-shuffle dev set as the CTC pilot; seed 13; rows 800-899; not speaker-disjoint",
        "examples": len(results),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": str(device),
        "dtype": "float16",
        "decoding": {
            "language": "si",
            "task": "transcribe",
            "num_beams": 1,
            "max_length": args.max_length,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": 1.05,
        },
        "normalization": "Unicode NFC and whitespace collapse",
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
        "empty_hypotheses": sum(not text for text in hyps),
        "wall_seconds": time.perf_counter() - started,
    }
    (args.output / "hypotheses.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results)
    )
    (args.output / "run_config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(run, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
