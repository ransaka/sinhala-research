# Run the SLR52 CTC experiments

Run these commands from a clone of this repository. They use real SLR52 FLAC files and one trainer for code points, SentencePiece unigram, and `sinlib` phonological units. The only arm-specific training argument is `--target` (plus `--sp-vocab-size` for SentencePiece).

## One-command download and run

After installing `uv` and `hf` as below, use `hf auth login` only if your dataset access requires it. From the repo root, run a two-GPU staged code-point experiment using the frozen split:

```bash
bash scripts/run_slr52.sh --split /absolute/path/to/approved_speaker_split.csv \
  --gpus 2 --target codepoint --steps 2048 \
  --output outputs/slr52-codepoint-stage-2gpu
```

For an exploratory run without your frozen split, replace `--split ...` with `--exploratory-split`. The script downloads revision `bd1d968241e7edf8ce1577f569f59aac5c0f6b37` of `Ransaka/SinhalaASR`, verifies the archive SHA-256, extracts `train/` and `test/`, downloads pinned XLS-R, prepares the private manifest, and launches `torchrun` for `--gpus 2`. The archive is about 12.9 GB compressed; extracted audio and checkpoints need additional space. Use `--audio-root /absolute/path` to reuse an existing extracted corpus and skip the archive download. Run `bash scripts/run_slr52.sh --help` for all arguments.

To run SentencePiece or `sinlib`, change `--target` and `--output`. To resume, repeat the original arguments with `--resume latest` and a larger `--steps` total. Use `--train-limit 0 --dev-limit 0` for full partitions, with a profiled and frozen step budget. For two GPUs, `--batch-size 2` means **2 utterances per GPU and 4 globally per optimizer update**. On 2,048 train rows, 2,048 two-GPU updates are four passes, matching the audio exposure of 4,096 one-GPU updates at batch size 2. An incomplete last global batch repeats up to three rows from the deterministic shuffled order so every source row is seen. Keep GPU count, batch size, and encoder directory fixed when resuming. The trainer supports `--regularization none` in distributed mode.

## 1. Environment and inputs

If needed, install `uv` and the Hugging Face `hf` CLI, then reopen the shell so both commands are on `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash -s
```

Set up the locked environment and pinned encoder:

```bash
cd /path/to/sinhala-research
uv sync --locked
hf download facebook/wav2vec2-xls-r-300m --revision 1a640f32ac3e39899438a2931f9924c02f080a54 --local-dir checkpoints/xls-r-300m-1a640f3
```

Use the pinned `Ransaka/SinhalaASR` metadata and your approved, frozen speaker split. If the metadata CSVs are missing, use the authenticated `hf` CLI:

```bash
hf download Ransaka/SinhalaASR --type dataset --revision bd1d968241e7edf8ce1577f569f59aac5c0f6b37 --include '*metadata.csv' --local-dir data/source_metadata
```

The extracted corpus root must contain `train/<utterance_id>.flac` and `test/<utterance_id>.flac`. Metadata files are `train_metadata.csv` and `test_metadata.csv`, with columns `0` (ID), `1` (speaker), `2` (transcript), `split`. The frozen split CSV needs `utterance_id,speaker_id,source_split,partition`, with partition `train`, `dev`, or `test`. Set these paths once:

```bash
export SLR_AUDIO_ROOT=/absolute/path/to/extracted/asrsinhala
export SLR_SPLIT_CSV=/absolute/path/to/approved_speaker_split.csv
export SLR_TRAIN_CSV=data/source_metadata/data/train_metadata.csv
export SLR_TEST_CSV=data/source_metadata/data/test_metadata.csv
```

If there is no approved split for an **exploratory** run, generate a deterministic speaker-grouped metadata split and set `SLR_SPLIT_CSV=data/manifests/slr52_v1/speaker_grouped_manifest.csv`. This generated split is row-balanced only, not an audited final benchmark split.

```bash
.venv/bin/python tools/build_split_manifest.py \
  --train-metadata "$SLR_TRAIN_CSV" --test-metadata "$SLR_TEST_CSV" \
  --output-dir data/manifests/slr52_v1 \
  --source-revision bd1d968241e7edf8ce1577f569f59aac5c0f6b37
```

## 2. Prepare the private training manifest

```bash
.venv/bin/python tools/prepare_slr52_training_manifest.py \
  --train-metadata "$SLR_TRAIN_CSV" --test-metadata "$SLR_TEST_CSV" \
  --split-manifest "$SLR_SPLIT_CSV" --audio-root "$SLR_AUDIO_ROOT" \
  --output data/manifests/slr52_training.csv
```

This checks every train/dev audio header for 16 kHz and writes `data/manifests/slr52_training.csv` plus a summary JSON. Fix source data or the split if it fails; do not remove rows from only one arm. Test paths and references are omitted.

## 3. Staged learning and throughput runs

Run one command at a time. Each selects the same deterministic 2,048 train utterances, 512 dev utterances, 128 train evaluation utterances, seed, batch size, optimizer settings, and 4,096 updates. Each starts from the same XLS-R encoder; only output targets and head dimensions differ.

```bash
.venv/bin/python tools/train_slr52_ctc.py \
  --manifest data/manifests/slr52_training.csv --encoder checkpoints/xls-r-300m-1a640f3 \
  --target codepoint --output outputs/slr52-stage-codepoint \
  --train-limit 2048 --dev-limit 512 --train-eval-examples 128 \
  --batch-size 2 --max-steps 4096 --eval-every 512 \
  --checkpoint-every 1024 --keep-checkpoints 2 --regularization none --seed 13

.venv/bin/python tools/train_slr52_ctc.py \
  --manifest data/manifests/slr52_training.csv --encoder checkpoints/xls-r-300m-1a640f3 \
  --target sentencepiece --sp-vocab-size 512 --output outputs/slr52-stage-sentencepiece \
  --train-limit 2048 --dev-limit 512 --train-eval-examples 128 \
  --batch-size 2 --max-steps 4096 --eval-every 512 \
  --checkpoint-every 1024 --keep-checkpoints 2 --regularization none --seed 13

.venv/bin/python tools/train_slr52_ctc.py \
  --manifest data/manifests/slr52_training.csv --encoder checkpoints/xls-r-300m-1a640f3 \
  --target sinlib --output outputs/slr52-stage-sinlib \
  --train-limit 2048 --dev-limit 512 --train-eval-examples 128 \
  --batch-size 2 --max-steps 4096 --eval-every 512 \
  --checkpoint-every 1024 --keep-checkpoints 2 --regularization none --seed 13
```

`none` disables SpecAugment, dropout, and layerdrop while retaining AdamW weight decay 0.01, matching the successful small-data pilot. The target vocabulary and SentencePiece model use selected **training references only**. `sinlib` 0.3.2 supplies segmentation; sorted train-only units receive deterministic CTC IDs. Train round-trip and unknown-unit checks must pass before optimization; dev unknowns and reconstruction mismatches are saved in `run_config.json`.

Inspect `metrics.jsonl` for train/dev WER, CER, empty hypotheses, blank-argmax fraction, and processed audio seconds. `eval_predictions/` has reference/hypothesis rows; `best_dev_model/` stores the best dev-CER weights. `target/` has the vocabulary, metadata and, for SentencePiece, `spm.model`. `checkpoints/` stores optimizer and RNG state. These subset scores are exploratory.

## 4. Resume a stopped run

Read the run's `latest_checkpoint.json` and use its `path` value as `--resume`. All original arguments must match; `--max-steps` is the **new total** updates. Example when the latest checkpoint is step 4096:

```bash
.venv/bin/python tools/train_slr52_ctc.py \
  --manifest data/manifests/slr52_training.csv --encoder checkpoints/xls-r-300m-1a640f3 \
  --target codepoint --output outputs/slr52-stage-codepoint \
  --resume outputs/slr52-stage-codepoint/checkpoints/step-00004096 \
  --train-limit 2048 --dev-limit 512 --train-eval-examples 128 \
  --batch-size 2 --max-steps 8192 --eval-every 512 \
  --checkpoint-every 1024 --keep-checkpoints 2 --regularization none --seed 13
```

For other arms, change target, output/resume paths, and retain `--sp-vocab-size 512` for SentencePiece. A full-data run needs a **new** output directory so its target inventory uses all training transcripts. Do not resume a subset model onto a different manifest or vocabulary.

## 5. Full-corpus comparison

After the staged profile, freeze one update budget, dev set, seed list, regularization setting, and scoring script in the experiment record. Set `FULL_STEPS` below to that common **total optimizer-update count**. This example uses the complete train and dev partitions for all three arms with seed 13. Use a new output directory per arm/seed; repeat the loop with each predeclared seed. The only target-specific option is SentencePiece's fixed 512-unit request.

```bash
FULL_STEPS=100000  # replace with the profiled, frozen budget before starting
for TARGET in codepoint sentencepiece sinlib; do
  if [ "$TARGET" = sentencepiece ]; then SP_OPTION='--sp-vocab-size=512'; else SP_OPTION=''; fi
  .venv/bin/python tools/train_slr52_ctc.py \
    --manifest data/manifests/slr52_training.csv \
    --encoder checkpoints/xls-r-300m-1a640f3 \
    --target "$TARGET" $SP_OPTION \
    --output "outputs/slr52-full-${TARGET}-seed13" \
    --train-limit 0 --dev-limit 0 --train-eval-examples 128 \
    --batch-size 2 --max-steps "$FULL_STEPS" --eval-every 5000 \
    --checkpoint-every 5000 --keep-checkpoints 2 \
    --regularization none --seed 13
done
```

Start each arm from the same encoder, not a stage checkpoint. Keep test sealed until the system matrix and scorer are frozen. The [protocol](../EXPERIMENT_PROTOCOL_v1.md) defines primary fixed-exposure comparisons and separately labelled dev-selected checkpoints.
