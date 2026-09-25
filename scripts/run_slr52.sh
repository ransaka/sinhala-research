#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_DIR="$ROOT_DIR/data/slr52"
ENCODER_DIR="$ROOT_DIR/checkpoints/xls-r-300m-1a640f3"
AUDIO_ROOT=""
SPLIT_CSV=""
EXPLORATORY=0
TARGET=codepoint
SP_VOCAB_SIZE=512
GPUS=1
STEPS=4096
TRAIN_LIMIT=2048
DEV_LIMIT=512
BATCH_SIZE=2
EVAL_EVERY=512
CHECKPOINT_EVERY=1024
SEED=13
OUTPUT=""
RESUME=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_slr52.sh (--split SPLIT.csv | --exploratory-split) [options]

Downloads the pinned SLR52 mirror, verifies/extracts its archive, prepares a
private train/dev manifest, downloads pinned XLS-R, and trains one CTC target.
Defaults are a 2048-row, 4096-step staged run. Test references are not trained on.

Options:
  --split PATH              Approved speaker split CSV (required for final runs)
  --exploratory-split        Generate a provisional speaker-grouped row split
  --target NAME             codepoint|sentencepiece|sinlib (default codepoint)
  --sp-vocab-size N         SentencePiece unigram vocabulary size (default 512)
  --gpus N                  CUDA GPUs for torchrun; 1 uses plain Python
  --steps N                 Total optimizer updates (default 4096)
  --train-limit N           0 means all train rows (default 2048)
  --dev-limit N             0 means all dev rows (default 512)
  --batch-size N            Per-GPU batch size (default 2)
  --eval-every N            Update interval (default 512)
  --checkpoint-every N      Update interval (default 1024)
  --seed N                  Seed (default 13)
  --data-dir PATH           Download/extraction directory (default data/slr52)
  --audio-root PATH         Existing extracted root; skip archive download
  --encoder-dir PATH        Local pinned XLS-R directory
  --output PATH             Run directory (default outputs/slr52-TARGET-stage)
  --resume latest|PATH      Resume this output's latest or an explicit checkpoint
  --help                    Show this help

Example: bash scripts/run_slr52.sh --exploratory-split --gpus 2 --target codepoint
EOF
}

while (($#)); do
  case "$1" in
    --split|--target|--sp-vocab-size|--gpus|--steps|--train-limit|--dev-limit|--batch-size|--eval-every|--checkpoint-every|--seed|--data-dir|--audio-root|--encoder-dir|--output|--resume)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --split) SPLIT_CSV="$2";;
        --target) TARGET="$2";;
        --sp-vocab-size) SP_VOCAB_SIZE="$2";;
        --gpus) GPUS="$2";;
        --steps) STEPS="$2";;
        --train-limit) TRAIN_LIMIT="$2";;
        --dev-limit) DEV_LIMIT="$2";;
        --batch-size) BATCH_SIZE="$2";;
        --eval-every) EVAL_EVERY="$2";;
        --checkpoint-every) CHECKPOINT_EVERY="$2";;
        --seed) SEED="$2";;
        --data-dir) DATA_DIR="$2";;
        --audio-root) AUDIO_ROOT="$2";;
        --encoder-dir) ENCODER_DIR="$2";;
        --output) OUTPUT="$2";;
        --resume) RESUME="$2";;
      esac
      shift 2;;
    --exploratory-split) EXPLORATORY=1; shift;;
    --help|-h) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ "$TARGET" == codepoint || "$TARGET" == sentencepiece || "$TARGET" == sinlib ]] || { echo "Invalid target" >&2; exit 2; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be a positive integer" >&2; exit 2; }
for value in "$STEPS" "$BATCH_SIZE" "$EVAL_EVERY" "$CHECKPOINT_EVERY" "$SP_VOCAB_SIZE"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Steps, batch size and intervals must be positive integers" >&2; exit 2; }
done
for value in "$TRAIN_LIMIT" "$DEV_LIMIT" "$SEED"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo "Limits and seed must be nonnegative integers" >&2; exit 2; }
done
if [[ -z "$SPLIT_CSV" && "$EXPLORATORY" -ne 1 ]]; then
  echo "Provide --split PATH or --exploratory-split" >&2; exit 2
fi
if [[ -n "$SPLIT_CSV" && "$EXPLORATORY" -eq 1 ]]; then
  echo "Choose either --split or --exploratory-split" >&2; exit 2
fi
command -v uv >/dev/null || { echo "uv is required; see docs/RUNNING_EXPERIMENTS.md" >&2; exit 2; }
command -v hf >/dev/null || { echo "hf CLI is required; see docs/RUNNING_EXPERIMENTS.md" >&2; exit 2; }

mkdir -p "$DATA_DIR" "$(dirname "$ENCODER_DIR")"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"
if [[ -z "$OUTPUT" ]]; then OUTPUT="$ROOT_DIR/outputs/slr52-${TARGET}-stage"; fi
if [[ -n "$SPLIT_CSV" ]]; then
  [[ -f "$SPLIT_CSV" ]] || { echo "Split CSV does not exist: $SPLIT_CSV" >&2; exit 2; }
  SPLIT_CSV="$(realpath "$SPLIT_CSV")"
fi
if [[ -n "$AUDIO_ROOT" ]]; then AUDIO_ROOT="$(realpath "$AUDIO_ROOT")"; fi
if [[ -e "$OUTPUT" && -z "$RESUME" ]]; then
  [[ ! -d "$OUTPUT" || -z "$(ls -A "$OUTPUT")" ]] || { echo "Output is not empty; choose --output or --resume latest" >&2; exit 2; }
fi

uv sync --locked
.venv/bin/python - "$GPUS" <<'PY'
import sys, torch
wanted = int(sys.argv[1])
available = torch.cuda.device_count()
if wanted > 1 and available < wanted:
    raise SystemExit(f'Requested {wanted} CUDA GPUs, but only {available} are visible')
PY
hf download facebook/wav2vec2-xls-r-300m \
  --revision 1a640f32ac3e39899438a2931f9924c02f080a54 \
  --local-dir "$ENCODER_DIR"

SOURCE_DIR="$DATA_DIR/source"
if [[ -z "$AUDIO_ROOT" ]]; then
  ARCHIVE="$SOURCE_DIR/data/asrsinhala.tar.gz"
  AUDIO_ROOT="$DATA_DIR/audio"
  if [[ -f "$AUDIO_ROOT/.extraction_complete" && -d "$AUDIO_ROOT/train" && -d "$AUDIO_ROOT/test" ]] &&
     [[ "$(cat "$AUDIO_ROOT/.extraction_complete")" == 736e2c7320cc96b460a6b872e65f80dd48df844eb86b96097f6bca131c8617d0 ]]; then
    hf download Ransaka/SinhalaASR data/train_metadata.csv data/test_metadata.csv \
      --type dataset --revision bd1d968241e7edf8ce1577f569f59aac5c0f6b37 \
      --local-dir "$SOURCE_DIR"
  else
    hf download Ransaka/SinhalaASR \
      data/asrsinhala.tar.gz data/train_metadata.csv data/test_metadata.csv \
      --type dataset --revision bd1d968241e7edf8ce1577f569f59aac5c0f6b37 \
      --local-dir "$SOURCE_DIR"
    mkdir -p "$AUDIO_ROOT"
    .venv/bin/python - "$ARCHIVE" "$AUDIO_ROOT" <<'PY'
import hashlib, sys, tarfile
from pathlib import Path
archive, root = map(Path, sys.argv[1:])
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
expected = '736e2c7320cc96b460a6b872e65f80dd48df844eb86b96097f6bca131c8617d0'
if digest != expected:
    raise SystemExit(f'Archive SHA-256 mismatch: {digest}')
with tarfile.open(archive, 'r:gz') as tar:
    tar.extractall(root, filter='data')
if not (root / 'train').is_dir() or not (root / 'test').is_dir():
    raise SystemExit('Archive did not produce train/ and test/ under audio root')
(root / '.extraction_complete').write_text(digest + '\n')
PY
  fi
else
  hf download Ransaka/SinhalaASR \
    data/train_metadata.csv data/test_metadata.csv \
    --type dataset --revision bd1d968241e7edf8ce1577f569f59aac5c0f6b37 \
    --local-dir "$SOURCE_DIR"
  [[ -d "$AUDIO_ROOT/train" && -d "$AUDIO_ROOT/test" ]] || { echo "--audio-root must contain train/ and test/" >&2; exit 2; }
fi

TRAIN_CSV="$SOURCE_DIR/data/train_metadata.csv"
TEST_CSV="$SOURCE_DIR/data/test_metadata.csv"
if [[ "$EXPLORATORY" -eq 1 ]]; then
  .venv/bin/python tools/build_split_manifest.py \
    --train-metadata "$TRAIN_CSV" --test-metadata "$TEST_CSV" \
    --output-dir "$DATA_DIR/split" \
    --source-revision bd1d968241e7edf8ce1577f569f59aac5c0f6b37
  SPLIT_CSV="$DATA_DIR/split/speaker_grouped_manifest.csv"
fi

MANIFEST="$DATA_DIR/slr52_training.csv"
.venv/bin/python tools/prepare_slr52_training_manifest.py \
  --train-metadata "$TRAIN_CSV" --test-metadata "$TEST_CSV" \
  --split-manifest "$SPLIT_CSV" --audio-root "$AUDIO_ROOT" \
  --output "$MANIFEST"

if [[ "$RESUME" == latest ]]; then
  RESUME="$(.venv/bin/python - "$OUTPUT/latest_checkpoint.json" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['path'])
PY
)"
fi

TRAIN_ARGS=(--manifest "$MANIFEST" --encoder "$ENCODER_DIR" --target "$TARGET"
  --sp-vocab-size "$SP_VOCAB_SIZE"
  --output "$OUTPUT" --train-limit "$TRAIN_LIMIT" --dev-limit "$DEV_LIMIT"
  --batch-size "$BATCH_SIZE" --max-steps "$STEPS" --eval-every "$EVAL_EVERY"
  --checkpoint-every "$CHECKPOINT_EVERY" --regularization none --seed "$SEED")
if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi

if ((GPUS > 1)); then
  .venv/bin/torchrun --standalone --nnodes 1 --nproc-per-node "$GPUS" \
    tools/train_slr52_ctc.py "${TRAIN_ARGS[@]}"
else
  .venv/bin/python tools/train_slr52_ctc.py "${TRAIN_ARGS[@]}"
fi
