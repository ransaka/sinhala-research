#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="$ROOT_DIR/outputs/tokenizer-comparison"
SP_VOCAB_SIZE=512
RESUME_ALL=0
FORWARD=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_tokenizer_comparison.sh [--output-root PATH] [--sp-vocab-size N] [run_ctc.sh options]

Runs codepoint, SentencePiece unigram, and sinlib sequentially with the same
source, split, seed, training budget, and decoder. Each target has its own run
directory, metrics, predictions, training log, and checkpoints. SentencePiece
is learned from the selected training transcripts in its run directory.

For a local exploratory check:
  .venv/bin/python tools/prepare_asr1k_manifest.py
  bash scripts/run_tokenizer_comparison.sh --dataset manifest \
    --manifest data/manifests/asr1k_exploratory.csv --steps 100 \
    --train-limit 256 --dev-limit 32 --output-root outputs/asr1k-targets

Pass --resume latest to resume all three arms at a larger --steps total.
Other options are forwarded to scripts/run_ctc.sh. Do not pass --target or
--output; this script sets both per arm.
EOF
}

while (($#)); do
  case "$1" in
    --output-root|--sp-vocab-size)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --output-root ]]; then OUTPUT_ROOT="$2"; else SP_VOCAB_SIZE="$2"; fi
      shift 2;;
    --resume)
      (($# >= 2)) || { echo "Missing value for --resume" >&2; exit 2; }
      [[ "$2" == latest ]] || { echo "Only --resume latest is supported for a three-arm run" >&2; exit 2; }
      RESUME_ALL=1
      shift 2;;
    --target|--output)
      echo "$1 is managed by this script" >&2; exit 2;;
    --help|-h) usage; exit 0;;
    *) FORWARD+=("$1"); shift;;
  esac
done

[[ "$SP_VOCAB_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid --sp-vocab-size" >&2; exit 2; }
for TARGET in codepoint sentencepiece sinlib; do
  echo "Starting $TARGET run in $OUTPUT_ROOT/$TARGET" >&2
  RESUME_ARGS=()
  if ((RESUME_ALL)) && [[ -f "$OUTPUT_ROOT/$TARGET/latest_checkpoint.json" ]]; then
    RESUME_ARGS=(--resume latest)
  fi
  bash "$ROOT_DIR/scripts/run_ctc.sh" \
    "${FORWARD[@]}" "${RESUME_ARGS[@]}" --target "$TARGET" --sp-vocab-size "$SP_VOCAB_SIZE" \
    --output "$OUTPUT_ROOT/$TARGET"
done
