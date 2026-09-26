# Sinhala ASR research

This repository compares three CTC transcript targets for Sinhala ASR: Unicode code points, SentencePiece unigram, and `sinlib` phonological units. Whether `sinlib` improves recognition is an empirical question. The arms share one chosen dataset manifest, XLS-R encoder, audio loader, training loop, greedy decoder, and WER/CER calculation. CharBERT correction is a separate optional extension.

## Start here

- [Run the experiments](docs/RUNNING_EXPERIMENTS.md): complete setup, corpus layout, manifest preparation, staged commands for all targets, resume, outputs, and full-run procedure.
- [Experiment protocol](EXPERIMENT_PROTOCOL_v1.md): controlled comparison, scoring, test isolation, and reporting rules.
- [Benchmark and data record](docs/BENCHMARK.md): corpus provenance, split and audit status, and evidence to retain.
- [Phase 1](phase/PHASE1_STATUS.md) and [Phase 2](phase/PHASE2_STATUS.md): progress record.

The real-audio 256-row XLS-R/code-point pilot learned after 1,280 updates: train CER 21.5%, dev CER 35.8%; every dev hypothesis was nonempty. That row split lacks speaker IDs and is an engineering check, not a benchmark result. No full SLR52 run or SentencePiece/sinlib comparison has been executed yet.

`tools/train_ctc.py` selects the output target with `--target codepoint|sentencepiece|sinlib`. Source adapters produce a common manifest; the default source is the pinned `IAmNotAnanth/sinhala-ctc-111h` audio/text/duration dataset. SLR52 and an existing manifest remain selectable. The trainer builds a train-only target inventory/model and saves it with each run. Dataset files, transcript manifests, model weights, and outputs stay outside Git.

For an automated pinned download and run, use `scripts/run_ctc.sh`. It accepts `--gpus 2` for single-node CUDA training with `torchrun`. The new dataset has no speaker IDs, so its internal split cannot establish unseen-speaker performance; see the [runbook](docs/RUNNING_EXPERIMENTS.md) for exact commands and the evaluation boundary.

To run all three transcript targets at the same budget, use `scripts/run_tokenizer_comparison.sh`. It trains SentencePiece unigram on the selected training transcripts from scratch and builds the sinlib unit inventory from those same transcripts. Each arm writes its own training log, metrics, decoded predictions, tokenizer assets, and resumable checkpoint.
