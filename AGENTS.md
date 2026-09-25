# Sinhala ASR research program

## Objective

Develop and evaluate a state-of-the-art Sinhala automatic speech recognition system and a reproducible benchmark for low-resource Sinhala speech. Research may include data quality and coverage, text representation and tokenization, acoustic encoders, language-model decoders, post-training, and inference-time correction.

## Research stance

Treat the claim that phonological-unit or phonetic representations outperform SentencePiece and other subword tokenizers as a hypothesis, not a premise. Design controlled experiments that can support or reject it. Keep terminology precise: `sinlib` calls its text units phonological units; do not conflate those text units with phonemes or acoustic features unless an experiment explicitly defines that mapping.

For every proposed improvement, state the mechanism, baseline, controlled variables, compute/data budget, primary metric, and failure conditions. Preserve strong conventional baselines and report negative results. Separate acoustic-model gains from decoder, language-model, normalization, and post-processing gains.

## `sinlib` use

The `sinlib` tokenizer is a first-class research candidate. Read its installed or linked documentation and inspect the actual version/API before integrating it. Its documented example segments `ආයුබෝවන්` as `['ආ', 'යු', 'බෝ', 'ව', 'න්']`. Verify coverage, unknown handling, normalization, reversibility, special tokens, sequence-length effects, and alignment with transcript conventions. Do not assume phonological text units preserve acoustic information by themselves; evaluate how target units interact with frame-level acoustic encoders and decoders.

## Experimental standards

- Keep train/dev/test speaker-disjoint where the corpus permits; document any exception.
- Record corpus provenance, license/consent constraints, hours, speakers, domains, dialects, recording conditions, preprocessing, and split manifests.
- Prevent transcript, speaker, and recording leakage. Do not place private or restricted data in source control.
- Compare systems under matched data, splits, decoding conditions, and feasible compute budgets. Report parameters, training steps, hardware, seeds, and decoding settings.
- Report WER and CER with explicit Sinhala text normalization; include raw and normalized scoring where useful. Add targeted analyses for diacritics/akshara errors, word boundaries, code-switching, names, dialect, and noise when data allows.
- Quantify uncertainty and use paired significance testing or bootstrap intervals for headline comparisons. Avoid calling a system SOTA without a current, sourced comparison on a comparable benchmark.
- Post-processing must be measured against unmodified hypotheses, with correction precision/coverage and regressions reported. Never silently rewrite references or use test references for tuning.

## Agent coordination

Use the specialist agents in `.codex/agents/` for bounded research and implementation tasks. Delegate only independent work; avoid concurrent edits to the same files. Ask agents to return concise findings with sources, assumptions, files changed, commands run, and unresolved risks. The main agent owns integration, experiment design, and final claims.
