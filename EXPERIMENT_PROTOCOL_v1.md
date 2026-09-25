# Sinhala ASR experiment protocol v1

**Date:** 24 September 2026
**Purpose:** Freeze the primary experimental question and comparison rules before full training.
**Status:** Research-ready draft; architecture and corpus gates below are explicit and remain unresolved only where evidence is missing.

## Scope lock

Primary intervention: transcript output-unit representation for CTC. Conditions are Unicode code points, train-only SentencePiece unigram, and sinlib phonological units. One encoder, one data manifest, one normalization/scoring profile, one decoder, matched audio exposure and stopping rules. CharBERT, encoder continuation, external LM, punctuation restoration, and broad NLU are excluded from the primary experiment.

## Pre-run gates

1. Confirm the corpus license, attribution, lineage, and permitted processing/release.
2. Complete audio decode, duration and sample-rate audit; inspect references and duplicates.
3. Freeze the selected data manifest. Use a speaker-grouped split where speaker IDs exist, reporting duration balance and transcript overlap. For a source without speaker IDs, label its internal split as speaker-overlap unknown and reserve an independent speaker-identified evaluation set for unseen-speaker claims.
4. Pin model and software revisions; establish the candidate encoder's license and known training-data overlap.
5. Pin sinlib version and characterize segmentation, normalization, Unicode coverage, unknown handling, reversibility, special tokens, sequence lengths, and CTC alignment feasibility.
6. Train a code-point pipeline smoke/overfit run and profile 5–8 GPU hours before fixing the final run count.
7. Freeze scorer, normalization profile, seed list, train/development selection rule, and test access protocol.

No failed gate is silently waived; its impact becomes an explicit limitation or triggers a corpus/system change before final protocol freeze.

## Matched conditions

| Component | Fixed rule |
|---|---|
| Audio | Same immutable utterance manifest and waveform preprocessing for all arms; preserve original audio; resample explicitly to 16 kHz only when verified necessary |
| Encoder | One pinned multilingual pretrained waveform encoder; same initial checkpoint in every arm; same frozen/fine-tuned policy |
| Acoustic architecture | Same CTC architecture, initialization rule, optimizer, schedule and regularization; only output target map/head dimensions vary as required |
| Text | Same immutable references and normalizer; no reference-specific spelling correction |
| Code points | Unicode scalar/code-point targets after the shared normalizer; special CTC blank configured consistently |
| SentencePiece | Unigram model trained on train references only; one fixed vocabulary-size choice selected on train/dev; store model file and revision/hash |
| sinlib | Pinned sinlib implementation; retain documented “phonological unit” terminology; save exact token map; never assume output is a phoneme sequence |
| Budget | Primary checkpoint at a predeclared fixed exposure: same utterance order, audio-seconds seen, and optimizer-update count across arms; no primary early stopping. Any dev-selected checkpoint is a separately labeled operational analysis |
| Decode | Greedy CTC only for the primary comparison; identical collapse and blank handling; no language model or post-correction |
| Seeds | Target 3 predeclared seeds per arm; if cost gate fails, run 1 equal-budget seed per arm and 2 additional seeds for sinlib and the strongest conventional arm; third arm remains exploratory |

If an arm cannot fit or align due to its output sequence length, record examples affected and failure rate. Do not filter difficult utterances for only one arm. If a shared utterance is impossible in one condition, predeclare a common evaluable subset and also report full-coverage results.

The `IAmNotAnanth/sinhala-ctc-111h` source contains audio, text, and duration but no speaker IDs. Its prepared internal split groups exact audio duplicates and identical normalized transcripts; this prevents those exact forms of leakage but cannot certify speaker independence or independence from other OpenSLR-derived corpora. Treat internal dev scores as exploratory until a separate, lineage-checked external test is available.

## Normalization and metrics

Freeze a versioned script. Proposed primary text transform: Unicode NFC; collapse consecutive whitespace; casefold Latin-script spans; convert a fixed list of punctuation to whitespace; preserve Sinhala marks, virama, ZWJ/ZWNJ, digits, and code-switch text. No NFKC, spelling repair, or reference changes. Retain and publish raw references separately when rights permit.

Primary: corpus-micro normalized WER = total word edit distance / total reference words. Word tokenization is whitespace split after the frozen transform. Also report substitution/deletion/insertion counts and denominator. Secondary: normalized code-point CER with spaces retained, raw WER/CER, utterance exact match, no-space CER (diagnostic only), and optional sinlib-unit distance (diagnostic only). Clearly distinguish code-point CER from akshara/phonological-unit scores.

Uncertainty: paired bootstrap by speaker (all their utterances resampled as a cluster), 10,000 replicates, 95% percentile intervals for absolute WER/CER differences; preserve system pairing within each sampled speaker. Report seed-wise results and pooled summaries without hiding seed variability. No significance claim from unpaired utterance counts.

## Error and efficiency records

Per run: system ID, source revision, manifest hash, text-map hash, seed, total/trainable parameters, steps, audio-seconds, batch/audio policy, hardware, wall/GPU hours, peak memory, software/library versions, checkpoint rule, decoding settings, failure counts, and logs. Per tokenizer: corpus coverage, unit inventory, target-length percentiles, output-head size, special/unknown behavior, exact round-trip/reconstruction, CTC minimum-frame feasibility and infeasible examples. Error slices: diacritics/akshara, spacing, code-switch, names/numerals, noise/domain/dialect only when labels and support are adequate.

## Test isolation

Development is the only split used to tune the tokenizer vocabulary size, normalizer, checkpoint, training schedule, or analysis decisions. Sinlib coverage/reversibility diagnostics before training use train/development only; after final freeze, test unknown/round-trip rates are descriptive and cannot trigger repair. Keep sealed test audio/references outside training and cloud workspaces until the system matrix and scorer are frozen. Run each planned final system once on the sealed test. Store raw predictions, normalized predictions, scoring output, and manifest/config hashes together. Any post-result change is a new experiment and requires a new untouched test or is labeled exploratory.

## CharBERT extension protocol

Separate experiment after primary results: train-side or out-of-fold ASR hypotheses paired with their corresponding train references; never train on development/test references or in-sample predictions from the same fitted recognizer unless explicitly controlled. Freeze one selected ASR's outputs. Compare identity, a simple dictionary/rule baseline, and a trained CharBERT. Select on development outputs. Test once. Report corpus WER/CER, edit precision and coverage using a validated sequence-alignment metric, false edits to correct source words, counts of utterances improved/unchanged/worsened, and category regressions. If there is no checkpoint, valid assets, or safe pair generation within the cap, do not run this extension.

## Predeclared run-budget fallback

First do a 5–8 GPU-hour code-point profile. Estimate full-run cost from observed rate. Prefer 3 arms × 3 seeds at a matched budget if affordable. If not, use the seven-run plan above; if that still exceeds the cap, reduce training duration equally across arms and retain the seed count for the primary pair. Do not spend on CharBERT before primary runs and one confirmatory rerun are complete. Stop cloud instances when idle; record actual spend.

## Implementation status, 25 September 2026

The local XLS-R 300M/code-point CTC pipeline passed a real-data learning pilot: 256 train rows, 100 dev rows, 1,280 updates, 21.52% train CER and 35.82% dev CER with regularization disabled. The earlier regularized recipe produced all-blank dev output under the same small-data update count. These are engineering results from a row split without speaker IDs, not primary comparison scores. The [runbook](docs/RUNNING_EXPERIMENTS.md) now covers a shared, resumable SLR52 trainer with code-point, SentencePiece unigram, and sinlib target adapters. No SLR52 tokenizer comparison or full-corpus run has started. The regularization choice, complete-corpus training budget, scorer freeze, and final test evaluation are not yet established.

## Deviations

Record date, reason, affected conditions, and whether made before test access. Any deviation after viewing test results is flagged prominently and cannot replace the original confirmatory score.
