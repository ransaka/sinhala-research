# Phase 2 status

**Current result:** XLS-R 300M with a Unicode code-point CTC head learned from 256 real training utterances. After 1,280 updates on Apple MPS, train WER/CER was 72.3%/21.5%; the 100-row dev split reached 93.2%/35.8%. All hypotheses were nonempty. This row split has no speaker IDs, so the result is a training-path pilot, not a benchmark score.

**Methods run:** the original code-point CTC settings collapsed to blank output; tiny real-audio ablations examined SpecAugment, dropout, and layerdrop; the successful 256-row condition disabled those three while retaining AdamW weight decay. A cached Whisper large-v3 inference check was exploratory. SentencePiece CTC, sinlib CTC, full OpenSLR training, and CharBERT correction have not been run.

**Code ready:** `tools/prepare_slr52_training_manifest.py` joins the source metadata, speaker split, and extracted FLAC paths. `tools/train_slr52_ctc.py` trains the Unicode code-point baseline with lazy audio loading, train/dev decoding, and resumable checkpoints. It does not train SentencePiece or sinlib targets. The full archive and final audited split are outside this workspace; no full-corpus run has started.

**Next:** profile a 2,048-row OpenSLR training subset and fixed speaker-held-out dev subset; use its actual learning and throughput to set the full code-point run budget. Then implement SentencePiece and sinlib target adapters and compare all three under matched audio, encoder initialization, updates, seeds, decoding, and scoring. Keep the test partition sealed until final evaluation.
