# Data and benchmark record

## Current audio/text/duration training source

The default runner now uses [`IAmNotAnanth/sinhala-ctc-111h`](https://huggingface.co/datasets/IAmNotAnanth/sinhala-ctc-111h) at revision `a59ffd444b39ff3c9fca387f2d4879201511f93b`: one train split, 178,364 rows, 26 Parquet shards, with `audio`, `text`, and `duration`. The embedded audio is reported as mono 16 kHz and the card describes about 111 hours of cleaned OpenSLR-derived speech. No speaker ID, source utterance ID, or official dev/test split is published. The source paths such as `si_0000001.wav` do not map directly to the original OpenSLR FileIDs; transcript-only joining cannot recover speakers reliably.

`tools/prepare_hf_audio_manifest.py` streams the pinned rows, verifies decoded audio headers, records duration discrepancies, and creates a deterministic 80/10/10 exploratory train/dev/test allocation. It stores only the requested train/dev audio as lossless FLAC where source PCM permits, or records mounted Parquet row references in direct mode. Exact duplicate audio and identical NFC/whitespace-normalized transcripts stay in one partition; test references go to a separate sealed CSV without training audio paths. **Speaker overlap remains unknown.** This split supports pipeline work and controlled target comparisons on the same audio, but cannot establish generalization to new voices. A speaker-identified independent test set is needed for that claim.

The Hub card's license metadata says CC BY 4.0 while the [original OpenSLR 52 page](https://openslr.org/52/) lists Attribution-ShareAlike 4.0. Preserve both source records and resolve the applicable obligations before distributing derived audio or models.

## Optional SLR52 source

The intended corpus is OpenSLR SLR52, mirrored at `Ransaka/SinhalaASR` revision `bd1d968241e7edf8ce1577f569f59aac5c0f6b37`. The mirror metadata has 155,970 utterance IDs and 478 anonymized speakers; its original train/test split repeats speakers across partitions. The mirror card describes CC BY-SA 4.0. Retain the source citation, license text, archive and metadata hashes, attribution decision, and processing permission with the private experiment record.

The researcher reports that the complete archive and its integrity, duration, duplicate, lineage, and split audit have passed. The artifacts visible in this workspace do not establish that result: `data/manifests/slr52_v1/speaker_grouped_manifest_summary.json` is a provisional metadata-only, row-balanced split, and the full archive is not present locally. It has 124,740/15,615/15,615 train/dev/test rows and 384/47/47 speakers. Supply the actual frozen split and audit evidence for a headline benchmark run.

For the final experiment record, retain source/archive SHA-256, metadata SHA-256, split manifest SHA-256, utterance and speaker counts, train/dev/test hours, sample-rate/channel/duration distributions, exclusions, and versioned reference normalization. Document dialect, domain, noise, and demographic coverage only where measured. Record ID and acoustic duplicate checks and exact transcript overlap. Repeated-prompt and text-novel results are separate conditions.

`tools/prepare_slr52_training_manifest.py` checks the metadata-to-split join, unique IDs, speaker separation, and train/dev FLAC headers. It does not perform a full audio decode, near-duplicate acoustic search, duration-balanced split audit, or test-audio validation. Keep those audit results separately. The training manifest and all audio stay outside Git. Its test rows contain neither paths nor references; freeze all tuning on train/dev before opening the sealed test.
