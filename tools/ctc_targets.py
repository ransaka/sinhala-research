"""Train-only, reproducible CTC text targets shared by every SLR52 run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import sentencepiece as spm
from sinlib import Tokenizer


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


class CTCTargets:
    def __init__(self, kind: str, vocab: dict[str, int], processor=None):
        self.kind, self.vocab, self.processor = kind, vocab, processor
        self.inverse = {index: unit for unit, index in vocab.items()}
        if sorted(vocab.values()) != list(range(len(vocab))) or vocab.get("<blank>") != 0:
            raise ValueError("CTC vocabulary must be contiguous with blank at ID 0")

    def units(self, text: str) -> list[str]:
        if self.kind == "codepoint":
            return ["|" if char == " " else char for char in text]
        if self.kind == "sinlib":
            return self.processor.tokenize(text)
        return [self.processor.id_to_piece(index) for index in self.processor.encode(text, out_type=int)]

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(unit, 1) for unit in self.units(text)]

    def decode(self, frame_ids: list[int]) -> str:
        collapsed = []
        previous = None
        for index in frame_ids:
            if index != previous and index != 0:
                collapsed.append(index)
            previous = index
        return self.decode_units(collapsed)

    def decode_units(self, ids: list[int]) -> str:
        units = [self.inverse.get(index, "<unk>") for index in ids]
        if self.kind == "sentencepiece":
            sentencepiece_ids = [self.processor.piece_to_id(unit) for unit in units]
            if 1 in ids:
                return "�".join(self.processor.decode(group)
                                 for group in _groups_without_unknown(sentencepiece_ids))
            return self.processor.decode(sentencepiece_ids)
        return "".join(" " if unit == "|" else "�" if unit == "<unk>" else unit for unit in units)

    def audit(self, rows: list[dict], name: str, *, strict: bool = False) -> dict:
        unknowns = 0
        unknown_rows = 0
        mismatches = []
        lengths = []
        for row in rows:
            ids = self.encode(row["text"])
            row_unknowns = ids.count(1)
            unknowns += row_unknowns
            unknown_rows += bool(row_unknowns)
            lengths.append(len(ids))
            if self.decode_units(ids) != row["text"]:
                mismatches.append(row["id"])
        result = {"rows": len(rows), "unknown_units": unknowns,
                  "unknown_rows": unknown_rows,
                  "unknown_unit_rate": unknowns / sum(lengths) if sum(lengths) else 0.0,
                  "roundtrip_mismatches": len(mismatches),
                  "example_mismatch_ids": mismatches[:10],
                  "target_length_min": min(lengths, default=None),
                  "target_length_p50": _percentile(lengths, 0.50),
                  "target_length_p90": _percentile(lengths, 0.90),
                  "target_length_p99": _percentile(lengths, 0.99),
                  "target_length_max": max(lengths, default=None),
                  "target_length_mean": sum(lengths) / len(lengths) if lengths else None}
        if strict and (unknowns or mismatches):
            raise ValueError(f"{name} target audit failed: {result}")
        return result


def _groups_without_unknown(ids: list[int]) -> list[list[int]]:
    groups = [[]]
    for index in ids:
        if index == 0:
            groups.append([])
        else:
            groups[-1].append(index)
    return groups


def build_targets(kind: str, train: list[dict], output: Path, sp_vocab_size: int) -> tuple[CTCTargets, dict]:
    target_dir = output / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    texts = [row["text"] for row in train]
    if kind == "codepoint":
        units = sorted({"|" if char == " " else char for text in texts for char in text})
        processor = None
    elif kind == "sinlib":
        processor = Tokenizer()
        processor.train(texts)
        # sinlib 0.3.2 assigns learned IDs by iterating a set. Canonicalize
        # those IDs so identical training text yields identical saved files.
        special_tokens = processor.all_special_tokens
        units = sorted(set(processor.vocab_map) - set(special_tokens))
        processor.vocab_map = {
            token: index for index, token in enumerate([*special_tokens, *units])
        }
        processor.token_id_to_token_map = {
            index: token for token, index in processor.vocab_map.items()
        }
        sinlib_dir = target_dir / "sinlib"
        processor.save_pretrained(str(sinlib_dir))
    elif kind == "sentencepiece":
        corpus = target_dir / "train_text.txt"
        corpus.write_text("\n".join(texts) + "\n", encoding="utf-8")
        prefix = str(target_dir / "spm")
        spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix,
            vocab_size=sp_vocab_size, model_type="unigram", character_coverage=1.0,
            normalization_rule_name="identity", add_dummy_prefix=False,
            remove_extra_whitespaces=False, bos_id=-1, eos_id=-1, pad_id=-1,
            unk_id=0, hard_vocab_limit=False, num_threads=1)
        processor = spm.SentencePieceProcessor(model_file=prefix + ".model")
        units = [processor.id_to_piece(index) for index in range(1, processor.vocab_size())]
        corpus.unlink()
        (target_dir / "spm.vocab").unlink()
    else:
        raise ValueError(f"Unknown target type: {kind}")
    vocab = {"<blank>": 0, "<unk>": 1}
    for unit in units:
        if unit in vocab:
            raise ValueError(f"Reserved token appears in target units: {unit}")
        vocab[unit] = len(vocab)
    (target_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = CTCTargets(kind, vocab, processor)
    meta = {"type": kind, "vocab_size": len(vocab),
            "vocab_sha256": _digest(target_dir / "vocab.json"),
            "inventory_source": "selected_training_transcripts_only",
            "training_transcript_count": len(texts),
            "training_text_sha256": hashlib.sha256(
                json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()}
    if kind == "sentencepiece":
        meta["spm_model_sha256"] = _digest(target_dir / "spm.model")
        meta["sentencepiece_version"] = importlib.metadata.version("sentencepiece")
        meta["sentencepiece_config"] = {
            "model_type": "unigram", "requested_vocab_size": sp_vocab_size,
            "actual_vocab_size": processor.vocab_size(),
            "character_coverage": 1.0, "normalization_rule_name": "identity",
            "add_dummy_prefix": False, "remove_extra_whitespaces": False,
            "bos_id": -1, "eos_id": -1, "pad_id": -1, "unk_id": 0,
            "hard_vocab_limit": False, "num_threads": 1,
        }
    if kind == "sinlib":
        meta["sinlib_version"] = importlib.metadata.version("sinlib")
        meta["sinlib_training"] = "Tokenizer.train(selected_training_transcripts); sorted learned IDs after training"
        meta["sinlib_segmentation"] = "Tokenizer.tokenize; segmentation is fixed by sinlib 0.3.2"
        meta["sinlib_special_tokens_excluded"] = processor.all_special_tokens
        meta["sinlib_vocab_sha256"] = _digest(target_dir / "sinlib" / "vocab.json")
        meta["sinlib_config_sha256"] = _digest(target_dir / "sinlib" / "config.json")
    (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if kind == "sinlib":
        # Exercise the saved artifact, rather than trusting the in-memory fit.
        result, meta = load_targets(output)
        for text in texts:
            if result.decode_units(result.encode(text)) != text:
                raise ValueError("Saved sinlib tokenizer cannot round-trip training text")
    return result, meta


def load_targets(output: Path) -> tuple[CTCTargets, dict]:
    target_dir = output / "target"
    meta = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    vocab_path = target_dir / "vocab.json"
    if _digest(vocab_path) != meta["vocab_sha256"]:
        raise ValueError("Saved target vocabulary changed")
    kind = meta["type"]
    if kind == "sinlib" and importlib.metadata.version("sinlib") != meta["sinlib_version"]:
        raise ValueError("Installed sinlib version differs from the saved run")
    if kind == "sentencepiece" and importlib.metadata.version("sentencepiece") != meta["sentencepiece_version"]:
        raise ValueError("Installed SentencePiece version differs from the saved run")
    if kind == "sentencepiece":
        model_path = target_dir / "spm.model"
        if _digest(model_path) != meta["spm_model_sha256"]:
            raise ValueError("Saved SentencePiece model changed")
        processor = spm.SentencePieceProcessor(model_file=str(model_path))
    elif kind == "sinlib":
        if "sinlib_vocab_sha256" in meta:
            sinlib_dir = target_dir / "sinlib"
            for filename, key in (("vocab.json", "sinlib_vocab_sha256"),
                                  ("config.json", "sinlib_config_sha256")):
                if _digest(sinlib_dir / filename) != meta[key]:
                    raise ValueError(f"Saved sinlib tokenizer {filename} changed")
            processor = Tokenizer.from_pretrained(str(sinlib_dir))
            expected_units = set(processor.vocab_map) - set(processor.all_special_tokens)
            saved_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            if expected_units != set(saved_vocab) - {"<blank>", "<unk>"}:
                raise ValueError("Saved sinlib tokenizer and CTC vocabulary differ")
        else:
            # Runs made before corpus-trained sinlib artifacts used only the
            # fixed segmenter and a saved CTC vocabulary.
            processor = Tokenizer()
    elif kind == "codepoint":
        processor = None
    else:
        raise ValueError(f"Unknown saved target type: {kind}")
    return CTCTargets(kind, json.loads(vocab_path.read_text(encoding="utf-8")), processor), meta
