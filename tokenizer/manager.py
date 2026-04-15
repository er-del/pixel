"""SentencePiece tokenizer management for PIXEL."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - optional dependency
    spm = None

from tokenizer.bootstrap_text import BOOTSTRAP_TEXTS


@dataclass(slots=True)
class PixelTokenizer:
    """Wrap a SentencePiece tokenizer with simple encode/decode helpers."""

    model_path: str
    processor: Any

    @classmethod
    def load(cls, model_path: str | Path) -> "PixelTokenizer":
        """Load a tokenizer model from disk."""
        source = Path(model_path)
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer file was not found: {source}")
        payload = source.read_bytes()
        is_json_tokenizer = _looks_like_json_payload(payload)
        if source.suffix == ".model" and not is_json_tokenizer:
            if spm is not None:
                processor = spm.SentencePieceProcessor()
                processor.load(str(source))
                return cls(model_path=str(source), processor=processor)
            warnings.warn(
                (
                    "sentencepiece is unavailable but a binary tokenizer model was found. "
                    "Falling back to a byte tokenizer; install sentencepiece for full fidelity."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            processor = SimpleTokenizerProcessor()
            return cls(model_path=str(source), processor=processor)
        processor = SimpleTokenizerProcessor.from_path(source, payload=payload)
        return cls(model_path=str(source), processor=processor)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return int(self.processor.vocab_size())

    @property
    def bos_id(self) -> int:
        """Return the BOS token id."""
        return int(self.processor.bos_id())

    @property
    def eos_id(self) -> int:
        """Return the EOS token id."""
        return int(self.processor.eos_id())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into token ids."""
        pieces = list(self.processor.encode(text, out_type=int))
        if add_bos and self.bos_id >= 0:
            pieces.insert(0, self.bos_id)
        if add_eos and self.eos_id >= 0:
            pieces.append(self.eos_id)
        return pieces

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids into text."""
        return str(self.processor.decode(token_ids))


def ensure_tokenizer(
    model_prefix: str = "tokenizer/pixel_tokenizer",
    data_paths: Iterable[str] | None = None,
    vocab_size: int = 2048,
) -> PixelTokenizer:
    """Ensure a tokenizer exists, training one when needed."""
    prefix = Path(model_prefix)
    model_path = prefix.with_suffix(".model")
    if model_path.exists():
        existing_tokenizer = PixelTokenizer.load(model_path)
        if existing_tokenizer.vocab_size == vocab_size:
            return existing_tokenizer
        else:
            print(f"Existing tokenizer vocab_size ({existing_tokenizer.vocab_size}) != required ({vocab_size})")
            print(f" Deleting old tokenizer and retraining with vocab_size={vocab_size}...")
            model_path.unlink()  # Delete the old tokenizer
    corpus_path = prefix.parent / "bootstrap_corpus.txt"
    write_training_text(corpus_path, data_paths=data_paths)
    train_sentencepiece(corpus_path, prefix, vocab_size=vocab_size)
    validate_tokenizer(model_path)
    return PixelTokenizer.load(model_path)


def write_training_text(output_path: str | Path, data_paths: Iterable[str] | None = None) -> Path:
    """Write plain-text training data for SentencePiece."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        if data_paths:
            for path in data_paths:
                source = Path(path)
                if source.suffix.lower() == ".jsonl":
                    for raw_line in source.read_text(encoding="utf-8").splitlines():
                        if not raw_line.strip():
                            continue
                        payload = json.loads(raw_line)
                        text = payload.get("text", "")
                        if isinstance(text, str) and text.strip():
                            handle.write(text.strip() + "\n")
                else:
                    for line in source.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            handle.write(line.strip() + "\n")
        else:
            for text in BOOTSTRAP_TEXTS:
                handle.write(text + "\n")
    return target


def train_sentencepiece(input_path: str | Path, model_prefix: str | Path, vocab_size: int = 2048) -> None:
    """Train a byte-fallback SentencePiece BPE model."""
    if spm is None:
        SimpleTokenizerProcessor.train(input_path, Path(model_prefix).with_suffix(".model"), vocab_size=vocab_size)
        return
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3,
        byte_fallback=True,
        character_coverage=0.9995,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        split_digits=False,
        hard_vocab_limit=False,
    )


def validate_tokenizer(model_path: str | Path) -> None:
    """Validate round-trip behavior for a trained tokenizer."""
    tokenizer = PixelTokenizer.load(model_path)
    samples = [
        "Hello world",
        "def add(a, b): return a + b",
        "English हिन्दी العربية 中文",
    ]
    for sample in samples:
        if tokenizer.decode(tokenizer.encode(sample)) != sample:
            raise AssertionError(f"Tokenizer validation failed for sample: {sample}")


def build_argparser() -> argparse.ArgumentParser:
    """Build the tokenizer CLI."""
    parser = argparse.ArgumentParser(description="Train or validate the PIXEL tokenizer.")
    parser.add_argument("--output-prefix", default="tokenizer/pixel_tokenizer", help="SentencePiece model prefix.")
    parser.add_argument("--input", nargs="*", default=None, help="Optional text or JSONL inputs used for tokenizer training.")
    parser.add_argument("--vocab-size", type=int, default=2048, help="Tokenizer vocabulary size.")
    parser.add_argument("--validate-only", action="store_true", help="Validate an existing tokenizer without retraining it.")
    return parser


def main() -> None:
    """CLI entrypoint for tokenizer management."""
    args = build_argparser().parse_args()
    model_path = Path(args.output_prefix).with_suffix(".model")
    if args.validate_only:
        validate_tokenizer(model_path)
        print(f"Tokenizer valid: {model_path}")
        return
    tokenizer = ensure_tokenizer(args.output_prefix, data_paths=args.input, vocab_size=args.vocab_size)
    print(f"Tokenizer ready: {tokenizer.model_path} (vocab={tokenizer.vocab_size})")


class SimpleTokenizerProcessor:
    """Fallback character tokenizer used when sentencepiece is unavailable."""

    def __init__(self):
        """Create the fallback processor from a fixed byte vocabulary."""
        self._special = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3}
        self._vocab_size = 260

    @classmethod
    def train(cls, input_path: str | Path, output_path: str | Path, vocab_size: int = 2048) -> None:
        """Save a fixed byte-level tokenizer description."""
        Path(output_path).write_text(json.dumps({"type": "simple-byte", "vocab_size": 260}), encoding="utf-8")

    @classmethod
    def from_path(cls, model_path: str | Path, payload: bytes | None = None) -> "SimpleTokenizerProcessor":
        """Load the fallback tokenizer from disk."""
        raw = payload if payload is not None else Path(model_path).read_bytes()
        processor = cls()
        if _looks_like_json_payload(raw):
            metadata = json.loads(raw.decode("utf-8"))
            vocab_size = metadata.get("vocab_size")
            if isinstance(vocab_size, int) and vocab_size > 0:
                processor._vocab_size = vocab_size
        return processor

    def vocab_size(self) -> int:
        """Return the size of the fallback vocabulary."""
        return self._vocab_size

    def bos_id(self) -> int:
        """Return the BOS token id."""
        return self._special["<bos>"]

    def eos_id(self) -> int:
        """Return the EOS token id."""
        return self._special["<eos>"]

    def encode(self, text: str, out_type=int) -> list[int]:
        """Encode text into byte ids."""
        return [byte + 4 for byte in text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        """Decode byte ids back into text."""
        ignored = {self.bos_id(), self.eos_id(), self._special["<pad>"]}
        payload = bytes(max(index - 4, 0) for index in token_ids if index not in ignored)
        return payload.decode("utf-8", errors="ignore")


def _looks_like_json_payload(payload: bytes) -> bool:
    """Return True when tokenizer bytes appear to be JSON metadata."""
    return bool(payload.lstrip()[:1] == b"{")


if __name__ == "__main__":
    main()
