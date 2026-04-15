"""
tests/test_checkpoint_and_data.py

Regression tests for checkpoint inspection, inference compatibility, and
dataset normalization. These tests cover checkpoint metadata resolution so
corrupt vocab fields do not break tokenizer/model alignment during inference.
"""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from configs.registry import get_preset
from core.checkpoint import (
    CheckpointInspection,
    CheckpointManager,
    resolve_inference_vocab_size,
)
from inference.generator import PixelGenerator
from models.transformer import PixelForCausalLM
from tokenizer.manager import ensure_tokenizer
from training.data import TokenDataset, TokenDatasetConfig, normalize_corpus


class StubTokenizer:
    """Small tokenizer stub used to validate generator setup behavior."""

    def __init__(self, vocab_size: int):
        """Create a stub tokenizer with a fixed vocabulary size."""
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        """Return the fixed tokenizer vocabulary size."""
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        """Return the BOS token id."""
        return 0

    @property
    def eos_id(self) -> int:
        """Return the EOS token id."""
        return 1

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text into deterministic token ids for testing."""
        tokens = [2, 3]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids into a simple fixed string for testing."""
        return "x"


@pytest.fixture(scope="module")
def pixel_100m_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, CheckpointInspection]:
    """Create one reusable PIXEL checkpoint with model metadata."""
    checkpoint_dir = tmp_path_factory.mktemp("pixel_checkpoint")
    model_config, training_config = get_preset("100m")
    model_config.vocab_size = 256
    training_config.output_dir = str(checkpoint_dir)
    model = PixelForCausalLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manager = CheckpointManager(checkpoint_dir)
    manager.save(
        step=1,
        model=model,
        optimizer=optimizer,
        scaler=None,
        config=training_config,
        metadata={"model": asdict(model_config)},
    )
    inspection = manager.inspect(device="cpu")
    assert inspection is not None
    return checkpoint_dir / "latest.pt", inspection


def test_normalize_corpus_reads_text_and_jsonl(tmp_path: Path) -> None:
    text_path = tmp_path / "demo.txt"
    jsonl_path = tmp_path / "demo.jsonl"
    text_path.write_text("alpha\nbeta\n", encoding="utf-8")
    jsonl_path.write_text('{"text":"gamma"}\n{"text":"delta"}\n', encoding="utf-8")
    samples = normalize_corpus([str(text_path), str(jsonl_path)])
    assert samples == ["alpha", "beta", "gamma", "delta"]


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model_config, training_config = get_preset("100m")
    model_config.vocab_size = 256
    model = PixelForCausalLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manager = CheckpointManager(tmp_path)
    manager.save(step=1, model=model, optimizer=optimizer, scaler=None, config=training_config)
    restored = PixelForCausalLM(model_config)
    payload = manager.load(restored, optimizer=None, scaler=None, device="cpu")
    assert payload is not None
    assert payload["step"] == 1


def test_checkpoint_inspection_reads_model_metadata(
    pixel_100m_checkpoint: tuple[Path, CheckpointInspection],
) -> None:
    checkpoint_path, inspection = pixel_100m_checkpoint
    assert inspection.path == str(checkpoint_path)
    assert inspection.model_config.name == "pixel_100m"
    assert inspection.model_config.hidden_size == 768
    assert inspection.model_config.vocab_size == 256
    assert inspection.training_config is not None
    assert inspection.training_config.size == "100m"


def test_resolve_inference_vocab_size_corrects_corrupt_checkpoint_metadata() -> None:
    """Inference should prefer the checkpoint state vocab when metadata is corrupt."""
    preset_model_config, _ = get_preset("100m")
    corrupt_model_config, _ = get_preset("100m")
    corrupt_model_config.vocab_size = 1262
    inspection = CheckpointInspection(
        path="checkpoints/pixel_100m/latest.pt",
        step=10,
        model_config=corrupt_model_config,
        training_config=None,
        metadata={"model": asdict(corrupt_model_config)},
        state_vocab_size=16_000,
    )
    resolved_vocab_size = resolve_inference_vocab_size(inspection, preset_model_config)
    assert resolved_vocab_size == 16_000
    assert inspection.model_config.vocab_size == 16_000


def test_resolve_inference_vocab_size_keeps_valid_small_checkpoint_vocab() -> None:
    """Inference should keep a valid small vocab when checkpoint weights confirm it."""
    preset_model_config, _ = get_preset("100m")
    small_model_config, _ = get_preset("100m")
    small_model_config.vocab_size = 1262
    inspection = CheckpointInspection(
        path="checkpoints/pixel_100m/latest.pt",
        step=10,
        model_config=small_model_config,
        training_config=None,
        metadata={"model": asdict(small_model_config)},
        state_vocab_size=1262,
    )
    resolved_vocab_size = resolve_inference_vocab_size(inspection, preset_model_config)
    assert resolved_vocab_size == 1262
    assert inspection.model_config.vocab_size == 1262


def test_generator_uses_checkpoint_model_metadata_over_requested_preset(
    pixel_100m_checkpoint: tuple[Path, CheckpointInspection],
) -> None:
    checkpoint_path, inspection = pixel_100m_checkpoint
    requested_model_config, _ = get_preset("1b")
    generator = PixelGenerator(
        requested_model_config,
        StubTokenizer(inspection.model_config.vocab_size),
        checkpoint_path=str(checkpoint_path),
        checkpoint_info=inspection,
    )
    assert generator.checkpoint_loaded is True
    assert generator.config_source == "checkpoint"
    assert generator.model_config.name == "pixel_100m"
    assert generator.model_config.hidden_size == 768
    assert generator.requested_model_config.hidden_size == 2048


def test_generator_rejects_tokenizer_vocab_mismatch(
    pixel_100m_checkpoint: tuple[Path, CheckpointInspection],
) -> None:
    checkpoint_path, inspection = pixel_100m_checkpoint
    requested_model_config, _ = get_preset("1b")
    with pytest.raises(ValueError, match="Tokenizer and checkpoint are incompatible"):
        PixelGenerator(
            requested_model_config,
            StubTokenizer(64),
            checkpoint_path=str(checkpoint_path),
            checkpoint_info=inspection,
        )


def test_checkpoint_inspection_rejects_missing_model_metadata(tmp_path: Path) -> None:
    model_config, training_config = get_preset("100m")
    model_config.vocab_size = 256
    training_config.output_dir = str(tmp_path)
    model = PixelForCausalLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    manager = CheckpointManager(tmp_path)
    manager.save(step=1, model=model, optimizer=optimizer, scaler=None, config=training_config, metadata={})
    with pytest.raises(ValueError, match="missing metadata.model"):
        manager.inspect(device="cpu")


def test_token_dataset_builds_windows(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nhello pixel\n", encoding="utf-8")
    tokenizer = ensure_tokenizer(model_prefix=str(tmp_path / "tok"), data_paths=[str(corpus)], vocab_size=512)
    dataset = TokenDataset(TokenDatasetConfig(paths=(str(corpus),), sequence_length=8, cache_dir=str(tmp_path / "cache")), tokenizer)
    item = dataset[0]
    assert set(item) == {"input_ids", "labels"}
