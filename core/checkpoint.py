"""
core/checkpoint.py

Checkpoint save, load, inspection, and inference compatibility helpers for
PIXEL. This module keeps checkpoint metadata handling centralized so CLI and
web inference resolve the same model configuration from the same checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from configs.base import ModelConfig, TrainingConfig


@dataclass(slots=True)
class CheckpointInspection:
    """Describe one checkpoint's metadata without constructing a model."""

    path: str
    step: int
    model_config: ModelConfig
    training_config: TrainingConfig | None
    metadata: dict[str, Any]
    state_vocab_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint details for CLI and web responses."""
        return {
            "path": self.path,
            "step": self.step,
            "model": asdict(self.model_config),
            "training": asdict(self.training_config) if self.training_config is not None else None,
            "metadata": self.metadata,
            "state_vocab_size": self.state_vocab_size,
        }


def resolve_inference_vocab_size(
    checkpoint_info: CheckpointInspection | None,
    preset_model_config: ModelConfig,
) -> int:
    """Resolve the tokenizer vocab size for inference.

    When a checkpoint is available, inference normally trusts the saved model
    metadata. Some imported Hugging Face checkpoints, however, have been
    observed to carry obviously corrupt small vocabulary sizes in metadata
    despite storing full PIXEL embedding tables. In that narrow case, prefer
    the preset vocabulary so tokenizer generation and model construction stay
    aligned with the real checkpoint weights.

    Args:
        checkpoint_info: Inspected checkpoint metadata, if available.
        preset_model_config: Requested PIXEL preset used as the compatibility
            baseline when no checkpoint is loaded or when checkpoint metadata
            is clearly corrupt.

    Returns:
        The vocabulary size that inference should use for tokenizer and model
        construction.
    """
    if checkpoint_info is None:
        return min(preset_model_config.vocab_size, 4096)
    checkpoint_vocab = checkpoint_info.model_config.vocab_size
    state_vocab = checkpoint_info.state_vocab_size
    if state_vocab is not None and state_vocab != checkpoint_vocab:
        checkpoint_info.model_config.vocab_size = state_vocab
        return state_vocab
    return checkpoint_vocab


def _infer_state_vocab_size(state_dict: dict[str, Any]) -> int | None:
    """Infer the model vocabulary size from checkpoint tensor shapes."""
    vocab_sizes: set[int] = set()
    for key in ("embed_tokens.weight", "model.embed_tokens.weight", "lm_head.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            vocab_sizes.add(int(tensor.shape[0]))
    if not vocab_sizes:
        return None
    return max(vocab_sizes)


class CheckpointManager:
    """Manage PIXEL checkpoints in a simple and resumable format."""

    def __init__(self, root: str | Path, create: bool = True):
        """Create a checkpoint manager rooted at one directory."""
        self.root = Path(root)
        if create:
            self.root.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, step: int) -> Path:
        """Return the filename for a numbered checkpoint."""
        return self.root / f"step_{step:07d}.pt"

    def _resolve_target(self, path: str | Path | None = None, require_explicit: bool = False) -> Path | None:
        """Resolve a checkpoint file path from a file, directory, or latest pointer."""
        if path is None:
            return self.latest()
        target = Path(path)
        if target.is_dir():
            manager = CheckpointManager(target, create=False)
            resolved = manager.latest()
            if resolved is not None:
                return resolved
            if require_explicit:
                raise FileNotFoundError(f"No checkpoint file was found in directory: {target}")
            return None
        if target.exists():
            return target
        if require_explicit:
            raise FileNotFoundError(f"Checkpoint file was not found: {target}")
        return None

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
        config: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write one checkpoint file and update the `latest.pt` pointer."""
        payload = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": asdict(config) if hasattr(config, "__dataclass_fields__") else config,
            "metadata": metadata or {},
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        target = self.checkpoint_path(step)
        torch.save(payload, target)
        latest = self.root / "latest.pt"
        torch.save(payload, latest)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"latest": latest.name, "step": step}, indent=2), encoding="utf-8")
        return target

    def latest(self) -> Path | None:
        """Return the latest checkpoint path if one exists."""
        latest = self.root / "latest.pt"
        if latest.exists():
            return latest
        numbered = sorted(self.root.glob("step_*.pt"))
        return numbered[-1] if numbered else None

    def inspect(
        self,
        path: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> CheckpointInspection | None:
        """Read checkpoint metadata and reconstruct the saved PIXEL model config."""
        target = self._resolve_target(path=path, require_explicit=path is not None)
        if target is None:
            return None
        payload = torch.load(target, map_location=device)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Checkpoint {target} does not contain PIXEL metadata and cannot be used for inference."
            )
        raw_model = metadata.get("model")
        if not isinstance(raw_model, dict):
            raise ValueError(
                f"Checkpoint {target} is missing metadata.model and cannot be matched to a PIXEL architecture."
            )
        try:
            model_config = ModelConfig.from_dict(raw_model)
        except Exception as exc:
            raise ValueError(
                f"Checkpoint {target} contains malformed model metadata and cannot be loaded."
            ) from exc
        training_payload = payload.get("config")
        training_config = (
            TrainingConfig.from_dict(training_payload)
            if isinstance(training_payload, dict)
            else None
        )
        step = payload.get("step", 0)
        state_payload = payload.get("model")
        state_vocab_size = (
            _infer_state_vocab_size(state_payload)
            if isinstance(state_payload, dict)
            else None
        )
        return CheckpointInspection(
            path=str(target),
            step=int(step) if isinstance(step, int) else 0,
            model_config=model_config,
            training_config=training_config,
            metadata=metadata,
            state_vocab_size=state_vocab_size,
        )

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scaler: torch.amp.GradScaler | None = None,
        path: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any] | None:
        """Load a checkpoint into the provided model and optional optimizer."""
        target = self._resolve_target(path=path, require_explicit=path is not None)
        if target is None or not target.exists():
            return None
        payload = torch.load(target, map_location=device)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        torch.set_rng_state(payload["rng_cpu"])
        if payload.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        return payload
