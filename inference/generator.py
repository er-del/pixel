"""
inference/generator.py

Inference and sampling helpers for PIXEL. The generator is responsible for
loading one resolved model configuration, validating tokenizer compatibility,
and producing token streams for the CLI and web entrypoints.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import torch

from configs.base import ModelConfig
from core.checkpoint import CheckpointInspection, CheckpointManager
from core.runtime import RuntimeManager
from core.types import GenerationRequest, GenerationResponse
from models.transformer import PixelForCausalLM
from tokenizer.manager import PixelTokenizer
from utils.text import clean_prompt


class PixelGenerator:
    """Load a PIXEL model and generate text from prompts."""

    def __init__(
        self,
        model_config: ModelConfig,
        tokenizer: PixelTokenizer,
        checkpoint_path: str | None = None,
        checkpoint_info: CheckpointInspection | None = None,
    ):
        """Create a generation helper around one model configuration."""
        self.requested_model_config = model_config
        self.tokenizer = tokenizer
        self.runtime = RuntimeManager()
        self.hardware = self.runtime.detect_hardware()
        self.device = self.runtime.build_device(self.hardware)
        manager = None
        if checkpoint_path:
            checkpoint_target = Path(checkpoint_path)
            manager = CheckpointManager(
                checkpoint_target.parent if checkpoint_target.suffix == ".pt" else checkpoint_target,
                create=False,
            )
        self.checkpoint_info = checkpoint_info or (
            manager.inspect(path=checkpoint_path, device="cpu") if manager is not None else None
        )
        self.config_source = "checkpoint" if self.checkpoint_info is not None else "preset"
        self.active_checkpoint_path = self.checkpoint_info.path if self.checkpoint_info is not None else None
        resolved_model_config = (
            self.checkpoint_info.model_config if self.checkpoint_info is not None else model_config
        )
        if self.checkpoint_info is not None and tokenizer.vocab_size != resolved_model_config.vocab_size:
            raise ValueError(
                "Tokenizer and checkpoint are incompatible: "
                f"tokenizer vocab_size={tokenizer.vocab_size}, "
                f"checkpoint vocab_size={resolved_model_config.vocab_size}."
            )
        self.model_config = resolved_model_config
        self.model = PixelForCausalLM(self.model_config).to(self.device)
        self.checkpoint_loaded = False
        if checkpoint_path and manager is not None:
            payload = manager.load(self.model, path=checkpoint_path, device=self.device)
            self.checkpoint_loaded = payload is not None
        self.model.eval()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text and return the final response object."""
        tokens = list(self.stream(request))
        output = "".join(tokens)
        return GenerationResponse(
            output=output,
            tokens_generated=len(tokens),
            model_name=self.model_config.name,
            used_checkpoint=self.checkpoint_loaded,
        )

    def stream(self, request: GenerationRequest) -> Iterator[str]:
        """Yield generated text chunks token by token."""
        prompt = clean_prompt(request.prompt)
        input_ids = self.tokenizer.encode(prompt, add_bos=True)

        if not input_ids:
            print(f"WARNING: Prompt encoded to empty token list. Prompt: {prompt[:50]}...")
            return

        generated = input_ids[:]
        past = None

        with torch.inference_mode():
            for step in range(request.max_tokens):
                window = generated[-self.model_config.context_length :]
                tensor_ids = torch.tensor([window if past is None else [window[-1]]], dtype=torch.long, device=self.device)

                try:
                    output = self.model(tensor_ids, past_key_values=past)
                except Exception as e:
                    print(f"WARNING: Error during model forward pass: {e}")
                    break

                past = output.past_key_values
                next_token = self._sample_next(output.logits[:, -1, :], request.temperature, request.top_p)

                if step == 0 and next_token == self.tokenizer.eos_id:
                    print("WARNING: EOS token generated immediately. Check tokenizer/model compatibility.")

                if next_token == self.tokenizer.eos_id:
                    if step == 0:
                        print("INFO: Generation stopped at step 0 (EOS token)")
                    break

                generated.append(next_token)
                decoded = self.tokenizer.decode([next_token])

                if decoded:
                    yield decoded

    def _sample_next(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        """Sample one token from the model logits."""
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: Invalid logits detected (NaN or Inf). Using argmax instead of sampling.")
            return int(torch.argmax(logits, dim=-1).item())

        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)

        if torch.isnan(probs).any() or probs.sum() <= 0:
            print("WARNING: Invalid probabilities after softmax. Using argmax.")
            return int(torch.argmax(logits, dim=-1).item())

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        
        if sorted_probs.sum() <= 0:
            print("WARNING: No valid probabilities after top-p filtering. Using argmax.")
            return int(torch.argmax(logits, dim=-1).item())

        next_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_sorted)
        return int(next_token.item())

    def describe(self) -> dict[str, object]:
        """Return a compact generator summary."""
        return {
            "requested_model": asdict(self.requested_model_config),
            "resolved_model": asdict(self.model_config),
            "config_source": self.config_source,
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_path": self.active_checkpoint_path,
            "checkpoint_step": self.checkpoint_info.step if self.checkpoint_info is not None else None,
            "hardware": self.hardware.to_dict(),
        }
