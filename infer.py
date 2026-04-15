"""
infer.py

Flat command-line inference entrypoint for PIXEL. This script resolves the
tokenizer and checkpoint pair, corrects obviously corrupt checkpoint vocab
metadata when needed, and prints one JSON payload describing the generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from configs.registry import get_preset, list_presets
from core.checkpoint import (
    CheckpointInspection,
    CheckpointManager,
    resolve_inference_vocab_size,
)
from core.types import GenerationRequest
from inference.generator import PixelGenerator
from tokenizer.manager import ensure_tokenizer, PixelTokenizer
from training.bootstrap import ensure_bootstrap_corpus

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def build_argparser() -> argparse.ArgumentParser:
    """Build the PIXEL inference CLI."""
    parser = argparse.ArgumentParser(description="Run inference with a PIXEL model.")
    parser.add_argument("--prompt", default="Write a short paragraph about reliable local AI tooling.", help="Prompt text to generate from.")
    parser.add_argument("--model", default=None, help="Checkpoint file, directory, or HuggingFace Hub model ID (e.g., sage002/pixel).")
    parser.add_argument(
        "--size",
        default="100m",
        choices=list_presets(),
        help="Model size preset used only when no compatible checkpoint is loaded.",
    )
    parser.add_argument("--max-tokens", type=int, default=96, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling cutoff.")
    parser.add_argument("--mode", default="chat", choices=["chat", "completion", "summarize", "code"], help="Inference UI mode.")
    return parser


def _is_hf_model_id(model_str: str) -> bool:
    """Check if string is a HuggingFace Hub model ID (format: org/repo)."""
    if not model_str or "/" not in model_str:
        return False
    # If it's already a local path, return False
    if Path(model_str).exists():
        return False
    # HuggingFace IDs have exactly one or two slashes
    parts = model_str.split("/")
    return len(parts) in (2, 3)


def _download_hf_model(hf_model_id: str) -> tuple[str, str]:
    """Download model checkpoint and tokenizer from HuggingFace Hub.

    Returns:
        A tuple of `(checkpoint_path, tokenizer_path)`.
    """
    if hf_hub_download is None:
        raise ImportError(
            f"HuggingFace Hub model ID detected ({hf_model_id}), "
            "but huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        )

    print(f"Downloading PIXEL model from HuggingFace Hub: {hf_model_id}")

    checkpoint_path = hf_hub_download(
        repo_id=hf_model_id,
        filename="latest.pt",
        repo_type="model"
    )
    print(f"  OK checkpoint: {checkpoint_path}")

    tokenizer_path = hf_hub_download(
        repo_id=hf_model_id,
        filename="pixel_tokenizer.model",
        repo_type="model"
    )
    print(f"  OK tokenizer: {tokenizer_path}")

    return str(checkpoint_path), str(tokenizer_path)


def _latest_checkpoint() -> str | None:
    """Find the newest PIXEL checkpoint file, if any exist."""
    candidates = sorted(Path("checkpoints").glob("*/latest.pt"))
    if not candidates:
        return None
    return str(candidates[-1])


def _inspect_checkpoint(checkpoint: str | None) -> CheckpointInspection | None:
    """Inspect a checkpoint file or directory when one is available."""
    if checkpoint is None:
        return None
    target = Path(checkpoint)
    manager = CheckpointManager(target.parent if target.suffix == ".pt" else target, create=False)
    return manager.inspect(path=checkpoint, device="cpu")


def main() -> None:
    """Run PIXEL inference from the command line."""
    args = build_argparser().parse_args()
    model_config, _ = get_preset(args.size)
    checkpoint = args.model
    hf_tokenizer_path = None
    if checkpoint and _is_hf_model_id(checkpoint):
        checkpoint, hf_tokenizer_path = _download_hf_model(checkpoint)

    checkpoint = checkpoint or _latest_checkpoint()
    checkpoint_info = _inspect_checkpoint(checkpoint)
    data_path = str(ensure_bootstrap_corpus())

    checkpoint_vocab = (
        checkpoint_info.model_config.vocab_size if checkpoint_info is not None else None
    )
    checkpoint_state_vocab = (
        checkpoint_info.state_vocab_size if checkpoint_info is not None else None
    )
    required_vocab_size = resolve_inference_vocab_size(checkpoint_info, model_config)
    if checkpoint_vocab is not None and checkpoint_vocab != required_vocab_size:
        print("WARNING: Checkpoint metadata vocab_size does not match checkpoint weights.")
        print(f"         Metadata: vocab_size={checkpoint_vocab}")
        if checkpoint_state_vocab is not None:
            print(f"         Weights:  vocab_size={checkpoint_state_vocab}")
        print(f"         Using checkpoint weight vocab_size={required_vocab_size}")

    print(f"Using vocab_size={required_vocab_size} for tokenizer")

    if hf_tokenizer_path:
        hf_tokenizer_temp = PixelTokenizer.load(hf_tokenizer_path)
        print(f"HuggingFace tokenizer vocab_size={hf_tokenizer_temp.vocab_size}")

        if hf_tokenizer_temp.vocab_size == required_vocab_size:
            tokenizer = hf_tokenizer_temp
            print(
                "Using HuggingFace tokenizer "
                f"(vocab match: {hf_tokenizer_temp.vocab_size})"
            )
        else:
            local_tokenizer_path = Path(__file__).parent / "tokenizer" / "pixel_tokenizer.model"
            if local_tokenizer_path.exists():
                try:
                    local_tok = PixelTokenizer.load(str(local_tokenizer_path))
                    print(
                        "WARNING: HF tokenizer vocab "
                        f"({hf_tokenizer_temp.vocab_size}) != "
                        f"checkpoint requires ({required_vocab_size})"
                    )
                    if local_tok.vocab_size == required_vocab_size:
                        print(
                            "   Local tokenizer matches. "
                            f"Using vocab_size={local_tok.vocab_size}"
                        )
                        tokenizer = local_tok
                    else:
                        print(
                            f"   Local tokenizer vocab ({local_tok.vocab_size}) "
                            "also doesn't match"
                        )
                        print(
                            "   Generating new tokenizer "
                            f"with vocab_size={required_vocab_size}..."
                        )
                        tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=required_vocab_size)
                        print(f"   OK generated tokenizer with vocab_size={tokenizer.vocab_size}")
                except Exception as e:
                    print(f"   Could not load local tokenizer: {e}")
                    print(
                        "   Generating new tokenizer "
                        f"with vocab_size={required_vocab_size}..."
                    )
                    tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=required_vocab_size)
                    print(f"   OK generated tokenizer with vocab_size={tokenizer.vocab_size}")
            else:
                print(
                    "WARNING: HF tokenizer vocab "
                    f"({hf_tokenizer_temp.vocab_size}) != "
                    f"checkpoint requires ({required_vocab_size})"
                )
                print(
                    "   Generating new tokenizer "
                    f"with vocab_size={required_vocab_size}..."
                )
                tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=required_vocab_size)
                print(f"   OK generated tokenizer with vocab_size={tokenizer.vocab_size}")
    else:
        tokenizer = ensure_tokenizer(data_paths=[data_path], vocab_size=required_vocab_size)

    if checkpoint_info is None:
        model_config.vocab_size = tokenizer.vocab_size
    generator = PixelGenerator(
        model_config,
        tokenizer,
        checkpoint_path=checkpoint,
        checkpoint_info=checkpoint_info,
    )
    request = GenerationRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        mode=args.mode,
    )
    response = generator.generate(request)
    payload = asdict(response)
    payload.update(generator.describe())
    if not response.used_checkpoint:
        payload["warning"] = "No trained checkpoint was loaded. Output comes from randomly initialized weights."
    elif response.tokens_generated == 0:
        payload["warning"] = "No tokens were generated. Check prompt encoding, tokenizer compatibility, or model state."
    elif generator.config_source == "checkpoint":
        payload["note"] = (
            f"Checkpoint metadata overrode the requested preset '{args.size}'."
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
