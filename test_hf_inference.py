#!/usr/bin/env python3
"""
Quick test to verify HuggingFace model inference works correctly.
Tests that tokenizer and checkpoint are compatible.
"""

from pathlib import Path
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)

from tokenizer.manager import PixelTokenizer
from core.checkpoint import CheckpointManager

def test_hf_model_download():
    """Test downloading and loading model from HuggingFace."""
    print("🧪 Testing HuggingFace model download and inference setup...\n")
    
    repo_id = "sage002/pixel"
    
    # Download checkpoint
    print(f"1️⃣  Downloading checkpoint from {repo_id}...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename="latest.pt",
            repo_type="model"
        )
        print(f"   ✓ Checkpoint: {checkpoint_path}\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return False
    
    # Download tokenizer
    print(f"2️⃣  Downloading tokenizer from {repo_id}...")
    try:
        tokenizer_path = hf_hub_download(
            repo_id=repo_id,
            filename="pixel_tokenizer.model",
            repo_type="model"
        )
        print(f"   ✓ Tokenizer: {tokenizer_path}\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return False
    
    # Load and inspect checkpoint
    print("3️⃣  Loading checkpoint metadata...")
    try:
        checkpoint_dir = Path(checkpoint_path).parent
        manager = CheckpointManager(checkpoint_dir, create=False)
        checkpoint_info = manager.inspect(checkpoint_path, device="cpu")
        print(f"   ✓ Model vocab_size: {checkpoint_info.model_config.vocab_size}")
        print(f"   ✓ Checkpoint step: {checkpoint_info.step}\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return False
    
    # Load and inspect tokenizer
    print("4️⃣  Loading tokenizer...")
    try:
        tokenizer = PixelTokenizer.load(tokenizer_path)
        print(f"   ✓ Tokenizer vocab_size: {tokenizer.vocab_size}")
        print(f"   ✓ BOS ID: {tokenizer.bos_id}")
        print(f"   ✓ EOS ID: {tokenizer.eos_id}\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return False
    
    # Check compatibility
    print("5️⃣  Checking compatibility...")
    if checkpoint_info.model_config.vocab_size == tokenizer.vocab_size:
        print(f"   ✓ Vocab sizes match: {tokenizer.vocab_size}\n")
    else:
        print(f"   ❌ MISMATCH! Model vocab: {checkpoint_info.model_config.vocab_size}, Tokenizer vocab: {tokenizer.vocab_size}\n")
        return False
    
    # Test encoding
    print("6️⃣  Testing tokenizer encoding...")
    try:
        test_prompt = "Hello, world!"
        tokens = tokenizer.encode(test_prompt, add_bos=True)
        print(f"   ✓ Encoded '{test_prompt}' to {len(tokens)} tokens")
        print(f"   ✓ Token IDs: {tokens}\n")
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
        return False
    
    print("✅ All checks passed! Model is ready for inference.")
    return True

if __name__ == "__main__":
    success = test_hf_model_download()
    sys.exit(0 if success else 1)
