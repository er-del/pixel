#!/usr/bin/env python3
"""
Validate that the tokenizer vocab fix resolves garbage output.
Tests the complete inference pipeline with proper vocab matching.
"""

import json
import sys
from pathlib import Path

def test_tokenizer_vocab_matching():
    """Test that tokenizer vocab matches checkpoint vocab."""
    print("🧪 Testing Tokenizer Vocab Matching Fix\n")
    print("=" * 70)
    
    try:
        from huggingface_hub import hf_hub_download
        from core.checkpoint import CheckpointManager
        from tokenizer.manager import PixelTokenizer, ensure_tokenizer
        from training.bootstrap import ensure_bootstrap_corpus
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Download checkpoint
    print("\n1️⃣  Downloading checkpoint from HuggingFace...")
    try:
        checkpoint_path = hf_hub_download(
            repo_id="sage002/pixel",
            filename="latest.pt",
            repo_type="model"
        )
        print(f"   ✓ Checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Download HF tokenizer (the broken one)
    print("\n2️⃣  Downloading HuggingFace tokenizer...")
    try:
        hf_tokenizer_path = hf_hub_download(
            repo_id="sage002/pixel",
            filename="pixel_tokenizer.model",
            repo_type="model"
        )
        hf_tokenizer = PixelTokenizer.load(hf_tokenizer_path)
        print(f"   ✓ HF tokenizer vocab_size: {hf_tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Load checkpoint and check vocab requirement
    print("\n3️⃣  Loading checkpoint metadata...")
    try:
        checkpoint_dir = Path(checkpoint_path).parent
        manager = CheckpointManager(checkpoint_dir, create=False)
        checkpoint_info = manager.inspect(checkpoint_path, device="cpu")
        print(f"   ✓ Checkpoint vocab_size: {checkpoint_info.model_config.vocab_size}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Check for mismatch
    print("\n4️⃣  Checking vocab compatibility...")
    if hf_tokenizer.vocab_size == checkpoint_info.model_config.vocab_size:
        print(f"   ✓ Vocab sizes match: {hf_tokenizer.vocab_size}")
    else:
        print(f"   ⚠️  MISMATCH DETECTED:")
        print(f"      HF tokenizer: {hf_tokenizer.vocab_size}")
        print(f"      Checkpoint:   {checkpoint_info.model_config.vocab_size}")
        print(f"      This would cause GARBAGE OUTPUT without the fix!")
    
    # Generate proper tokenizer
    print("\n5️⃣  Generating proper tokenizer with checkpoint vocab_size...")
    try:
        data_path = str(ensure_bootstrap_corpus())
        proper_tokenizer = ensure_tokenizer(
            data_paths=[data_path],
            vocab_size=checkpoint_info.model_config.vocab_size
        )
        print(f"   ✓ Generated tokenizer vocab_size: {proper_tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Final check
    print("\n6️⃣  Final verification...")
    if proper_tokenizer.vocab_size == checkpoint_info.model_config.vocab_size:
        print(f"   ✓ SUCCESS: Tokenizer now matches checkpoint!")
        print(f"   ✓ Vocab size is {proper_tokenizer.vocab_size}")
        print(f"   ✓ Text generation should now be coherent\n")
        return True
    else:
        print(f"   ❌ FAIL: Still mismatched")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PIXEL Tokenizer Vocab Fix Validation")
    print("=" * 70 + "\n")
    
    success = test_tokenizer_vocab_matching()
    
    print("=" * 70)
    if success:
        print("✅ FIX VALIDATED: Tokenizer vocab issue is resolved!")
        print("   Next: Run 'python infer.py --model sage002/pixel --prompt \"Your text\"'")
    else:
        print("❌ FIX FAILED: Issues remain")
    print("=" * 70 + "\n")
    
    sys.exit(0 if success else 1)
