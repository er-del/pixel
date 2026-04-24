# models

This folder contains the decoder-only transformer used by Balchand AI.

## What Is Implemented

- RoPE positional embeddings
- RMSNorm pre-norm transformer blocks
- Grouped-query attention with FlashAttention fallback to PyTorch SDPA
- SwiGLU feed-forward layers
- Optional LoRA adapters
- Optional top-k MoE blocks

## Key Files

- `transformer.py`: `PixelForCausalLM` model class
- `block.py`: transformer block composition
- `attention.py`: attention implementation and KV handling
- `rope.py`: rotary embedding cache and application
- `lora.py`: LoRA module injection
- `moe.py`: optional expert routing blocks
- `norms.py`: RMSNorm

## Related Commands

- `python train.py --mode pretrain`
- `python train.py --mode lora`
- `python train.py --size 3b --use-moe`
- `python infer.py --size 1b`
