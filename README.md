# Balchand AI

Balchand AI (Progressive Intelligent eXecution and Efficient Learning) is a local, modular LLM framework built for simple commands and clear defaults. It supports CPU, single-GPU, and multi-GPU usage with the same entrypoints.

## Before You Start

Run all commands from the `balchand-ai/` directory:

```powershell
cd %USERPROFILE%\balchand-ai
```

## Quick Start (Under 5 Commands)

```bash
python setup.py
python train.py
python infer.py --prompt "Once upon a time"
python run_smoke_test.py
python web/app.py
```

Then open `http://127.0.0.1:8000`.

## Command Reference

| Command | What it does | Safe default behavior |
|---|---|---|
| `python setup.py` | Installs dependencies and prints hardware summary | Does not train or modify checkpoints |
| `python train.py` | Trains a model | Uses `100m` preset + local bootstrap corpus |
| `python infer.py` | Generates text | Uses latest checkpoint if available |
| `python run_smoke_test.py` | Verifies end-to-end system health | Runs a short fixed test flow |
| `python web/app.py` | Starts local FastAPI + browser UI | Runs at `127.0.0.1:8000` |
| `python hf_push.py` | Prepares a Hugging Face model bundle and optionally uploads it | Creates a local export bundle even with no repo id |
| `python scripts/import_legacy_sage.py --legacy-root PATH` | Imports legacy tokenizer/data/checkpoints | Only copies files you request |

## Detailed Usage

### 1. Setup

```bash
python setup.py
```

What you should see:
- Package installation logs
- Device detection (`cpu`, `cuda`, or `mps`)
- Runtime summary fields like `gpu_count` and `dtype`

### 2. Training

Default training:

```bash
python train.py
```

Common variants:

```bash
python train.py --size 1b
python train.py --size 3b --use-moe
python train.py --data data/my_corpus.txt --steps 50
python train.py --mode lora --size 1b
python train.py --mode qlora --size 1b
python train.py --output checkpoints/my_run
```

Arguments:
- `--size {100m,1b,3b,7b}`
- `--data` supports `.txt`, `.jsonl`, `.parquet`
- `--steps` overrides preset step count
- `--mode {pretrain,lora,qlora}`
- `--use-moe` applies to supported larger presets

1B training guide:
- Read [TRAIN_1B.md](/c:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/balchand-ai/training/TRAIN_1B.md) for the exact step-by-step flow.
- Keep in mind that Balchand AI currently uses one shared tokenizer path, so replacing the tokenizer can break older checkpoints.

### 3. Inference

Default inference:

```bash
python infer.py
```

Common variants:

```bash
python infer.py --prompt "Explain gradient checkpointing."
python infer.py --size 1b --max-tokens 128
python infer.py --model checkpoints/pixel_100m/latest.pt --temperature 0 --top-p 1.0
python infer.py --mode code --prompt "Write a Python function that parses CSV."
# Model call from hugging face 
python infer.py --model sage002/balchand-ai --prompt "hi"
```

Arguments:
- `--prompt`
- `--model` checkpoint file or checkpoint directory
- `--size {100m,1b,3b,7b}` is only used when no checkpoint is loaded
- `--max-tokens`
- `--temperature`
- `--top-p`
- `--mode {chat,completion,summarize,code}`

Checkpoint behavior:
- If a checkpoint is present, Balchand AI rebuilds the model from checkpoint metadata.
- If checkpoint metadata disagrees with the checkpoint weight shapes, inference corrects the vocab to match the checkpoint weights before tokenizer/model setup.
- Real checkpoint/tokenizer vocab mismatches still fail fast with a clear error.
- Legacy checkpoints without `metadata.model` are rejected instead of guessed.

### 4. Smoke Test

```bash
python run_smoke_test.py
```

Notes:
- This script currently takes no CLI arguments.
- It validates model init, forward pass, short training, and inference.
- Exit code `0` means pass, `1` means fail.

### 5. Web App

```bash
python web/app.py
```

Key endpoints:
- `GET /api/health`
- `GET /api/models`
- `GET /api/ui-modes`
- `POST /api/generate`
- `POST /api/generate/stream`

### 6. Hugging Face Export

Prepare a Hugging Face-ready bundle locally:

```bash
python hf_push.py
```

Upload the bundle to a model repo:

```bash
python hf_push.py --repo-id your-name/balchand-ai-100m
```

This script exports only Hugging Face model assets:
- checkpoint file
- tokenizer files
- generated model card
- exported config JSON files

GitHub should contain the Balchand AI source code and documentation. Hugging Face should contain model artifacts and the model card.

## Project Structure

- `core/`: runtime detection, checkpoint utilities, shared response types
- `tokenizer/`: tokenizer creation, loading, validation, bootstrap text
- `models/`: transformer implementation (RoPE, RMSNorm, attention, LoRA, MoE)
- `training/`: dataset loading and training loop
- `inference/`: token generation and sampling
- `configs/`: typed model/training presets (`100m`, `1b`, `3b`, `7b`)
- `web/`: FastAPI app and HTML UI
- `scripts/`: utility scripts, including legacy import
- `tests/`: targeted automated validation

## Hardware Guidance

- CPU: fully supported with `100m` preset
- Single GPU: supported, mixed precision selected automatically
- Multi-GPU: `train.py` auto-launches distributed training when multiple CUDA GPUs are visible
- Optional acceleration: FlashAttention and bitsandbytes are used only when available

## Troubleshooting

1. `ModuleNotFoundError` for tokenizer/runtime dependencies:
   - Run `python setup.py`.
2. Training is too slow on CPU:
   - Keep `--size 100m` and lower `--steps`.
3. No checkpoint appears in web UI:
   - Run `python train.py` once.
4. Inference warns about random weights:
   - Pass `--model checkpoints/.../latest.pt`.
5. Legacy import copied nothing:
   - Re-check `--legacy-root` and include explicit copy flags.
6. Web/API generation fails with tokenizer decode errors:
   - Run `python setup.py` to ensure `sentencepiece` is installed.
   - If you must run without `sentencepiece`, Balchand AI now falls back to a byte tokenizer automatically.
7. Inference fails after changing `--size`:
   - `--size` does not override checkpoint architecture anymore.
   - Load the correct checkpoint with `--model`, or remove the incompatible checkpoint.
8. Hugging Face inference decodes to replacement characters like `�`:
   - Balchand AI now trusts checkpoint weight shapes over checkpoint metadata when those values disagree.
   - If the issue remains while metadata, weights, and tokenizer all agree, the output is likely model quality rather than a tokenizer mismatch.
