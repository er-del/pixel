# Train A 1B Balchand AI Model

This guide explains how to create a new `1b` Balchand AI model using the code that exists in this repository today.

The `1b` preset is a 1B-class architecture. Its current approximate parameter count is about `1.198B`.

## Before You Start

Run all commands from the `balchand-ai/` folder:

```powershell
cd "C:\Users\Lenovo\OneDrive\Desktop\Documents\LLM_MOdel\balchand-ai"
```

Important current limitation:

- Balchand AI currently uses one shared tokenizer path: `tokenizer/pixel_tokenizer`
- If you replace that tokenizer, older checkpoints that were trained with the previous tokenizer can stop working
- Your current working `pixel_100m` checkpoint expects a tokenizer vocab of `1262`

If you want to keep your current `pixel_100m` checkpoint usable, back up these files before starting a new tokenizer run:

```powershell
Copy-Item tokenizer\pixel_tokenizer.model tokenizer\pixel_tokenizer.model.bak
Copy-Item tokenizer\pixel_tokenizer.vocab tokenizer\pixel_tokenizer.vocab.bak
```

## Step 1. Install Dependencies

```bash
python setup.py
python -c "import sentencepiece; print(sentencepiece.__version__)"
```

The second command must print a version number. If it fails, stop and fix the environment first.

## Step 2. Prepare Training Data

Balchand AI supports:

- `.txt`
- `.jsonl` with a `text` field
- `.parquet` with a `text` column

Examples:

```bash
python train.py --size 1b --data data/my_corpus.txt
python train.py --size 1b --data data/my_corpus.jsonl
python train.py --size 1b --data data/my_corpus.parquet
```

For real 1B training, use a large clean corpus. The bootstrap demo corpus is only for smoke testing and local checks.

## Step 3. Decide Whether To Reuse Or Replace The Tokenizer

### Reuse the current tokenizer

Use this if you want to keep compatibility with the current `pixel_100m` checkpoint:

```bash
python tokenizer/manager.py --output-prefix tokenizer/pixel_tokenizer --validate-only
```

If validation succeeds, you can train the `1b` model with the same tokenizer.

### Replace the tokenizer for a fresh 1B run

Use this only if you are intentionally starting a new tokenizer + checkpoint family:

```bash
python tokenizer/manager.py --output-prefix tokenizer/pixel_tokenizer --input data/my_corpus.txt --vocab-size 4096
```

Notes:

- The current training entrypoint auto-loads `tokenizer/pixel_tokenizer`
- The current code caps tokenizer creation to `4096` tokens during `train.py`
- If you replace the tokenizer, old checkpoints trained with the previous tokenizer may stop working

## Step 4. Start 1B Training

Minimal command:

```bash
python train.py --size 1b --data data/my_corpus.txt
```

Recommended explicit output directory:

```bash
python train.py --size 1b --data data/my_corpus.txt --output checkpoints/pixel_1b_run01
```

Short test run first:

```bash
python train.py --size 1b --data data/my_corpus.txt --output checkpoints/pixel_1b_test --steps 10
```

What this does:

- builds the `pixel_1b` architecture
- loads the shared tokenizer from `tokenizer/pixel_tokenizer`
- adjusts model vocab size to the tokenizer vocab
- writes checkpoints into the output directory you choose

## Step 5. Wait For A Checkpoint

After training starts, check for:

```bash
checkpoints/pixel_1b_run01/latest.pt
```

That file is the easiest checkpoint to use for inference.

## Step 6. Test Inference

```bash
python infer.py --model checkpoints/pixel_1b_run01/latest.pt --prompt "Explain local language model training in simple terms."
```

The output JSON should show:

- `"used_checkpoint": true`
- `"model_name": "pixel_1b"` or the resolved checkpoint model metadata for your run

## Step 7. Test In The Browser

```bash
python web/app.py
```

Open `http://127.0.0.1:8000`.

The UI should show:

- active checkpoint path
- resolved model config
- whether the checkpoint metadata overrode the preset selector

## Recommended Safe Workflow

Use this order if you want fewer surprises:

1. Run a short test training job with `--steps 10`
2. Verify `latest.pt` exists
3. Run `python infer.py --model ...`
4. Only then start a longer training run

## What To Avoid

- Do not overwrite `tokenizer/pixel_tokenizer` unless you intend to start a new checkpoint family
- Do not assume `python train.py --size 1b` alone creates a good model from small demo data
- Do not mix a checkpoint from one tokenizer run with a different tokenizer

## Current Reality Of This Repository

This repo can launch and checkpoint a 1B-class model, but a high-quality 1B model still depends on:

- enough GPU memory
- enough training time
- enough clean text data
- a stable tokenizer/checkpoint pairing

The command surface is ready. The main practical limit is compute and dataset quality, not the CLI itself.
