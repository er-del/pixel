# tokenizer

This folder manages tokenization in Balchand AI. It trains and loads tokenizers, validates round-trip behavior, and provides a fallback tokenizer when `sentencepiece` is unavailable.

## When You Use This Folder

- You want to create a tokenizer from local text or JSONL data.
- You want to validate tokenizer behavior before training.
- You want zero-argument flows to auto-create tokenizer assets.

## Key Files

- `manager.py`: tokenizer load/train/validate logic
- `bootstrap_text.py`: built-in corpus used for default local setup

## Related Commands

- `python train.py` auto-calls tokenizer setup.
- `python infer.py` auto-calls tokenizer setup.
- `python run_smoke_test.py` validates tokenizer + model path end-to-end.

## Direct Tokenizer Command

```bash
python -m tokenizer.manager --output-prefix tokenizer/pixel_tokenizer --vocab-size 2048
```

Direct script execution also works from the `balchand-ai/` folder:

```bash
python tokenizer/manager.py --output-prefix tokenizer/pixel_tokenizer --vocab-size 2048
```

Validate-only mode:

```bash
python -m tokenizer.manager --output-prefix tokenizer/pixel_tokenizer --validate-only
```
