# training

This folder contains Balchand AI training data preparation and the optimization loop used by `train.py`.

## Supported Data Inputs

- `.txt`
- `.jsonl` with a `text` field
- `.parquet` with a `text` column

## Key Files

- `bootstrap.py`: creates local demo corpus for zero-argument training
- `data.py`: normalizes inputs and caches tokenized windows
- `trainer.py`: optimizer, scheduler, checkpointing, and distributed-aware training

## Related Commands

Default training:

```bash
python train.py
```

Custom data:

```bash
python train.py --data data/my_data.jsonl --steps 40
```

Preset and mode overrides:

```bash
python train.py --size 1b --mode lora
python train.py --size 3b --use-moe
```

## Step-By-Step 1B Guide

For a full `1b` walkthrough, including tokenizer handling and checkpoint safety, read [TRAIN_1B.md](/c:/Users/Lenovo/OneDrive/Desktop/Documents/LLM_MOdel/balchand-ai/training/TRAIN_1B.md).
