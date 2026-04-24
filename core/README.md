# core

This folder is the runtime spine of Balchand AI. It centralizes device detection, checkpoint handling, and shared request/response data shapes used by CLI and web surfaces.

## When You Use This Folder

- You need to detect whether Balchand AI should run on CPU, CUDA, or MPS.
- You need to save, load, or discover checkpoints in a stable format.
- You need shared response models for scripts and API responses.

## Key Files

- `runtime.py`: hardware profile detection and health payloads
- `checkpoint.py`: save/load/latest checkpoint helpers and vocab resolution from checkpoint weights
- `types.py`: typed request/response dataclasses

## Related Commands

- `python setup.py` uses runtime detection.
- `python train.py` uses runtime + checkpoint managers.
- `python infer.py` and `python web/app.py` use runtime + shared response types.
